"""
Digital Identity Graph
Models User → Device → IP → Transaction relationships.
Detects anomalies via frequency shifts, new devices, abnormal patterns.
"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class IdentityNode:
    def __init__(self, node_id: str, node_type: str):
        self.node_id = node_id
        self.node_type = node_type
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.connections: Dict[str, set] = defaultdict(set)
        self.event_count = 0
        self.metadata: Dict = {}

    def add_connection(self, target_type: str, target_id: str):
        self.connections[target_type].add(target_id)
        self.last_seen = time.time()
        self.event_count += 1

    def connection_count(self, target_type: str = None) -> int:
        if target_type:
            return len(self.connections.get(target_type, set()))
        return sum(len(v) for v in self.connections.values())


class IdentityGraph:
    """
    Lightweight in-memory identity graph.
    Tracks User → Device → IP → Merchant connections.
    In production: replace with Neo4j or TigerGraph.
    """

    def __init__(self):
        self.users: Dict[str, IdentityNode] = {}
        self.devices: Dict[str, IdentityNode] = {}
        self.ips: Dict[str, IdentityNode] = {}
        self.shared_device_threshold = 5    # > 5 users on same device = suspicious
        self.shared_ip_threshold = 10       # > 10 users on same IP = suspicious

    def record_event(self, user_id: str, device_id: str, ip_address: str, 
                     action: str = "transaction") -> Dict[str, Any]:
        """Record an identity event and update graph."""
        # Upsert nodes
        if user_id not in self.users:
            self.users[user_id] = IdentityNode(user_id, "user")
        if device_id not in self.devices:
            self.devices[device_id] = IdentityNode(device_id, "device")
        if ip_address not in self.ips:
            self.ips[ip_address] = IdentityNode(ip_address, "ip")

        user = self.users[user_id]
        device = self.devices[device_id]
        ip_node = self.ips[ip_address]

        # Build edges
        user.add_connection("device", device_id)
        user.add_connection("ip", ip_address)
        device.add_connection("user", user_id)
        device.add_connection("ip", ip_address)
        ip_node.add_connection("user", user_id)
        ip_node.add_connection("device", device_id)

        return self._analyze_risk(user_id, device_id, ip_address)

    def _analyze_risk(self, user_id: str, device_id: str, ip_address: str) -> Dict[str, Any]:
        user = self.users.get(user_id)
        device = self.devices.get(device_id)
        ip_node = self.ips.get(ip_address)

        anomaly_signals = []
        device_risk = 0.0
        ip_risk = 0.0
        behavioral_risk = 0.0

        # --- Device Risk Analysis ---
        is_new_device = device.event_count <= 2
        users_on_device = device.connection_count("user")
        
        if is_new_device:
            device_risk += 0.3
            anomaly_signals.append("New device detected for this user")
        
        if users_on_device > self.shared_device_threshold:
            device_risk += 0.5
            anomaly_signals.append(f"Device shared by {users_on_device} users (possible shared/compromised device)")

        # --- IP Risk Analysis ---
        users_on_ip = ip_node.connection_count("user")
        devices_on_ip = ip_node.connection_count("device")
        
        if users_on_ip > self.shared_ip_threshold:
            ip_risk += 0.4
            anomaly_signals.append(f"IP used by {users_on_ip} users (data center/proxy suspected)")
        
        if devices_on_ip > 20:
            ip_risk += 0.3
            anomaly_signals.append(f"IP associated with {devices_on_ip} devices")

        # --- Behavioral Analysis ---
        user_devices = user.connection_count("device")
        user_ips = user.connection_count("ip")
        
        if user_devices > 5:
            behavioral_risk += 0.3
            anomaly_signals.append(f"User active on {user_devices} different devices")
        
        if user_ips > 10:
            behavioral_risk += 0.4
            anomaly_signals.append(f"User active from {user_ips} different IPs")

        # Rapid frequency check
        time_since_first = time.time() - user.first_seen
        if time_since_first < 3600 and user.event_count > 10:
            behavioral_risk += 0.5
            anomaly_signals.append("High-frequency activity in short time window")

        composite = min(
            0.4 * min(device_risk, 1.0) + 0.35 * min(ip_risk, 1.0) + 0.25 * min(behavioral_risk, 1.0),
            1.0
        )

        return {
            "device_risk": round(min(device_risk, 1.0), 3),
            "ip_risk": round(min(ip_risk, 1.0), 3),
            "behavioral_risk": round(min(behavioral_risk, 1.0), 3),
            "composite_risk": round(composite, 3),
            "is_new_device": is_new_device,
            "is_suspicious_ip": users_on_ip > self.shared_ip_threshold,
            "anomaly_signals": anomaly_signals,
            "graph_stats": {
                "user_devices": user.connection_count("device"),
                "user_ips": user.connection_count("ip"),
                "users_on_device": users_on_device,
                "users_on_ip": users_on_ip,
            }
        }

    def get_user_graph(self, user_id: str) -> Dict:
        """Get identity graph snapshot for a user."""
        user = self.users.get(user_id)
        if not user:
            return {"user_id": user_id, "found": False}
        
        return {
            "user_id": user_id,
            "found": True,
            "devices": list(user.connections.get("device", set())),
            "ips": list(user.connections.get("ip", set())),
            "event_count": user.event_count,
            "first_seen": user.first_seen,
            "last_seen": user.last_seen,
            "device_details": {
                did: {
                    "users": len(self.devices[did].connections.get("user", set())),
                    "events": self.devices[did].event_count
                }
                for did in user.connections.get("device", set())
                if did in self.devices
            }
        }


# Singleton graph instance
_graph = IdentityGraph()

def get_graph() -> IdentityGraph:
    return _graph
