"""Identity Graph API Router."""

from fastapi import APIRouter, HTTPException
from backend.models.schemas import IdentityGraphRequest, IdentityRiskResponse
from backend.core.identity_graph import get_graph

router = APIRouter()


@router.post("/analyze", response_model=IdentityRiskResponse)
async def analyze_identity(req: IdentityGraphRequest):
    graph = get_graph()
    result = graph.record_event(req.user_id, req.device_id, req.ip_address, req.action)
    return IdentityRiskResponse(
        user_id=req.user_id,
        device_risk=result["device_risk"],
        ip_risk=result["ip_risk"],
        behavioral_risk=result["behavioral_risk"],
        composite_risk=result["composite_risk"],
        is_new_device=result["is_new_device"],
        is_suspicious_ip=result["is_suspicious_ip"],
        anomaly_signals=result["anomaly_signals"],
    )


@router.get("/graph/{user_id}")
async def get_user_graph(user_id: str):
    graph = get_graph()
    result = graph.get_user_graph(user_id)
    if not result.get("found"):
        raise HTTPException(status_code=404, detail="User not found in identity graph")
    return result


@router.get("/stats")
async def graph_stats():
    graph = get_graph()
    return {
        "total_users": len(graph.users),
        "total_devices": len(graph.devices),
        "total_ips": len(graph.ips),
        "high_risk_devices": sum(
            1 for d in graph.devices.values()
            if d.connection_count("user") > graph.shared_device_threshold
        ),
        "high_risk_ips": sum(
            1 for ip in graph.ips.values()
            if ip.connection_count("user") > graph.shared_ip_threshold
        ),
    }
