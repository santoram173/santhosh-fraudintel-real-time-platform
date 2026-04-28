"""Setup script for FraudIntel platform."""
from setuptools import setup, find_packages

setup(
    name="santhosh-fraudintel-real-time-platform",
    version="1.0.0",
    description="Real-Time Fraud Detection and Digital Identity Anomaly System",
    author="Santhosh",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "scikit-learn>=1.4.2",
        "numpy>=1.26.4",
        "xgboost>=2.0.3",
    ],
)
