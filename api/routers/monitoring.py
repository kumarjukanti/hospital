# cat > api/routers/monitoring.py <<'EOF'
from fastapi import APIRouter
from pexpect import EOF

router = APIRouter()


@router.get("/psi")
def get_psi():
    return {
        "status": "ok",
        "message": "PSI monitoring endpoint is available"
    }
EOF