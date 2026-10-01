from fastapi import APIRouter

router = APIRouter()

@router.get("/psi")
def get_psi():
    return {
        "status": "ok",
        "message": "PSI monitoring endpoint is available"
    }