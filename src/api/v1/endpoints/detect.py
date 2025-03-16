from fastapi import APIRouter

router = APIRouter()


@router.get("/detect")
async def detect_objects():
    return {"message": "Detect Objects"}
