from io import BytesIO

from fastapi import APIRouter, File, Form, UploadFile

from src.modules import config

router = APIRouter()


count_action = config.get_count_action()


@router.post("/object-count")
async def object_detection(file: UploadFile = File(...), threshold: float = Form(0.5)):
    image = BytesIO(await file.read())
    count_response = count_action.execute(image, threshold)
    return count_response
