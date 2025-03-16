from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from counter import config

app = FastAPI()
count_action = config.get_count_action()


@app.post("/object-count")
async def object_detection(file: UploadFile = File(...), threshold: float = Form(0.5)):
    image = BytesIO(await file.read())
    count_response = count_action.execute(image, threshold)
    return count_response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
