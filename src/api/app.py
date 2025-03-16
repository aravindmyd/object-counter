from fastapi import FastAPI

from .v1.routers import v1_router

app = FastAPI()
API_PREFIX = "/api/v1"
app.include_router(v1_router, prefix=API_PREFIX)


@app.get("/health")
async def root():
    return {"status": "Ok"}
