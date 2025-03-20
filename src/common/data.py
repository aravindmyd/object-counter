from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class RequestModel(BaseModel):
    class Config:
        extra = "forbid"


class ResponseModel(BaseModel):
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%SZ")}
        orm_mode = True
