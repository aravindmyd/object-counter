from sqlalchemy import BigInteger, Column, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

DeclarativeBase = declarative_base()


class Base(DeclarativeBase):
    __abstract__ = True

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(
        name="created_at", type_=DateTime(timezone=True), server_default=func.now()
    )
    expired_at = Column(
        name="expired_at", type_=DateTime(timezone=True), nullable=True, index=True
    )
    updated_at = Column(
        name="updated_at",
        type_=DateTime(timezone=True),
        nullable=True,
        server_onupdate=func.now(),
    )
