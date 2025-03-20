from typing import List, Type, Union

from fastapi import Depends, HTTPException
from sqlalchemy import create_engine, func
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, sessionmaker

from src.common.models import Base as BaseModel
from src.common.settings import settings


def get_database_url():
    db_driver = settings.db_driver
    db_host = settings.db_host
    db_port = settings.db_port
    db_user = settings.db_user
    db_password = settings.db_password
    db_name = settings.db_name
    print(
        "url :",
        {f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"},
    )
    return f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


database_url = get_database_url()
engine = create_engine(url=database_url)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Database = Session


# Dependency
def get_database():
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


class DatabaseMixin:
    def __init__(self, db: Database = Depends(get_database, use_cache=True)):
        self.db = db

    def query_existing(self, entity: Union[BaseModel, Type]):
        return self.db.query(entity).filter(entity.expired_at.is_(None))

    def get_by_id(self, entity: Union[BaseModel, Type], entity_id: int):
        try:
            return (
                self.db.query(entity)
                .filter(entity.id == entity_id)
                .filter(entity.expired_at.is_(None))
                .one()
            )
        except NoResultFound:
            raise HTTPException(
                status_code=404, detail=f"No entity found with id: [{entity_id}]"
            )

    def add(self, entity: Union[BaseModel, Type]) -> BaseModel:
        try:
            self.db.add(entity)
            return self.commit_and_refresh(entity)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def add_all(self, entities: List[BaseModel]) -> List[BaseModel]:
        try:
            self.db.add_all(entities)
            self.db.commit()
            return entities
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def delete(self, entity: Union[BaseModel, Type]) -> BaseModel:
        try:
            entity.expired_at = func.now()
            return self.commit_and_refresh(entity)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def commit_and_refresh(self, entity: Union[BaseModel, Type]) -> BaseModel:
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def delete_entities(self, entities: List[BaseModel]):
        for entity in entities:
            entity.expired_at = func.now()
        self.db.commit()
