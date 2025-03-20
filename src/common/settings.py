from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # database settings
    db_driver: str = "mysql+pymysql"
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = "new-password"
    db_name: str = "object"

    # disk storage settings
    base_documents_path: Path = Path("./documents/")

    # model name
    model_name: str = "resnet"


settings = Settings()
