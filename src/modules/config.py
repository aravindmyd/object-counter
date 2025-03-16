import os

from src.modules.adapters.count_repo import CountInMemoryRepo, CountMongoDBRepo
from src.modules.adapters.object_detector import FakeObjectDetector, TFSObjectDetector
from src.modules.domain.actions import CountDetectedObjects


def dev_count_action() -> CountDetectedObjects:
    return CountDetectedObjects(FakeObjectDetector(), CountInMemoryRepo())


def prod_count_action() -> CountDetectedObjects:
    tfs_host = os.environ.get("TFS_HOST", "localhost")
    tfs_port = os.environ.get("TFS_PORT", 8501)
    mongo_host = os.environ.get("MONGO_HOST", "localhost")
    mongo_port = os.environ.get("MONGO_PORT", 27017)
    mongo_db = os.environ.get("MONGO_DB", "prod_counter")
    return CountDetectedObjects(
        TFSObjectDetector(tfs_host, tfs_port, "resnet"),
        CountMongoDBRepo(host=mongo_host, port=mongo_port, database=mongo_db),
    )


def get_count_action() -> CountDetectedObjects:
    env = os.environ.get("ENV", "prod")
    count_action_fn = f"{env}_count_action"
    return globals()[count_action_fn]()
