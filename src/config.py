import os


class Config:
    MONGO_URL = "mongodb://localhost:27017/"
    MONGO_HOST = os.environ.get("MONGO_HOST", "localhost")
    MONGO_URL = f"mongodb://{MONGO_HOST}:27017/"
    PRODUCTION_MODEL = os.environ.get("MODEL", "v2")
