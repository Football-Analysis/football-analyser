from dataclasses import dataclass


@dataclass
class Prediction:
    match_id: str
    home_win: float
    away_win: float
    draw: float

    @staticmethod
    def from_mongo_doc(mongo_doc: dict) -> "Prediction":
        if "_id" in mongo_doc:
            del mongo_doc["_id"]
            return Prediction(**mongo_doc)