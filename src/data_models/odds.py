from dataclasses import dataclass

@dataclass
class Odds:
    date: str
    home_team: int
    away_team: int
    home_odds: float
    away_odds: float
    draw_odds: float

    @staticmethod
    def from_mongo_doc(mongo_doc: dict) -> "Odds":
        if "_id" in mongo_doc:
            del mongo_doc["_id"]
            return Odds(**mongo_doc)
