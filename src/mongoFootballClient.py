from pymongo import MongoClient, ASCENDING
from typing import List
from .data_models.prediction import Prediction
from datetime import datetime
import pandas as pd
from .data_models.odds import Odds
from .data_models.result import Result


class MongoFootballClient:
    def __init__(self, url: str):
        self.url = url
        self.mc = MongoClient(self.url)
        self.football_db = self.mc["football"]
        self.match_collection = self.football_db["matches"]
        self.league_collection = self.football_db["leagues"]
        self.observation_collection = self.football_db["observations"]
        self.next_observation_collection = self.football_db["next_observations"]
        self.prediction_collection = self.football_db["predictions"]
        self.next_prediction_collection = self.football_db["next_predictions"]
        self.odds_collection = self.football_db["odds"]
        self.team_collection = self.football_db["teams"]

    def get_observations(self, date=None, match=True, next_games=False) -> pd.DataFrame:
        query = {
            "home_general_5": { "$ne": "N" },
            "away_general_5": { "$ne": "N" },
            "before_gw_ten": 0
            }
        if next_games:
            col = self.next_observation_collection
            query["result"] = "N/A"
        else:
            col = self.observation_collection
            query["result"] = {"$ne": "N/A"} 

        if date is not None and match:
            query["match_id"] = {"$regex": date}
        elif date is not None:
            query["match_id"] = {"$regex": f"^((?!{date}).)*$"}

        observation_df = pd.DataFrame(list(col.find(query)))
        return observation_df
    
    def save_prediction(self, prediction: dict, next_games=False) -> bool:
        if next_games:
            col = self.next_prediction_collection
        else:
            col = self.prediction_collection
        col.insert_one(prediction)
        return True
    
    def get_odds(self, year: str) -> List[Odds]:
        raw_odds = self.odds_collection.find({
            "date": {"$regex": year},
            "home_team": {"$type": 16},
            "away_team": {"$type": 16}
        })

        odds = []
        for raw_odd in raw_odds:
            odd = Odds.from_mongo_doc(raw_odd)
            if odd.home_odds != 0 and odd.away_odds != 0 and odd.draw_odds != 0:
                if 0.99 < (1/odd.home_odds) + (1/odd.away_odds) + (1/odd.draw_odds) < 1.01:
                    odds.append(odd)

        return odds
    
    def get_predictions(self, year) -> List[Prediction]:
        predictions = self.prediction_collection.find({"match_id": {"$regex": year}})

        preds_to_return = []
        for prediction in predictions:
            preds_to_return.append(Prediction.from_mongo_doc(prediction))

        return preds_to_return
    
    def get_prediction(self, date, home_team):
        pred = self.prediction_collection.find_one({
            "match_id": f"{date}-{home_team}"
            })
        if pred is not None:
            prediction = Prediction.from_mongo_doc(pred)
        else:
            return None
        return prediction
    
    def get_match_result(self, date, home_team) -> Result:
        matches = self.match_collection.find({
            "date": date,
            "home_team": home_team
        })

        for match in matches:
            if match["result"] in ["Home Win", "Away Win", "Draw"]:
                return Result(match["result"])

        return False
        
    def get_team_from_id(self, team_id: int) -> str:
        team = self.team_collection.find_one({"id": team_id})
        return team["name"]
    
    def delete_next_predictions(self):
        self.next_prediction_collection.delete_many({})
