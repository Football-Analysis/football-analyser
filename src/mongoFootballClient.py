from pymongo import MongoClient, ASCENDING
from typing import List
from .data_models.prediction import Prediction
from datetime import datetime
import pandas as pd


class MongoFootballClient:
    def __init__(self, url: str):
        self.url = url
        self.mc = MongoClient(self.url)
        self.football_db = self.mc["football"]
        self.match_collection = self.football_db["matches"]
        self.league_collection = self.football_db["leagues"]
        self.observation_collection = self.football_db["observations"]
        self.prediction_collection = self.football_db["predictions"]

    def get_observations(self, date=None, match=True):
        mongo_filter = {
            "home_general_5": { "$ne": "N" },
            "away_general_5": { "$ne": "N" }
        }

        if date is not None and match:
            mongo_filter["match_id"] = {"$regex": date}
        elif date is not None:
            mongo_filter["match_id"] = {"$regex": f"^((?!{date}).)*$"}

        observation_df = pd.DataFrame(list(self.observation_collection.find(mongo_filter)))
        return observation_df
    
    def save_prediction(self, prediction: dict):
        self.prediction_collection.insert_one(prediction)
        return True


