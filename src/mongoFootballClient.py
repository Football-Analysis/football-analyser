from pymongo import MongoClient, ASCENDING
from tqdm import tqdm
from typing import List
from .data_models.match import Match
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

    def get_observations(self):
        observation_df = pd.DataFrame(list(self.observation_collection.find({
            "home_general_5": { "$ne": "N" },
            "away_general_5": { "$ne": "N" }
        })))
        return observation_df


