
from src.predictor import FootballPredictor
from src.config import Config as conf
from src.mongoFootballClient import MongoFootballClient
import schedule
from pytz import timezone
from time import sleep

def create_next_predictions():
    mfc = MongoFootballClient(conf.MONGO_URL)
    mfc.delete_next_predictions()
    fp = FootballPredictor(model=conf.PRODUCTION_MODEL, next_games=True)
    fp.create_predictions()

if __name__ == "__main__":
    schedule.every().day.at("01:00", timezone("GMT")).do(create_next_predictions)

    print("Starting scheduled jobs")
    create_next_predictions()
    while True:
        schedule.run_pending()
        sleep(1)
