
from src.predictor import FootballPredictor

fp = FootballPredictor("2016", "model")
fp.evaluate_save_model()
fp.create_predictions()
