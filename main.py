
from src.predictor import FootballPredictor

fp = FootballPredictor("2024")
fp.evaluate_save_model()
fp.evaluate_model(fp.model_test_features, fp.model_test_labels)
fp.evaluate_model(fp.test_features, fp.test_labels)
fp.create_predictions()
