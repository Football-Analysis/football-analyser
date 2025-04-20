
from src.predictor import FootballPredictor


fp = FootballPredictor(model="production", next_games=True)
#fp = FootballPredictor()
#fp.evaluate_save_model()
#fp.wipe_predictions()
fp.create_predictions()
#fp.evaluate_save_model()

# fp = FootballPredictor(date="2016", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2017", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2018", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2019", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2020", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2021", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2022", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2023", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()

# fp = FootballPredictor(date="2024", grid_search=False)
# #fp.evaluate_save_model()
# fp.create_predictions()
