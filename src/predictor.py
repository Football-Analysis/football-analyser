from .mongoFootballClient import MongoFootballClient
from .config import Config as conf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os


class FootballPredictor:
    def __init__(self):
        self.mfc = MongoFootballClient(conf.MONGO_URL)
        self.raw_observations = self.mfc.get_observations()
        self.classifier = None
    
    def engineer_feature(self):
        for i in range(1,6):
            self.raw_observations[f"home_general_{i}"] = self.raw_observations[f"home_general_{i}"].astype('category')
            self.raw_observations[f"home_general_{i}"] = self.raw_observations[f"home_general_{i}"].replace({'W': 2, 'D': 1, 'L': 0})

            self.raw_observations[f"away_general_{i}"] = self.raw_observations[f"away_general_{i}"].astype('category')
            self.raw_observations[f"away_general_{i}"] = self.raw_observations[f"away_general_{i}"].replace({'W': 2, 'D': 1, 'L': 0})

            self.raw_observations[f"home_home_{i}"] = self.raw_observations[f"home_home_{i}"].astype('category')
            self.raw_observations[f"home_home_{i}"] = self.raw_observations[f"home_home_{i}"].replace({'W': 3, 'D': 2, 'L': 0, 'N': 1})

            self.raw_observations[f"away_away_{i}"] = self.raw_observations[f"away_away_{i}"].astype('category')
            self.raw_observations[f"away_away_{i}"] = self.raw_observations[f"away_away_{i}"].replace({'W': 3, 'D': 2, 'L': 0, 'N': 1})
    
        self.raw_observations['result'] = self.raw_observations.result.astype('category')
        self.target = self.raw_observations["result"].replace({'Home Win': 0, 'Away Win': 1, 'Draw': 2})
        self.features = self.raw_observations.drop(["result", "_id", "match_id"], axis=1)
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(self.features,
                                                                                                        self.target,
                                                                                                        test_size = 0.20,
                                                                                                        random_state = 42)
        
    def create_model(self, grid_search=False):
        if grid_search:
            param_grid = {"n_estimators": [50, 100, 200, 300, 500],
                        "max_features": ["log2", "sqrt"],
                        "max_depth": [5,15,25,35],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 5, 10],
                        'bootstrap': [True, False]}

            rf = RandomForestClassifier()
            rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            rf_random.fit(self.train_features, self.train_labels)
            best_params = rf_random.best_estimator_
            print(f"best params found were - {best_params}")

        self.classifier = RandomForestClassifier(n_estimators=300,
                                                 criterion='log_loss',
                                                 max_features='log2',
                                                 max_depth=15,
                                                 bootstrap=False,
                                                 random_state = 42,
                                                 verbose=2)
        
        self.classifier.fit(self.train_features, self.train_labels)


    def evaluate_model(self):
        y_pred_prob = self.classifier.predict_proba(self.test_features)
        y_pred = self.classifier.predict(self.test_features)

        ac_score = accuracy_score(self.test_labels, y_pred)
        loss_score = log_loss(self.test_labels, y_pred_prob)

        print(f"TEST ACCURACY: {ac_score}")
        print(f"TEST LOG LOSS: {loss_score}")

    def save_model(self):
        save_path = os.path.join(os.path.dirname(__file__), "..", "ml-models", "model.pkl")
        with open(save_path,'wb') as f:
            pickle.dump(self.classifier,f)

    def load_model(self):
        load_path = os.path.join(os.path.dirname(__file__), "..", "ml-models", "model.pkl")
        with open(load_path, 'rb') as f:
            self.classifier = pickle.load(f)

    def create_and_evaluate(self):
        self.engineer_feature()
        self.create_model()
        self.evaluate_model()
        self.save_model()

    def load_and_evaluate(self):
        self.load_model()
        self.evaluate_model()


    