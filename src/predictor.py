from .mongoFootballClient import MongoFootballClient
from .config import Config as conf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os
import pandas as pd
from .data_models.prediction import Prediction


class FootballPredictor:
    def __init__(self, date=None, model=None):
        self.mfc = MongoFootballClient(conf.MONGO_URL)
        self.raw_training_observations = self.mfc.get_observations(date)
        self.training_engineered_features = self.engineer_features(self.raw_training_observations)
        self.model_training_features, self.model_test_features, \
        self.model_training_labels, self.model_test_labels = self.create_train_test_split(self.training_engineered_features)
 
        if date is not None:
            self.raw_test_features = self.mfc.get_observations(date, match=False)
            self.test_engineered_features = self.engineer_features(self.raw_test_features)
            self.test_match_ids = pd.DataFrame(self.test_engineered_features["match_id"])
            print(self.test_match_ids.head())
            self.test_features = self.test_engineered_features.drop(["result", "_id", "match_id"], axis=1)
        
        if model is None:
            self.create_model()
        else:
            self.load_model(model)
    
    def engineer_features(self, df: pd.DataFrame):
        for i in range(1,6):
            df[f"home_general_{i}"] = df[f"home_general_{i}"].astype('category')
            df[f"home_general_{i}"] = df[f"home_general_{i}"].replace({'W': 2, 'D': 1, 'L': 0})

            df[f"away_general_{i}"] = df[f"away_general_{i}"].astype('category')
            df[f"away_general_{i}"] = df[f"away_general_{i}"].replace({'W': 2, 'D': 1, 'L': 0})

            df[f"home_home_{i}"] = df[f"home_home_{i}"].astype('category')
            df[f"home_home_{i}"] = df[f"home_home_{i}"].replace({'W': 3, 'D': 2, 'L': 0, 'N': 1})

            df[f"away_away_{i}"] = df[f"away_away_{i}"].astype('category')
            df[f"away_away_{i}"] = df[f"away_away_{i}"].replace({'W': 3, 'D': 2, 'L': 0, 'N': 1})
    
        df['result'] = df.result.astype('category')

        return df
    
    def create_train_test_split(self, observations: pd.DataFrame):
        target = observations["result"].replace({'Home Win': 0, 'Away Win': 1, 'Draw': 2})
        features = observations.drop(["result", "_id", "match_id"], axis=1)
        train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                    target,
                                                                                    test_size = 0.20,
                                                                                    random_state = 42)
        return train_features, test_features, train_labels, test_labels
        
    def create_model(self, train_features, train_labels, grid_search=False):
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
        
        self.classifier.fit(train_features, train_labels)


    def evaluate_model(self, features, labels):
        y_pred_prob = self.classifier.predict_proba(features)
        y_pred = self.classifier.predict(labels)

        ac_score = accuracy_score(labels, y_pred)
        loss_score = log_loss(labels, y_pred_prob)

        print(f"TEST ACCURACY: {ac_score}")
        print(f"TEST LOG LOSS: {loss_score}")

    def predict(self, observations: pd.DataFrame, metadata: pd.DataFrame):
        results = self.classifier.predict_proba(observations)

        home_win = results[:,0]
        away_win = results[:,1]
        draw = results[:,2]

        metadata["home_win"] = home_win
        metadata["away_win"] = away_win
        metadata["draw"] = draw

        for _, row in metadata.iterrows():
            self.mfc(row.to_dict())

    def save_model(self):
        save_path = os.path.join(os.path.dirname(__file__), "..", "ml-models", "model.pkl")
        with open(save_path,'wb') as f:
            pickle.dump(self.classifier,f)

    def load_model(self, model_name):
        print(f"Loading model {model_name}")
        load_path = os.path.join(os.path.dirname(__file__), "..", "ml-models", f"{model_name}.pkl")
        with open(load_path, 'rb') as f:
            self.classifier = pickle.load(f)

    def evaluate_save_model(self):
        self.evaluate_model()
        self.save_model()

    def create_predictions(self):
        self.predict(self.test_features, self.test_match_ids)


    