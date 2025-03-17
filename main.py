from src.mongoFootballClient import MongoFootballClient
from src.config import Config as conf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

mfc = MongoFootballClient(conf.MONGO_URL)

observation_df = mfc.get_observations()

for i in range(1,6):
    observation_df[f"home_general_{i}"] = observation_df[f"home_general_{i}"].astype('category')
    observation_df[f"home_general_{i}"] = observation_df[f"home_general_{i}"].replace({'W': 2, 'D': 1, 'L': 0})

    observation_df[f"away_general_{i}"] = observation_df[f"away_general_{i}"].astype('category')
    observation_df[f"away_general_{i}"] = observation_df[f"away_general_{i}"].replace({'W': 2, 'D': 1, 'L': 0})

    observation_df[f"home_home_{i}"] = observation_df[f"home_home_{i}"].astype('category')
    observation_df[f"home_home_{i}"] = observation_df[f"home_home_{i}"].replace({'W': 3, 'D': 2, 'L': 0, 'N': 1})

    observation_df[f"away_away_{i}"] = observation_df[f"away_away_{i}"].astype('category')
    observation_df[f"away_away_{i}"] = observation_df[f"away_away_{i}"].replace({'W': 3, 'D': 2, 'L': 0, 'N': 1})

observation_df['result'] = observation_df.result.astype('category')
target = observation_df["result"].replace({'Home Win': 0, 'Away Win': 1, 'Draw': 2})

match_id = observation_df["match_id"]
results = observation_df["result"]
features = observation_df.drop(["result", "_id", "match_id"], axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(features, target, test_size = 0.20, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42, verbose=1, class_weight="balanced", max_depth=10)
classifier.fit(train_features, train_labels)

y_pred = classifier.predict(test_features)
y_pred_prob = classifier.predict_proba(test_features)
train_pred = classifier.predict(train_features)

ac_score = accuracy_score(test_labels, y_pred)
loss_score = log_loss(test_labels, y_pred)
print(f"TEST ACCURACY: {ac_score}")
print(f"TEST LOG LOSS: {loss_score}")
