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

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'log_loss', random_state = 42, verbose=2, max_depth=15)
classifier.fit(train_features, train_labels)

y_pred_prob = classifier.predict_proba(test_features)
y_pred = classifier.predict(test_features)
#train_pred = classifier.predict_proba(train_features)


print(y_pred[:5])
print(test_labels.head(5))

ac_score = accuracy_score(test_labels, y_pred)
loss_score = log_loss(test_labels, y_pred_prob)


print(f"TEST ACCURACY: {ac_score}")
print(f"TEST LOG LOSS: {loss_score}")


#ac_score = accuracy_score(train_labels, train_pred)
#print(f"TRAIN ACCURACY: {ac_score}")


#y_pred = classifier.predict_proba(features)
#print(y_pred[:10])

# pred_df = pd.DataFrame(match_id)
# pred_df["results"] = results

# pred_df["home_win"] = y_pred[:,0]
# pred_df["away_win"] = y_pred[:,1]
# pred_df["draw"] = y_pred[:,2]
# print(pred_df.head(1))

