# Predicting-the-efficacy-of-PD-1-inhibitor-immunotherapy-for-hepatocellular-carcinoma
import pandas as  pd
import pickle
from sklearn.ensemble import RandomForestClassifier

#Load data

X = pd.read_csv("data.csv",index_col="ID")

#Predicte immunotherapy efficacy

with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
predictions = loaded_model.predict(X)
print(predictions)
