import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load datasets
matches = pd.read_csv('C:/Users/divya/Desktop/python/Cricket prediction/IPL_Matches_2008_2022.csv')
balls = pd.read_csv('C:/Users/divya/Desktop/python/Cricket prediction/IPL_Ball_by_Ball_2008_2022.csv')

# Data processing
inningScores = balls.groupby(['ID', 'innings']).sum()['total_run'].reset_index()
inningScores = inningScores[inningScores['innings'] == 1]
inningScores['target'] = inningScores['total_run'] + 1
matches = matches.merge(inningScores[['ID', 'target']], on='ID')

# Team renaming
def rename_teams(df, col):
    df[col] = df[col].str.replace('Delhi Daredevils', 'Delhi Capitals')
    df[col] = df[col].str.replace('Kings XI Punjab', 'Punjab Kings')
    df[col] = df[col].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    df[col] = df[col].str.replace('Rising Pune Supergiant', 'Pune Warriors')
    df[col] = df[col].str.replace('Rising Pune Supergiants', 'Pune Warriors')
    df[col] = df[col].str.replace('Pune Warriorss', 'Pune Warriors')
    df[col] = df[col].str.replace('Gujarat Lions', 'Gujarat Titans')
    return df

matches = rename_teams(matches, 'Team1')
matches = rename_teams(matches, 'Team2')
matches = rename_teams(matches, 'WinningTeam')

teams2023 = [
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad', 
    'Delhi Capitals', 
    'Chennai Super Kings',
    'Gujarat Titans', 
    'Lucknow Super Giants', 
    'Kolkata Knight Riders',
    'Punjab Kings', 
    'Mumbai Indians'
]

matches = matches[matches['Team1'].isin(teams2023)]
matches = matches[matches['Team2'].isin(teams2023)]
matches = matches[matches['WinningTeam'].isin(teams2023)]
matches = matches[['ID', 'City', 'Team1', 'Team2', 'WinningTeam', 'target']].dropna()

balls = rename_teams(balls, 'BattingTeam')
balls = balls[balls['BattingTeam'].isin(teams2023)]
final = matches.merge(balls, on='ID')
final = final[final['innings'] == 2]
final['current_score'] = final.groupby('ID')['total_run'].cumsum()
final['runs_left'] = np.where(final['target'] - final['current_score'] >= 0, final['target'] - final['current_score'], 0)
final['balls_left'] = np.where(120 - final['overs'] * 6 - final['ballnumber'] >= 0, 120 - final['overs'] * 6 - final['ballnumber'], 0)
final['wickets_left'] = 10 - final.groupby('ID')['isWicketDelivery'].cumsum()
final['current_run_rate'] = (final['current_score'] * 6) / (120 - final['balls_left'])
final['required_run_rate'] = np.where(final['balls_left'] > 0, final['runs_left'] * 6 / final['balls_left'], 0)
def result(row):
    return 1 if row['BattingTeam'] == row['WinningTeam'] else 0
final['result'] = final.apply(result, axis=1)

index1 = final[final['Team2'] == final['BattingTeam']]['Team1'].index
index2 = final[final['Team1'] == final['BattingTeam']]['Team2'].index
final.loc[index1, 'BowlingTeam'] = final.loc[index1, 'Team1']
final.loc[index2, 'BowlingTeam'] = final.loc[index2, 'Team2']

winningPred = final[['BattingTeam', 'BowlingTeam', 'City', 'runs_left', 'balls_left', 'wickets_left', 'current_run_rate', 'required_run_rate', 'target', 'result']]

# Model training
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['BattingTeam', 'BowlingTeam', 'City'])
], remainder='passthrough')

X = winningPred.drop('result', axis=1)
y = winningPred['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', RandomForestClassifier())
])
pipe.fit(X_train, y_train)

# Streamlit app
st.title("IPL Match Win Probability Predictor")

# User inputs
BattingTeam = st.selectbox("Select Batting Team", teams2023)
BowlingTeam = st.selectbox("Select Bowling Team", teams2023)
City = st.selectbox("Select City", matches['City'].unique())
runs_left = st.number_input("Runs Left", min_value=0, value=50)
balls_left = st.number_input("Balls Left", min_value=0, value=60)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, value=6)
current_run_rate = st.number_input("Current Run Rate", min_value=0.0, value=8.0)
required_run_rate = st.number_input("Required Run Rate", min_value=0.0, value=9.0)
target = st.number_input("Target", min_value=0, value=180)

# Prediction
input_data = pd.DataFrame({
    'BattingTeam': [BattingTeam],
    'BowlingTeam': [BowlingTeam],
    'City': [City],
    'runs_left': [runs_left],
    'balls_left': [balls_left],
    'wickets_left': [wickets_left],
    'current_run_rate': [current_run_rate],
    'required_run_rate': [required_run_rate],
    'target': [target]
})

if st.button("Predict"):  
    probability = pipe.predict_proba(input_data)[0][1]
    st.subheader(f"Win Probability: {probability * 100:.2f}%")
