# src/main.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load CSV
# -----------------------------
CSV_PATH = "data/games.csv"
df = pd.read_csv(CSV_PATH)

# -----------------------------
# 2. Preprocessing
# -----------------------------
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.sort_values('game_date')

# Target variable
df['home_win'] = (df['wl_home'] == 'W').astype(int)

# -----------------------------
# 3. Feature Engineering
# -----------------------------

# Step 1: Rolling averages (5 & 10 games)
windows = [5, 10]
for w in windows:
    df[f'home_avg_pts_last{w}'] = (
        df.groupby('team_id_home')['pts_home']
        .rolling(w).mean().reset_index(0, drop=True)
    )
    df[f'away_avg_pts_last{w}'] = (
        df.groupby('team_id_away')['pts_away']
        .rolling(w).mean().reset_index(0, drop=True)
    )

df['pts_diff_last5'] = df['home_avg_pts_last5'] - df['away_avg_pts_last5']
df['pts_diff_last10'] = df['home_avg_pts_last10'] - df['away_avg_pts_last10']

# Step 2: Defensive metrics
df['home_avg_pts_allowed_last5'] = (
    df.groupby('team_id_home')['pts_away'].rolling(5).mean().reset_index(0, drop=True)
)
df['away_avg_pts_allowed_last5'] = (
    df.groupby('team_id_away')['pts_home'].rolling(5).mean().reset_index(0, drop=True)
)
df['def_pts_diff_last5'] = df['away_avg_pts_allowed_last5'] - df['home_avg_pts_allowed_last5']

# Step 3: Efficiency & possession metrics
df['home_tov_rate'] = df['tov_home'] / df['fga_home']
df['away_tov_rate'] = df['tov_away'] / df['fga_away']
df['home_ast_rate'] = df['ast_home'] / df['fgm_home']
df['away_ast_rate'] = df['ast_away'] / df['fgm_away']
df['home_reb_rate'] = df['reb_home'] / (df['reb_home'] + df['reb_away'])
df['away_reb_rate'] = df['reb_away'] / (df['reb_home'] + df['reb_away'])
df['fg_pct_diff'] = df['fg_pct_home'] - df['fg_pct_away']
df['ft_pct_diff'] = df['ft_pct_home'] - df['ft_pct_away']
df['fg3_pct_diff'] = df['fg3_pct_home'] - df['fg3_pct_away']

# Step 4: Home court advantage
home_win_rate = df.groupby('team_id_home')['home_win'].expanding().mean().reset_index(level=0, drop=True)
df['team_home_win_rate'] = home_win_rate.fillna(df['home_win'].mean())

# Step 5: Head-to-head matchup
df['matchup_key'] = df['team_id_home'].astype(str) + "_" + df['team_id_away'].astype(str)
df['h2h_home_win_rate'] = (
    df.groupby('matchup_key')['home_win'].expanding().mean().reset_index(level=0, drop=True)
)
df['h2h_home_win_rate'] = df['h2h_home_win_rate'].fillna(df['home_win'].mean())

# -----------------------------
# 4. Handle infinite / missing values
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)

features = [
    'home_avg_pts_last5','away_avg_pts_last5','home_avg_pts_last10','away_avg_pts_last10',
    'home_avg_pts_allowed_last5','away_avg_pts_allowed_last5','def_pts_diff_last5',
    'home_tov_rate','away_tov_rate','home_ast_rate','away_ast_rate',
    'home_reb_rate','away_reb_rate',
    'fg_pct_diff','ft_pct_diff','fg3_pct_diff',
    'pts_diff_last5','pts_diff_last10',
    'team_home_win_rate','h2h_home_win_rate'
]

df = df.dropna(subset=features + ['home_win'])

# -----------------------------
# 5. Train / Test Split (by season)
# -----------------------------
latest_season = df['season_id'].max()
train = df[df['season_id'] < latest_season]
test = df[df['season_id'] == latest_season]

X_train = train[features]
y_train = train['home_win']
X_test = test[features]
y_test = test['home_win']

# Optional: scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. Define Base Models
# -----------------------------
lr_model = LogisticRegression(max_iter=5000, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

# -----------------------------
# 7. Create Voting Ensemble
# -----------------------------
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft',
    n_jobs=-1
)

ensemble.fit(X_train_scaled, y_train)
preds = ensemble.predict(X_test_scaled)

# -----------------------------
# 8. Evaluation
# -----------------------------
print("Ensemble Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# -----------------------------
# 9. Feature Importance Plots
# -----------------------------

# Random Forest Feature Importance
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Random Forest Feature Importance")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Gradient Boosting Feature Importance
gb_model.fit(X_train, y_train)
importances_gb = gb_model.feature_importances_
indices_gb = np.argsort(importances_gb)[::-1]

plt.figure(figsize=(12,6))
plt.title("Gradient Boosting Feature Importance")
plt.bar(range(len(features)), importances_gb[indices_gb], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices_gb], rotation=90)
plt.tight_layout()
plt.show()
