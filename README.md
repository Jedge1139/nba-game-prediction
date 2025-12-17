# NBA Game Prediction

Predict NBA game outcomes using historical stats, efficiency metrics, and ensemble machine learning models.

---

## 📖 Project Overview

This project predicts the outcome of NBA games by analyzing team performance trends, efficiency metrics, and historical matchups. It uses multiple machine learning models and combines them into a **Voting Ensemble** to achieve robust and accurate predictions.

**Key Features:**
- Rolling averages for points scored and allowed (last 5 & 10 games)  
- Efficiency metrics (turnover rate, assist rate, rebound rate)  
- Shooting differentials (FG%, FT%, 3P%)  
- Home court advantage  
- Head-to-head matchup history  
- Ensemble of Logistic Regression, Random Forest, and Gradient Boosting models  

---

## 🗂 Project Structure
nba-game-prediction/
│
├── data/ # Sample CSV or placeholder; full dataset must be downloaded from Kaggle
├── src/
│ ├── main.py # Full pipeline with feature engineering, model training, evaluation
│ └── utils.py # Optional helper functions for preprocessing and features
├── notebooks/ # Optional: EDA and feature engineering notebooks
├── figures/ # Optional: plots (feature importance, trends)
├── requirements.txt # Python dependencies
├── README.md
└── .gitignore


---

## 📥 Dataset

- Original dataset: [Kaggle Basketball Dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball)  
- Place `games.csv` in the `data/` folder.  

---

## ⚙️ Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/Jedge1139/nba-game-prediction.git
cd nba-game-prediction
```

Install dependencies:
```
pip install -r requirements.txt
```
Run the project:
```
python src/main.py
```

## 📊 Results

Prints Accuracy and Classification Report to terminal.

Generates Feature Importance plots for Random Forest and Gradient Boosting.

Classification Report:
```
              precision    recall  f1-score   support

           0       0.90      0.76      0.83        34
           1       0.85      0.94      0.90        50

    accuracy                           0.87        84
   macro avg       0.88      0.85      0.86        84
weighted avg       0.87      0.87      0.87        84
```

## 🛠 Models Used

Logistic Regression – linear baseline

Random Forest – captures non-linear interactions

Gradient Boosting – sequential ensemble for improved accuracy

Voting Ensemble – combines all three models for stable predictions

##💡 Feature Engineering

Rolling averages and differentials (points scored, points allowed)

Efficiency metrics: turnover rate, assist rate, rebound rate

Shooting percentages: FG%, FT%, 3P%

Home court advantage (historical home win rate)

Head-to-head win rate between teams

## 📈 Future Improvements

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Cross-validation evaluation

Incorporate advanced stats (e.g., player efficiency ratings)

Build an interactive dashboard (Streamlit / Plotly Dash)

