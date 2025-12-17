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
cd nba-game-prediction'''

