import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from features import engineer_features

def train(data_path='../data/raw/Social_Media_Advertising.csv',
          model_path='../model/xgb_model.pkl'):

    df = pd.read_csv(data_path)
    df = engineer_features(df)

    # Leakage-free features — ROI_Clean and Spend_Efficiency excluded
    features = [
        'CTR', 'CTR_vs_Campaign_Avg', 'Engagement_Score', 'Engagement_vs_Avg',
        'Conversion_Rate', 'Acquisition_Cost_Clean', 'Duration_Days',
        'Clicks', 'Impressions', 'Channel_Used', 'Target_Audience',
        'Campaign_Goal', 'Customer_Segment'
    ]

    cat_cols = ['Channel_Used', 'Target_Audience', 'Campaign_Goal', 'Customer_Segment']
    df_model = df[features + ['Fatigued']].dropna()

    le = LabelEncoder()
    for col in cat_cols:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model[features]
    y = df_model['Fatigued']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle imbalance via scale_pos_weight
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=50)

    # Evaluate at threshold 0.3 (accounts for class imbalance)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)

    print("\n--- Classification Report (threshold=0.3) ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()