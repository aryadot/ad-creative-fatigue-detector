import pandas as pd
import pickle
import shap

FATIGUE_THRESHOLD = 0.3  # Lower threshold due to class imbalance (6% fatigue rate)

CHANNEL_MAP   = {"Facebook": 0, "Instagram": 1, "Pinterest": 2, "Twitter": 3}
AUDIENCE_MAP  = {"All Ages": 0, "Men 18-24": 1, "Men 25-34": 2, "Men 35-44": 3,
                 "Men 45-60": 4, "Women 18-24": 5, "Women 25-34": 6,
                 "Women 35-44": 7, "Women 45-60": 8}
GOAL_MAP      = {"Brand Awareness": 0, "Increase Sales": 1,
                 "Market Expansion": 2, "Product Launch": 3}
SEGMENT_MAP   = {"Fashion": 0, "Food": 1, "Health": 2, "Home": 3, "Technology": 4}

FEATURES = [
    'CTR', 'CTR_vs_Campaign_Avg', 'Engagement_Score', 'Engagement_vs_Avg',
    'Conversion_Rate', 'Acquisition_Cost_Clean', 'Duration_Days',
    'Clicks', 'Impressions', 'Channel_Used', 'Target_Audience',
    'Campaign_Goal', 'Customer_Segment'
]

def load_model(path='../model/xgb_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_input(clicks, impressions, engagement_score, engagement_vs_avg,
                conversion_rate, acquisition_cost, duration_days,
                ctr_vs_avg, channel, target_audience, campaign_goal, customer_segment):

    ctr = clicks / impressions if impressions > 0 else 0

    return pd.DataFrame([{
        'CTR': ctr,
        'CTR_vs_Campaign_Avg': ctr_vs_avg,
        'Engagement_Score': engagement_score,
        'Engagement_vs_Avg': engagement_vs_avg,
        'Conversion_Rate': conversion_rate,
        'Acquisition_Cost_Clean': acquisition_cost,
        'Duration_Days': duration_days,
        'Clicks': clicks,
        'Impressions': impressions,
        'Channel_Used': CHANNEL_MAP[channel],
        'Target_Audience': AUDIENCE_MAP[target_audience],
        'Campaign_Goal': GOAL_MAP[campaign_goal],
        'Customer_Segment': SEGMENT_MAP[customer_segment]
    }])

def predict(input_df, model):
    prob = model.predict_proba(input_df)[0][1]
    pred = 1 if prob >= FATIGUE_THRESHOLD else 0
    return pred, prob

def explain(input_df, model):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    return shap_vals, explainer.expected_value, input_df