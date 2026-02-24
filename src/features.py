import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Campaign_ID', 'Date']).reset_index(drop=True)

    df['CTR'] = df['Clicks'] / df['Impressions'].replace(0, np.nan)

    # Fixed regex strings
    df['Duration_Days'] = df['Duration'].str.extract(r'(\d+)').astype(float)
    df['Acquisition_Cost_Clean'] = df['Acquisition_Cost'].replace(
        r'[\$,]', '', regex=True).astype(float)

    campaign_avg_ctr = df.groupby('Campaign_ID')['CTR'].transform('mean')
    df['CTR_vs_Campaign_Avg'] = df['CTR'] / campaign_avg_ctr.replace(0, np.nan)

    df['Spend_Efficiency'] = df['Conversion_Rate'] / df['Acquisition_Cost_Clean'].replace(0, np.nan)

    campaign_avg_eng = df.groupby('Campaign_ID')['Engagement_Score'].transform('mean')
    df['Engagement_vs_Avg'] = df['Engagement_Score'] / campaign_avg_eng.replace(0, np.nan)

    df['ROI_Clean'] = pd.to_numeric(df['ROI'], errors='coerce')

    # Revised fatigue label — at least 2 of 3 conditions
    roi_threshold = df['ROI_Clean'].quantile(0.35)
    ctr_low        = df['CTR_vs_Campaign_Avg'] < 0.90
    engagement_low = df['Engagement_vs_Avg'] < 0.90
    roi_low        = df['ROI_Clean'] < roi_threshold

    df['Fatigued'] = ((ctr_low.astype(int) +
                       engagement_low.astype(int) +
                       roi_low.astype(int)) >= 2).astype(int)
    return df