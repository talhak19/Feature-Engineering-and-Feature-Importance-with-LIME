import pandas as pd
import datetime
from create_feature import create_feature_based_on_hour, create_feature_based_on_month, create_feature_based_on_weekend, create_feature_based_on_purchased

def time_arrange(df):
    times =[]
    for i in df['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i//1000.0))
      
    df["timestamp"] = times
    return df

def pre_process(df):
    
    df["islem_anı"] = pd.to_datetime(df["islem_anı"])

    # df= create_feature_based_on_purchased(df)
    hour_df = create_feature_based_on_hour(df)
    weekend_df = create_feature_based_on_weekend(hour_df)
    df = create_feature_based_on_month(weekend_df)

    # #perform one-hot encoding on categorical columns 
    one_hot_encoded_data = pd.get_dummies(df, columns = ['times_of_day',"month"])

    return one_hot_encoded_data
