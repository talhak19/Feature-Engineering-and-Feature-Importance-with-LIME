import datetime
import numpy as np
import pandas as pd


def create_feature_based_on_purchased(df):
        conditions = [
            (df.event == "transaction"),
            (df.event != "transaction")
            ]
    
        values = [1,0]
        df['purchased_or_not'] = np.select(conditions, values)
        return df

def create_feature_based_on_hour(df):

        conditions = [
            (df['islem_anı'].dt.hour >= 0) & (df['islem_anı'].dt.hour <= 6),
            (df['islem_anı'].dt.hour >= 7) & (df['islem_anı'].dt.hour <= 12),
            (df['islem_anı'].dt.hour >= 13) & (df['islem_anı'].dt.hour <= 18),
            (df['islem_anı'].dt.hour >= 19) & (df['islem_anı'].dt.hour <= 24)
            ]

        # create a list of the values we want to assign for each condition
        values = ['night', 'morning', 'afternoon', 'evening']

        # create a new column and use np.select to assign values to it using our lists as arguments
        df['times_of_day'] = np.select(conditions, values)


        return df

#kaydı olan aylar : 5,6,7,8,9
def create_feature_based_on_month(df): 

        conditions = [
            (df['islem_anı'].dt.month == 5) ,
            (df['islem_anı'].dt.month == 6)  ,
            (df['islem_anı'].dt.month == 7)  ,
            (df['islem_anı'].dt.month == 8) ,
            (df['islem_anı'].dt.month == 9)
            ]

        values = ['may', 'june', 'july', 'august','september']

        df['month'] = np.select(conditions, values)
        return df


def create_feature_based_on_weekend(df):

    conditions = [
            (df['islem_anı'].dt.dayofweek> 4),
            (df['islem_anı'].dt.dayofweek<= 4)
            ]
    
    values = [1,0]
    df['is_weekend'] = np.select(conditions, values)
    return df