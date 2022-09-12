from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import geom
import matplotlib.pyplot as plt


def feature_arrangement(df):

    temp = []
    for i in df.index:
        if (pd.isna(df.loc[i,"son_view"])):
            temp.append(np.nan)
        else:
            dt1 = datetime.strptime(df.loc[i,"son_view"], '%Y-%m-%d %H:%M:%S')
            dt2 = datetime.strptime(df.loc[i,"ilk_view"], '%Y-%m-%d %H:%M:%S')


            dt = dt1-dt2
            duration_in_s = dt.total_seconds()
            minutes = divmod(duration_in_s, 60)[0] 

            temp.append(minutes)

    df["view_dakika_farki"] = temp

    return df


def xgb_probs(df):

    X = df[["islem_saati","islem_dakikasi","kac_kez_view","add_to_card_count","is_weekend","how_many_view_for_item_in_last_week","view_dakika_farki"]]
    y = df["purchased"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    xgbc = XGBClassifier(objective='binary:logistic')
    xgbc.fit(X_train, y_train)

    xgbc_preds = xgbc.predict(X_test)


    da = pd.DataFrame({'Actual': y_test, 'Predicted': xgbc_preds})
    # print(da.head(20))




    predicts = xgbc.predict_proba(X_test) #[0][0] kıyas yapıp büyüğü al sonra bunları geometrik dist'e koy ( p bunlar), (k ise günler)



    # accuracy=accuracy_score(xgbc_preds, y_test)
    # print(accuracy)
    geo_dist(predicts,55)


def geo_dist(predicts,index):

    #belirlenen index'in olma olasılığı
    if (predicts[index][0] >predicts[index][1]):
        p = predicts[index][0]
       
    else:
        p = predicts[index][1]
    
    # Calculate geometric probability distribution
    #
    gunler = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    geom_pd = geom.pmf(gunler, p)

    # Plot the probability distribution
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(gunler, geom_pd, 'bo', ms=8, label='geom pmf') # geom pmf function measures probability mass function (pmf) of the distribution.
    plt.ylabel("Probability", fontsize="18")
    plt.xlabel("Gunler", fontsize="18")
    plt.title("Geometric Distribution - Gunler Vs Probability", fontsize="18")
    ax.vlines(gunler, 0, geom_pd, colors='b', lw=5, alpha=0.5)
    plt.show()