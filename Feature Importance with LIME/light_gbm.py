import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report



def Light_GBM(df):
        X_col = df.iloc[:,[3,5,8,11,12,13,14,15,16,17,18,19,20]]
        X = df.iloc[:,[3,5,8,11,12,13,14,15,16,17,18,19,20]].values
        Y = df.iloc[:,[9]].values

        X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.2, random_state=0) #0,3 or 0.4 test size

        l_GBM = lgb.LGBMClassifier()
        l_GBM.fit(X_train, y_train)

        l_GBM_preds=l_GBM.predict(X_test)
        train_acc = l_GBM.score(X_train, y_train)
        accuracy=accuracy_score(l_GBM_preds, y_test)
        print('Random_Forest Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, l_GBM_preds)))

        scores = cross_val_score(l_GBM, X_train, y_train, cv=10) 
        print("cross val score = ",scores.mean())
        self.results(train_acc,y_test,l_GBM_preds)


        predict_fn_rf = lambda x: l_GBM.predict_proba(x).astype(float)
        local_interpret_with_lime.local_interpretation(predict_fn_rf,X_train,X_test,19,X_col)        
