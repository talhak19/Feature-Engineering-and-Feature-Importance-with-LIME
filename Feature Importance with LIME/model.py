from sklearn import metrics
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import local_interpret_with_lime



class Model:

        def __init__(self,df,model):
                self.df = df
                self.model = model


        def model_executer(self):
                if self.model == "random_forest":
                        self.R_Forest()
                        
                elif self.model == "l_gbm":
                        self.Light_GBM()
                        
                elif self.model == "xgb":
                    self.XGBoost()
                    
                else:
                        print("Gecerli bir model ismi giriniz....(random_forest , l_gbm, XGB)")
        

        def split_values(self):
                X_col = self.df.iloc[:,[3,5,8,11,12,13,14,15,16,17,18,19,20,21]]
                X = self.df.iloc[:,[3,5,8,11,12,13,14,15,16,17,18,19,20,21]].values
                Y = self.df.iloc[:,[9]].values
                X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.2, random_state=0)
                return X_train, X_test,y_train,y_test,X_col
        
        def R_Forest(self):
                X_train,X_test,y_train,y_test,X_col = self.split_values()

                random_forest = RandomForestClassifier(n_estimators=100,criterion="entropy")
                random_forest.fit(X_train, y_train)
                random_forest_preds = random_forest.predict(X_test)
                train_acc = random_forest.score(X_train, y_train)
                accuracy=accuracy_score(random_forest_preds, y_test)
                print("************************RANDOM FOREST************************")
                print(self.model,' accuracy score: {0:0.4f}'.format(accuracy_score(y_test, random_forest_preds)))

                scores = cross_val_score(random_forest, X_train, y_train, cv=10) 
                print("cross val score = ",scores.mean())
                self.results(train_acc,y_test,random_forest_preds)


                predict_fn_rf = lambda x: random_forest.predict_proba(x).astype(float)
                local_interpret_with_lime.local_interpretation(predict_fn_rf,X_train,X_test,5,X_col,y_test)

        def Light_GBM(self):
                X_train,X_test,y_train,y_test,X_col = self.split_values()
                

                l_GBM = lgb.LGBMClassifier( n_estimators=100, learning_rate=0.2)
                l_GBM.fit(X_train, y_train)

                l_GBM_preds = l_GBM.predict(X_test)
                train_acc = l_GBM.score(X_train, y_train)
                accuracy = accuracy_score(l_GBM_preds, y_test)
                print("************************L_GBM************************")
                print(self.model,' accuracy score: {0:0.4f}'.format(accuracy_score(y_test, l_GBM_preds)))

                scores = cross_val_score(l_GBM, X_train, y_train, cv=10) 
                print("cross val score = ",scores.mean())
                self.results(train_acc,y_test,l_GBM_preds)


                predict_fn_rf = lambda x: l_GBM.predict_proba(x).astype(float)
                local_interpret_with_lime.local_interpretation(predict_fn_rf,X_train,X_test,5,X_col,y_test)
                
                
        def XGBoost(self):
                X_train,X_test,y_train,y_test,X_col = self.split_values()
                
                xgbc = XGBClassifier()
                xgbc.fit(X_train, y_train)
                
                xgbc_preds = xgbc.predict(X_test)
                train_acc = xgbc.score(X_train, y_train)
                accuracy=accuracy_score(xgbc_preds, y_test)
                print("************************XGB************************")
                print(self.model,' accuracy score: {0:0.4f}'.format(accuracy_score(y_test, xgbc_preds)))

                scores = cross_val_score(xgbc, X_train, y_train, cv=10) 
                print("cross val score = ",scores.mean())
                self.results(train_acc,y_test,xgbc_preds)


                predict_fn_rf = lambda x: xgbc.predict_proba(x).astype(float)
                local_interpret_with_lime.local_interpretation(predict_fn_rf,X_train,X_test,5,X_col,y_test)


        def results(self,train_acc,test_y,y_pred):
                
                print("The Accuracy for Training Set is {}".format(train_acc*100))
                test_acc = accuracy_score(test_y, y_pred)
                print("The Accuracy for Test Set is {}".format(test_acc*100))
                print(classification_report(test_y, y_pred))
    