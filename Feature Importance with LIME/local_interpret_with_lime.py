import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def local_interpretation(predict_fn_rf,X_train,X_test,index,X_col,Y_test):
    #predict_fn_rf  the predict_proba parameter makes sure it also includes a plot to show which class is the most probable, according to the local surrogate model
    X = X_train

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = X_col.columns,class_names=['Not Purchased',' Purchased'])

    choosen_instance = X_test[index]
    y_test = Y_test[index]
    print("Real test data[X]: ",choosen_instance)
    print("Real test data[Y]: ",y_test)

    exp = lime_explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=12)
    exp.show_in_notebook(show_all=False)
