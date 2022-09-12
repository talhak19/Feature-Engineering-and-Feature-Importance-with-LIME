from typing_extensions import Self
import initilization
import pandas as pd
import pre_process
import model
import guess_purchase_model
import new_df_for_transaction

if __name__ == '__main__':
    

    # e_commerce_events = initilization.RocketRetail("./events.csv")
    # e_commerce_events.data_info()
    # first_df=pre_process.time_arrange(e_commerce_events.df)

    # df_atc = new_df_for_transaction.feature_engineering(first_df,"addtocart") #addtocart yapılanlar da purchased edildiği için sadece event olarak addtocart'a bakanlardan da çıkarım yapabiliriz
    # df_atc = new_df_for_transaction.fonks(first_df,"addtocart")
    # df_atc.to_csv(r'C:\Users\talha\OneDrive\Masaüstü\sonhaftalar.csv', index=False)   #Burada dataframe'i oluşturup kaydettim yaklaşık 14 saat sürdü.
    # print(df_atc.head(25))
 
    df_saved = pd.read_csv("./my_data_final.csv")
    df_week_view = pd.read_csv("./sonhaftalar.csv")

    last_df=pre_process.pre_process(df_saved)
    last_df["how_many_view_for_item_in_last_week"] = df_week_view["how_many_view_for_item_in_last_week"]


    
    rf_model= model.Model(last_df,"random_forest")
    rf_model.model_executer()

    print("\n-----------------------------------------------------------------------------\n")
    
    gbm_model= model.Model(last_df,"l_gbm")
    gbm_model.model_executer()

    print("\n-----------------------------------------------------------------------------\n")

    xgb_model = model.Model(last_df,"xgb")
    xgb_model.model_executer()




    #Geometric Distribution
    # last_df = guess_purchase_model.feature_arrangement(last_df)

    # guess_purchase_model.xgb_probs(last_df)