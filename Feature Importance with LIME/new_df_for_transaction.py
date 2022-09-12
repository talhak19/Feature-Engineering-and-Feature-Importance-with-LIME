import pandas as pd
import datetime as dt


def feature_engineering(main_df,event):
    #Bir müsterinin baktıgı iteme son baktıgı günden 1 hafta önceye kadar dahil olan sürede kac kere baktıgı.

    array_for_df= []
    buying_visitors = main_df[main_df.event == event].visitorid.sort_values().unique()

    for index in buying_visitors:           
        v_df = main_df[main_df.visitorid == index]
        
        for i in v_df[v_df["event"]== event].index:
                count = 0
                temp = []
                temp.append(v_df.loc[i].visitorid)
                temp.append(v_df.loc[i].itemid)
                view_anlari = v_df.loc[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'view')].timestamp

                son_view = view_anlari.max()
                son_view_bir_hafta_oncesi = son_view -  dt.timedelta(days=7)

                how_many_view_for_item = v_df.loc[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'view')].event.count()

                for i in view_anlari:
                    if son_view >= i > son_view_bir_hafta_oncesi:
                        count = count + 1

                how_many_view_for_item_in_last_week = count


                # temp.append(v_df.loc[i].visitorid)
                # temp.append(v_df.loc[i].itemid)
                temp.append(how_many_view_for_item)
                temp.append(how_many_view_for_item_in_last_week)

                array_for_df.append(temp)

    
    final_df =pd.DataFrame(array_for_df, columns=["visitor_id","item_id","how_many_view_for_item","how_many_view_for_item_in_last_week"])
    final_df = final_df.drop_duplicates(subset=["visitor_id","item_id"])
    return final_df






def fonks(main_df,event):
    array_for_df= []
    buying_visitors = main_df[main_df.event == event].visitorid.sort_values().unique()

    for index in buying_visitors:           
        v_df = main_df[main_df.visitorid == index]
        
        for i in v_df[v_df["event"]== event].index:
                temp = []

                temp.append(v_df.loc[i].visitorid)
                

                how_many_view_for_item = v_df.loc[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'view')].event.count()
                how_many_atc_for_item  = v_df.loc[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'addtocart')].event.count()
                how_many_transaction_for_item  = v_df.loc[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'transaction') ].event.count()

                
                #Transaction ve addtocart farklı event'ler olduğu için burada event ve saat için event'e özel işlem yapıyoruz
                if (how_many_transaction_for_item > 0):
                    purchased_or_not = 1
                    islem = "transaction"
                    ozellesmis_islem_anı_index = v_df[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'transaction')].index.astype("int64")
                    
                else:
                    purchased_or_not= 0
                    islem= "addtocart"
                    ozellesmis_islem_anı_index = v_df[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'addtocart')].index.astype("int64")

                
                temp.append(islem)


                temp.append(v_df.loc[ozellesmis_islem_anı_index[0]].timestamp)#islem_ani
                temp.append(v_df.loc[ozellesmis_islem_anı_index[0]].timestamp.hour)#saati
                temp.append(v_df.loc[ozellesmis_islem_anı_index[0]].timestamp.minute)#dakikası

                
                
                
                new_df = v_df.loc[(v_df.loc[i].itemid == main_df.itemid) & (main_df.event == 'view')] #visitor'un bu item'a baktığı zaman dilimleri için bir df
                iteme_ilk_bakis = new_df.timestamp.min()
                iteme_son_bakis = new_df.timestamp.max()

                temp.append(how_many_view_for_item)
                temp.append(iteme_ilk_bakis)
                temp.append(iteme_son_bakis)
                temp.append(how_many_atc_for_item)
                temp.append(purchased_or_not)
                temp.append(v_df.loc[i].itemid)
                array_for_df.append(temp)
                
                final_df =pd.DataFrame(array_for_df, columns=['visitorid',"islem","islem_anı","islem_saati","islem_dakikasi","kac_kez_view","ilk_view","son_view","add_to_card_count","purchased","itemid"])
                            
                if (event == "addtocart"):  #biz addtocart count istediğimiz için iki defa satır kaplasın istemiyoruz, visitor o itemid için purchased ettiyse tekrar addtocart yaptıgı eventleri almamak için
                    final_df = final_df.drop_duplicates(subset=['visitorid',"itemid","add_to_card_count"])

    return final_df

