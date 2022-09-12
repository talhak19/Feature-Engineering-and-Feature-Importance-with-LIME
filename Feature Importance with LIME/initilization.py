
import pandas as pd
import seaborn as sns
import warnings


from pre_process import pre_process
warnings.filterwarnings('ignore')

class RocketRetail:

    def __init__(self,data):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        self.df = pd.read_csv(data)

    def data_info(self):
        print("Rows and Columns number:\n",self.df.shape)
        print("-------------------------------------------------------------------------------------------")
        print("\nDataFrame Head:\n",self.df.head())
        print("-------------------------------------------------------------------------------------------")
        print("\nData info:"),self.df.info()
        print("-------------------------------------------------------------------------------------------")
        print("Missing value Check:\n",self.df.isnull().sum(),"\n")
        print("-------------------------------------------------------------------------------------------")
        #Sayısal değerli sütunların analizi
        print(self.df.describe())
        print("-------------------------------------------------------------------------------------------")
        








