import matplotlib.pyplot as plt
def graphics(self,df):
            data = df.event.value_counts()
            labels = data.index
            sizes = data.values
            explode = (0, 0.15, 0.15)  # explode 1st slice
            plt.subplots(figsize=(8,8))
            # Plot
            plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=False, startangle=0)
            plt.title("Event Situations")
            plt.axis('equal')
            plt.show()