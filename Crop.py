import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import  numpy as np
crop = pd.read_csv("dataset/Crop_recommendation.csv")
num_cols_crop = crop.select_dtypes(include=np.number)

# Visualizing with HeatMap
corr= num_cols_crop.corr()
# Visualizing with Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()