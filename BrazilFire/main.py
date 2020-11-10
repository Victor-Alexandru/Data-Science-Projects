import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from subprocess import check_output
import warnings
warnings.filterwarnings('ignore')

from p_decision_tree.DecisionTree import DecisionTree

def read_dataset(dataset):
    data = pd.read_csv(dataset)
    return data
#descriptive features
X = data[['year']] 
#target feature
Y = data[["Class"]]
data = read_dataset('amazon.csv')

# sns.barplot(x="year", y="number", data=data)
# plt.savefig('fires_brazil_from_1987_2019')

plt.boxplot(data['number'], 0, 'gd')
plt.show()