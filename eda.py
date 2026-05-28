import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/train.csv');

df.describe();
df.head()  ;
df.info()  ;

# Check for missing values
missing_values = df.isnull().sum();