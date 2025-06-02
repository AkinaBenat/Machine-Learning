import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv('Titanic-Dataset.csv')
print(df.head(10))
print(df.shape)
print(df.decribe())
print(df.info())
