import pandas as pd

df1 = pd.read_csv('all_fake_fakenews_dataset.csv')
df2 = pd.read_csv('all_mix_fakenews_dataset.csv')

print("ccleaned_news.csv 列名：", df1.columns.tolist())
print("fakenews_dataset.csv 列名：", df2.columns.tolist())