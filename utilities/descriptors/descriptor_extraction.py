import pandas as pd
df = pd.read_csv('descriptor_columns_full.csv')
desc_columns = list(df.columns)
df2 = pd.DataFrame(desc_columns, columns=['descriptor'])
df2.to_csv('descriptor_columns_full1.csv', index=False)