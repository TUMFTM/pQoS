import pandas as pd

file_path = "workspace/data/cellular_dataframe.parquet"
df = pd.read_parquet(file_path)
df.to_csv("server.csv", index=False)