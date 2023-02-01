from data_preparation.perpare import PreProcess
import pandas as pd

pre_process = PreProcess()

df_raw = pd.read_parquet("data/raw/ABEV3.parquet")