import pandas as pd
from scipy.stats import zscore

df = pd.read_excel("Model_Centered_Coord.xlsx")

df["Z_Econ_Coord_ChatGPT"] = zscore(df["Econ_Coord_ChatGPT"])
df["Z_Econ_Coord_DeepSeek"] = zscore(df["Econ_Coord_DeepSeek"])
df["Z_Soc_Coord_ChatGPT"] = zscore(df["Soc_Coord_ChatGPT"])
df["Z_Soc_Coord_DeepSeek"] = zscore(df["Soc_Coord_DeepSeek"])

df.to_excel("Model_Centered_Coord_ZScore.xlsx", index=False)
