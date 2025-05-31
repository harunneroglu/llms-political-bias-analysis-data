import pandas as pd
import statsmodels.api as sm

df = pd.read_excel("Raw_Log_Data.xlsx")

df["Context_Area_Label"] = df["Context"] + "_" + df["Topic_Tag"].map({"E": "economic", "S": "social"})

mean_scores = df.groupby(["Statement", "Context", "Model"])["Mapped_Score_0_3"].mean().unstack()
meta_info = df.groupby(["Statement", "Context"]).agg({
    "Topic_Tag": "first",
    "Context_Area_Label": "first"
}).reset_index()

meta_info["ChatGPT_Mean"] = mean_scores["ChatGPT"].values
meta_info["DeepSeek_Mean"] = mean_scores["DeepSeek"].values
meta_info["Mean_Diff"] = meta_info["ChatGPT_Mean"] - meta_info["DeepSeek_Mean"]

# new order 1
final_df = meta_info[[
    "Statement", "Context", "ChatGPT_Mean", "DeepSeek_Mean",
    "Mean_Diff", "Topic_Tag", "Context_Area_Label"
]]

# new order 2
final_df = meta_info[[
    "Statement", "Context", "ChatGPT_Mean", "DeepSeek_Mean",
    "Mean_Diff", "Topic_Tag", "Context_Area_Label"
]]

# for the best order
original_order = df[["Statement", "Context"]].drop_duplicates().reset_index(drop=True)
final_df = pd.merge(original_order, final_df, on=["Statement", "Context"], how="left")

final_df.to_excel("Statement_Summary.xlsx", index=False)
