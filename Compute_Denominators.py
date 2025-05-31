import pandas as pd

df = pd.read_excel("Raw_Log_Data.xlsx")

# Compute mean mapped score per model per statement
grouped = df.groupby(["Statement", "Context", "Model", "Topic_Tag"])["Mapped_Score_0_3"].mean().reset_index()
pivot = grouped.pivot(index=["Statement", "Context", "Topic_Tag"], columns="Model", values="Mapped_Score_0_3").reset_index()
pivot.columns.name = None

econ_rows = pivot[pivot["Topic_Tag"] == "E"]
soc_rows = pivot[pivot["Topic_Tag"] == "S"]

# Compute denominators
econ_denom = ((econ_rows["ChatGPT"] - econ_rows["DeepSeek"]) ** 2).sum()
soc_denom = ((soc_rows["ChatGPT"] - soc_rows["DeepSeek"]) ** 2).sum()

print(f"   Econ axis: {econ_denom:.2f}")
print(f"   Soc axis: {soc_denom:.2f}")
