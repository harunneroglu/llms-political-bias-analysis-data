import pandas as pd

df = pd.read_excel("Model_Centered_Coord.xlsx")

econ_long = df.melt(id_vars=["Statement", "Context", "Topic_Tag"],
                    value_vars=["Econ_Coord_ChatGPT", "Econ_Coord_DeepSeek"],
                    var_name="Model", value_name="Econ_Coord")
econ_long["Model"] = econ_long["Model"].str.replace("Econ_Coord_", "")

soc_long = df.melt(id_vars=["Statement", "Context", "Topic_Tag"],
                   value_vars=["Soc_Coord_ChatGPT", "Soc_Coord_DeepSeek"],
                   var_name="Model", value_name="Soc_Coord")
soc_long["Model"] = soc_long["Model"].str.replace("Soc_Coord_", "")

merged = pd.merge(
    econ_long[["Context", "Model", "Econ_Coord"]],
    soc_long[["Context", "Model", "Soc_Coord"]],
    on=["Context", "Model"]
)

summary_df = merged.groupby(["Context", "Model"], as_index=False).mean()

# overall = can be made offset of the method (decide later!)
overall = merged.groupby("Model", as_index=False)[["Econ_Coord", "Soc_Coord"]].mean()
overall["Context"] = "overall"

overall = overall[["Context", "Model", "Econ_Coord", "Soc_Coord"]]
summary_df = pd.concat([summary_df, overall], ignore_index=True)

summary_df.to_excel("Sum_Model_Centered_Coord.xlsx", index=False)
