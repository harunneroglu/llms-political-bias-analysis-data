import pandas as pd
import statsmodels.api as sm

df = pd.read_excel("Raw_Log_Data.xlsx")
df["Model_Code"] = df["Model"].map({"ChatGPT": 0, "DeepSeek": 1})

results = []

for stmt in df["Statement"].unique():
    df_stmt = df[df["Statement"] == stmt]
    if df_stmt["Model_Code"].nunique() < 2:
        continue

    tag = df_stmt["Topic_Tag"].iloc[0] if "Topic_Tag" in df_stmt.columns else None

    df_stmt = df_stmt.copy()
    if tag == "E":
        df_stmt["Mapped_Score_0_3"] = df_stmt["Mapped_Score_0_3"] / 14.71
    elif tag == "S":
        df_stmt["Mapped_Score_0_3"] = df_stmt["Mapped_Score_0_3"] / 31.42

    y = df_stmt["Mapped_Score_0_3"]
    X = sm.add_constant(df_stmt["Model_Code"])

    try:
        model = sm.OLS(y, X).fit()
        intercept = model.params["const"]
        coef = model.params["Model_Code"]
        r_squared = model.rsquared
        effect_type = "High" if r_squared >= 0.10 else "Low"
    except Exception as e:
        intercept, coef, r_squared, effect_type = [None] * 4

    results.append({
        "Statement": stmt,
        "Topic_Tag": tag,
        "Intercept": intercept,
        "a_Model": coef,
        "R_squared": r_squared,
        "Effect_Type": effect_type
    })

df_results = pd.DataFrame(results)
df_results = df_results.dropna(subset=["R_squared"])
df_results = df_results.sort_values(by="R_squared", ascending=False)
df_results.to_excel("Regression_Results_(descending).xlsx", index=False)
