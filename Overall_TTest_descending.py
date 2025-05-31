import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

df = pd.read_excel("Raw_Log_Data.xlsx")

chatgpt = df[df["Model"] == "ChatGPT"]
deepseek = df[df["Model"] == "DeepSeek"]

statements = df["Statement"].unique()

results = []

# for each statement
for stmt in statements:
    chat_scores = chatgpt[chatgpt["Statement"] == stmt]["Mapped_Score_0_3"]
    deep_scores = deepseek[deepseek["Statement"] == stmt]["Mapped_Score_0_3"]

    t_stat, p_val = ttest_ind(chat_scores, deep_scores, equal_var=False)

    chat_mean = chat_scores.mean()
    deep_mean = deep_scores.mean()
    diff = chat_mean - deep_mean
    log_p = -np.log10(p_val) if p_val > 0 else np.nan

    if p_val < 0.001:
        level = "Highly Significant"
    elif p_val < 0.05:
        level = "Significant"
    else:
        level = "Not Significant"

    results.append({
        "Statement": stmt,
        "ChatGPT_Mean": chat_mean,
        "DeepSeek_Mean": deep_mean,
        "Model_Diff": diff,
        "T-statistic": t_stat,
        "p-value": p_val,
        "-log10(p)": log_p,
        "Significance_Level": level
    })

df_results = pd.DataFrame(results)
df_results.sort_values(by="-log10(p)", ascending=False, inplace=True)

df_results.to_excel("Overall_TTest_(descending).xlsx", index=False)
