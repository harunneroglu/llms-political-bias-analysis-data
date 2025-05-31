import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

df = pd.read_excel("Raw_Log_Data.xlsx")

statements = df["Statement"].unique()
contexts = df["Context"].unique()

results = []

# for each statement-context pair
for ctx in contexts:
    df_ctx = df[df["Context"] == ctx]
    for stmt in statements:
        gpt_scores = df_ctx[(df_ctx["Statement"] == stmt) & (df_ctx["Model"] == "ChatGPT")]["Mapped_Score_0_3"]
        deep_scores = df_ctx[(df_ctx["Statement"] == stmt) & (df_ctx["Model"] == "DeepSeek")]["Mapped_Score_0_3"]

        if len(gpt_scores) > 1 and len(deep_scores) > 1:
            t_stat, p_val = ttest_ind(gpt_scores, deep_scores, equal_var=False)
            chat_mean = gpt_scores.mean()
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
                "Context": ctx,
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

df_results.to_excel("Contextual_TTest_(descending).xlsx", index=False)
