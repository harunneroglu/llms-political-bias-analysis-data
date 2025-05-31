import pandas as pd
import numpy as np

df_raw = pd.read_excel("Raw_Log_Data.xlsx")

order_df = df_raw[['Statement', 'Context']].drop_duplicates().reset_index(drop=True)

grouped = df_raw.groupby(['Statement', 'Context', 'Model'])['Mapped_Score_0_3'].agg(['mean', 'std', 'count']).reset_index()

df_chatgpt = grouped[grouped['Model'] == 'ChatGPT'].copy()
df_deepseek = grouped[grouped['Model'] == 'DeepSeek'].copy()

df_chatgpt.rename(columns={'mean': 'ChatGPT_Mean', 'std': 'ChatGPT_Std', 'count': 'n_ChatGPT'}, inplace=True)
df_deepseek.rename(columns={'mean': 'DeepSeek_Mean', 'std': 'DeepSeek_Std', 'count': 'n_DeepSeek'}, inplace=True)

df = pd.merge(
    df_chatgpt[['Statement', 'Context', 'ChatGPT_Mean', 'ChatGPT_Std', 'n_ChatGPT']],
    df_deepseek[['Statement', 'Context', 'DeepSeek_Mean', 'DeepSeek_Std', 'n_DeepSeek']],
    on=['Statement', 'Context']
)

z = 1.96
df['ChatGPT_CI_Low'] = df['ChatGPT_Mean'] - z * (df['ChatGPT_Std'] / np.sqrt(df['n_ChatGPT']))
df['ChatGPT_CI_High'] = df['ChatGPT_Mean'] + z * (df['ChatGPT_Std'] / np.sqrt(df['n_ChatGPT']))
df['DeepSeek_CI_Low'] = df['DeepSeek_Mean'] - z * (df['DeepSeek_Std'] / np.sqrt(df['n_DeepSeek']))
df['DeepSeek_CI_High'] = df['DeepSeek_Mean'] + z * (df['DeepSeek_Std'] / np.sqrt(df['n_DeepSeek']))

final_df = df[['Statement', 'Context',
               'ChatGPT_Mean', 'ChatGPT_CI_Low', 'ChatGPT_CI_High',
               'DeepSeek_Mean', 'DeepSeek_CI_Low', 'DeepSeek_CI_High']]

final_df_sorted = pd.merge(order_df, final_df, on=['Statement', 'Context'], how='left')

final_df_sorted.to_excel("CI.xlsx", index=False)
