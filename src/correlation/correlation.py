from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np 
import pandas as pd

# computes the pearson, spearman, and kendall tau correlation between each eval metric and the amt-annotated scores and updates output dataframe
def correlation(merged_df, metric_name, output_df, model_name):
    x = merged_df[metric_name].to_numpy()
    y = merged_df['amt_avg'].to_numpy()

    pearson, p = pearsonr(x, y)
    output_df.loc[output_df.metric == metric_name, model_name + '_amtavg_pearson'] = pearson
    output_df.loc[output_df.metric == metric_name, model_name + '_amtavg_pearson_pvalue'] = p

    spearman, p = spearmanr(x, y)
    output_df.loc[output_df.metric == metric_name, model_name + '_amtavg_spearman'] = spearman
    output_df.loc[output_df.metric == metric_name, model_name + '_amtavg_spearman_pvalue'] = p

    kendall, p = kendalltau(x, y, variant='b')
    output_df.loc[output_df.metric == metric_name, model_name + '_amtavg_kendall'] = kendall
    output_df.loc[output_df.metric == metric_name, model_name + '_amtavg_kendall_pvalue'] = p

    y = merged_df['amt_ovr'].to_numpy()

    pearson, p = pearsonr(x, y)
    output_df.loc[output_df.metric == metric_name, model_name + '_amtovr_pearson'] = pearson
    output_df.loc[output_df.metric == metric_name, model_name + '_amtovr_pearson_pvalue'] = p

    spearman, p = spearmanr(x, y)
    output_df.loc[output_df.metric == metric_name, model_name + '_amtovr_spearman'] = spearman
    output_df.loc[output_df.metric == metric_name, model_name + '_amtovr_spearman_pvalue'] = p

    kendall, p = kendalltau(x, y, variant='b')
    output_df.loc[output_df.metric == metric_name, model_name + '_amtovr_kendall'] = kendall
    output_df.loc[output_df.metric == metric_name, model_name + '_amtovr_kendall_pvalue'] = p

    return output_df

# combines and formats the automatic eval and llm eval dataframes for each generation model
def df_formatting(auto_df, llm_df, model_name):
    auto_df.drop(['TARGET', 'COUNTER_NARRATIVE'], axis=1)
    llm_df.drop(["HATE_SPEECH", "TARGET", "COUNTER_NARRATIVE", model_name], axis=1)
    final_df = pd.concat([auto_df, llm_df], axis=1)
    final_df = final_df.rename(columns={model_name: "counter_narrative", "HATE_SPEECH": "hate_speech"})

    return final_df

dialoGPT_auto = pd.read_csv('../evaluation/automatic_metrics/dialoGPT_scores_auto.csv')
dialoGPT_llm = pd.read_csv('../evaluation/dialoGPT_scores_llm.csv')
dialoGPT = df_formatting(dialoGPT_auto, dialoGPT_llm, 'dialoGPT')

chatGPT_auto = pd.read_csv('../evaluation/automatic_metrics/chatGPT_scores_auto.csv')
chatGPT_llm = pd.read_csv('../evaluation/chatGPT_scores_llm.csv')
chatGPT = df_formatting(chatGPT_auto, chatGPT_llm, 'chatGPT')

vicuna_auto = pd.read_csv('../evaluation/automatic_metrics/vicuna_scores_auto.csv')
vicuna_llm = pd.read_csv('../evaluation/vicuna_scores_llm.csv')
vicuna = df_formatting(vicuna_auto, vicuna_llm, 'vicuna')

all_models = pd.concat([dialoGPT, chatGPT, vicuna])

amt_scores = pd.read_csv('amt_scores_example.csv')
amt_scores = amt_scores.drop(['hate_speech'], axis=1)

# Merges automatic evaluation scores and amt-annotated scores into one dataframe
dialoGPT_merged = pd.merge(dialoGPT, amt_scores, on=['counter_narrative'], how='inner')
chatGPT_merged = pd.merge(chatGPT, amt_scores, on=['counter_narrative'], how='inner')
vicunaGPT_merged = pd.merge(vicuna, amt_scores, on=['counter_narrative'], how='inner')
all_models_merged = pd.merge(all_models, amt_scores, on=['counter_narrative'], how='inner')

metrics = ['bleu1', 'bleu3', 'bleu4', 'rouge', 'meteor', 'bertscore', 'bartscore_precision', 'bartscore_cnn_precision',
          'bartscore_cnn_para_precision', 'bartscore_recall', 'bartscore_cnn_recall', 'bartscore_cnn_para_recall',
        'bartscore_f1', 'bartscore_cnn_f1', 'bartscore_cnn_para_f1', 'chatgpt_avg', 'chatgpt_ovr', 'vicuna_avg', 'vicuna_ovr', 
        'gpt4_avg', 'gpt4_ovr', 'prometheus_avg', 'prometheus_ovr']

output_df = pd.DataFrame(columns=['metric', 'allmodels_amtavg_pearson', 'allmodels_amtavg_pearson_pvalue', 'allmodels_amtavg_spearman', 'allmodels_amtavg_spearman_pvalue',
                         'allmodels_amtavg_kendall', 'allmodels_amtavg_kendall_pvalue', 'allmodels_amtovr_pearson', 'allmodels_amtovr_pearson_pvalue', 'allmodels_amtovr_spearman',
                         'allmodels_amtovr_spearman_pvalue', 'allmodels_amtovr_kendall', 'allmodels_amtovr_kendall_pvalue', 'dialogpt_amtavg_pearson',
                         'dialogpt_amtavg_pearson_pvalue', 'dialogpt_amtavg_spearman', 'dialogpt_amtavg_spearman_pvalue', 'dialogpt_amtavg_kendall',
                         'dialogpt_amtavg_kendall_pvalue', 'dialogpt_amtovr_pearson', 'dialogpt_amtovr_pearson_pvalue', 'dialogpt_amtovr_spearman',
                         'dialogpt_amtovr_spearman_pvalue', 'dialogpt_amtovr_kendall', 'dialogpt_amtovr_kendall_pvalue', 'chatgpt_amtavg_pearson',
                         'chatgpt_amtavg_pearson_pvalue', 'chatgpt_amtavg_spearman', 'chatgpt_amtavg_spearman_pvalue', 'chatgpt_amtavg_kendall',
                         'chatgpt_amtavg_kendall_pvalue', 'chatgpt_amtovr_pearson', 'chatgpt_amtovr_pearson_pvalue', 'chatgpt_amtovr_spearman',
                         'chatgpt_amtovr_spearman_pvalue', 'chatgpt_amtovr_kendall', 'chatgpt_amtovr_kendall_pvalue', 'vicuna_amtavg_pearson',
                         'vicuna_amtavg_pearson_pvalue', 'vicuna_amtavg_spearman', 'vicuna_amtavg_spearman_pvalue', 'vicuna_amtavg_kendall',
                         'vicuna_amtavg_kendall_pvalue', 'vicuna_amtovr_pearson', 'vicuna_amtovr_pearson_pvalue', 'vicuna_amtovr_spearman',
                         'vicuna_amtovr_spearman_pvalue', 'vicuna_amtovr_kendall', 'vicuna_amtovr_kendall_pvalue'])

output_df['metric'] = metrics
for metric in metrics:
    output_df = correlation(dialoGPT_merged, metric, output_df, 'dialogpt')
    output_df = correlation(chatGPT_merged, metric, output_df, 'chatgpt')
    output_df = correlation(vicuna_merged, metric, output_df, 'vicuna')
    output_df = correlation(all_models_merged, metric, output_df, 'allmodels')

output_df.to_csv('amt_correlations.csv', index=False)