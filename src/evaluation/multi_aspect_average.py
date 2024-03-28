import pandas as pd

generation_model = 'dialoGPT' # generation_model must be dialoGPT, chatGPT, or vicuna

chatgpt = pd.read_csv("OpenAI_Models/" + generation_model + "_scores_gpt-3.5-turbo.csv")
gpt4 = pd.read_csv("OpenAI_Models/" + generation_model + "_scores_gpt-4.csv")
prometheus = pd.read_csv("Prometheus/" + generation_model + "_scores_prometheus.csv")
vicuna = pd.read_csv("Vicuna/" + generation_model + "_scores_vicuna-v1.3.csv")

df = pd.read_csv("all_candidateCNs.csv")
output_df = pd.DataFrame(columns=['HATE_SPEECH', 'COUNTER_NARRATIVE', 'TARGET', generation_model, 'chatgpt_avg', 'chatgpt_ovr', 'vicuna_avg',
                        'vicuna_ovr', 'gpt4_avg', 'gpt4_ovr', 'prometheus_avg', 'prometheus_ovr'])

# chatGPT multi aspect average
multi_aspect_scores, overall_scores = [], []
for i in range(0, len(chatgpt)):
    scores = chatgpt.iloc[i]
    multi_aspect_avg = (scores[generation_model + '_opposition_score'] + scores[generation_model + '_relatedness_score'] + scores[generation_model + '_specificity_score'] + scores[generation_model + '_toxicity_score'] + scores[generation_model + '_fluency_score']) / 5

    multi_aspect_scores.append(multi_aspect_avg)
    overall_scores.append(scores[generation_model + '_overall_score'])

output_df['chatgpt_avg'] = multi_aspect_scores
output_df['chatgpt_ovr'] = overall_scores

# vicuna multi-aspect average
multi_aspect_scores, overall_scores = [], []
for i in range(0, len(vicuna)):
    scores = vicuna.iloc[i]
    multi_aspect_avg = (scores[generation_model + '_opposition_score'] + scores[generation_model + '_relatedness_score'] + scores[generation_model + '_specificity_score'] + scores[generation_model + '_toxicity_score'] + scores[generation_model + '_fluency_score']) / 5

    multi_aspect_scores.append(multi_aspect_avg)
    overall_scores.append(scores[generation_model + '_overall_score'])

output_df['vicuna_avg'] = multi_aspect_scores
output_df['vicuna_ovr'] = overall_scores

# gpt-4 multi aspect average
multi_aspect_scores, overall_scores = [], []
for i in range(0, len(gpt4)):
    scores = gpt4.iloc[i]
    multi_aspect_avg = (scores[generation_model + '_opposition_score'] + scores[generation_model + '_relatedness_score'] + scores[generation_model + '_specificity_score'] + scores[generation_model + '_toxicity_score'] + scores[generation_model + '_fluency_score']) / 5

    multi_aspect_scores.append(multi_aspect_avg)
    overall_scores.append(scores[generation_model + '_overall_score'])

output_df['gpt4_avg'] = multi_aspect_scores
output_df['gpt4_ovr'] = overall_scores

# prometheus multi aspect average
multi_aspect_scores, overall_scores = [], []
for i in range(0, len(prometheus)):
    scores = prometheus.iloc[i]
    multi_aspect_avg = (scores[generation_model + '_opposition_score'] + scores[generation_model + '_relatedness_score'] + scores[generation_model + '_specificity_score'] + scores[generation_model + '_toxicity_score'] + scores[generation_model + '_fluency_score']) / 5

    multi_aspect_scores.append(multi_aspect_avg)
    overall_scores.append(scores[generation_model + '_overall_score'])

output_df['prometheus_avg'] = multi_aspect_scores
output_df['prometheus_ovr'] = overall_scores

output_df[generation_model] = chatgpt[generation_model]

output_df.to_csv(generation_model + "_scores_llm.csv", index=False)





