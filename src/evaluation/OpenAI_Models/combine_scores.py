import pandas as pd

generation_model = "dialoGPT" # generation_model must be dialoGPT, chatGPT, or vicuna
evaluation_model = "gpt-4" # evaluation_model must be gpt-4. gpt-3.5-turbo

output_df = pd.DataFrame(columns=[generation_model, generation_model + "_opposition_score", generation_model + "_opposition_exp",
    generation_model + "_relatedness_score", generation_model + "_relatedness_exp", generation_model + "_specificity_score", 
    generation_model + "_specificity_exp", generation_model + "_toxicity_score", generation_model + "_toxicity_exp", 
    generation_model + "_fluency_score", generation_model + "fluency_score", generation_model + "fluency_exp"])

opposition = pd.read_csv(generation_model + "_opposition_" + evaluation_model + '.csv')
relatedness = pd.read_csv(generation_model + "_relatedness_" + evaluation_model + '.csv')
specificity = pd.read_csv(generation_model + "_specificity_" + evaluation_model + '.csv')
toxicity = pd.read_csv(generation_model + "_toxicity_" + evaluation_model + '.csv')
fluency = pd.read_csv(generation_model + "_fluency_" + evaluation_model + '.csv')
overall = pd.read_csv(generation_model + "_overall_" + evaluation_model + '.csv')

output_df[generation_model] = opposition[generation_model]
output_df[generation_model + '_opposition_score'] = opposition[generation_model + '_opposition_score']
output_df[generation_model + '_opposition_exp'] = opposition[generation_model + '_opposition_exp']

output_df[generation_model + '_relatedness_score'] = relatedness[generation_model + '_relatedness_score']
output_df[generation_model + '_relatedness_exp'] = relatedness[generation_model + '_relatedness_exp']

output_df[generation_model + '_specificity_score'] = specificity[generation_model + '_specificity_score']
output_df[generation_model + '_specificity_exp'] = specificity[generation_model + '_specificity_exp']

output_df[generation_model + '_toxicity_score'] = toxicity[generation_model + '_toxicity_score']
output_df[generation_model + '_toxicity_exp'] = toxicity[generation_model + '_toxicity_exp']

output_df[generation_model + '_fluency_score'] = fluency[generation_model + '_fluency_score']
output_df[generation_model + '_fluency_exp'] = fluency[generation_model + '_fluency_exp']

output_df[generation_model + '_overall_score'] = overall[generation_model + '_overall_score']
output_df[generation_model + '_overall_exp'] = overall[generation_model + '_overall_exp']

output_df.to_csv(generation_model + "_scores_" + evaluation_model + '.csv')