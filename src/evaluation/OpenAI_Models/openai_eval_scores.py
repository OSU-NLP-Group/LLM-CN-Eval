import os
import openai
import pandas as pd

from dotenv import load_dotenv

# Due to potential rate limit errors when using the OpenAI API, functions for generating evaluation scores from gpt-3.5-turbp and gpt-4 are separated by aspect
# In order to combine the scores generated from this program into one file per evaluation model, use combine_scores.py

# Defines the output dataframe containing evaluation scores and explanations based on the given aspect
def output_dataframe(aspect):
    if aspect == "Opposition":
        return pd.DataFrame(columns=[generation_model, generation_model + "_opposition_score", generation_model + "_opposition_exp"])
    elif aspect == "Relatedness":
        return pd.DataFrame(columns=[generation_model, generation_model + "_relatedness_score", generation_model + "_relatedness_exp"])
    elif aspect == "Specificity":
        return pd.DataFrame(columns=[generation_model, generation_model + "_specificity_score", generation_model + "_specificity_exp"])
    elif aspect == "Toxicity":
        return pd.DataFrame(columns=[generation_model, generation_model + "_toxicity_score", generation_model + "_toxicity_exp"])
    elif aspect == "Fluency":
        return pd.DataFrame(columns=[generation_model, generation_model + "_fluency_score", generation_model + "_fluency_exp"])
    elif aspect == "Overall":
        return pd.DataFrame(columns=[generation_model, generation_model + "_overall_score", generation_model + "_overall_exp"])

# Defines the evaluation prompt to use based on the given aspect  
def prompt(aspect):
    if aspect == "Opposition":
        return prompts['Opposition'][0]
    elif aspect == "Relatedness":
        return prompts['Relatedness'][0]
    elif aspect == "Specificity":
        return prompts['Specificity'][0]
    elif aspect == "Toxicity":
        return prompts["Toxicity"][0]
    elif aspect == "Fluency":
        return prompts["Fluency"][0]
    elif aspect == "Overall":
        return prompts["Overall"][0]
    
# Updates the output dataframe based on the evaluation aspect used 
def finish(aspect):
    if aspect == "Opposition":
        output_df[generation_model + '_opposition_score'] = score
        output_df[generation_model + '_opposition_exp'] = exp
        output_df[generation_model] = df[generation_model]
        output_df.to_csv(generation_model + '_opposition_' + evaluation_model + '.csv', index=False)
    elif aspect == "Relatedness":
       output_df[generation_model + '_relatedness_score'] = score
        output_df[generation_model + '_relatedness_exp'] = exp
        output_df[generation_model] = df[generation_model]
        output_df.to_csv(generation_model + '_relatedness_' + evaluation_model + '.csv', index=False)
    elif aspect == "Specificity":
        output_df[generation_model + '_specificity_score'] = score
        output_df[generation_model + '_specificity_exp'] = exp
        output_df[generation_model] = df[generation_model]
        output_df.to_csv(generation_model + '_specificity_' + evaluation_model + '.csv', index=False)
    elif aspect == "Toxicity":
        output_df[generation_model + '_toxicity_score'] = score
        output_df[generation_model + '_toxicity_exp'] = exp
        output_df[generation_model] = df[generation_model]
        output_df.to_csv(generation_model + '_toxicity_' + evaluation_model + '.csv', index=False)
    elif aspect == "Fluency":
        output_df[generation_model + '_fluency_score'] = score
        output_df[generation_model + '_fluency_exp'] = exp
        output_df[generation_model] = df[generation_model]
        output_df.to_csv(generation_model + '_fluency_' + evaluation_model + '.csv', index=False)
    elif aspect == "Overall":
        output_df[generation_model + '_overall_score'] = score
        output_df[generation_model + '_overall_exp'] = exp
        output_df[generation_model] = df[generation_model]
        output_df.to_csv(generation_model + '_overall_' + evaluation_model + '.csv', index=False)

load_dotenv()

openai.organization = # add your organization ID here 
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv('../all_candidateCNs.csv')
prompts = pd.read_csv('../prompts.csv')
aspect = "Opposition" # aspect must be Opposition, Relatedness, Specificity, Toxicity, Fluency, or Overall
generation_model = "dialoGPT" # generation_model must be dialoGPT, chatGPT, or vicuna
evaluation_model = "gpt-4" # evaluation_model must be gpt-4 or gpt-3.5-turbo

# Gets output dataframe based on evaluation aspect being used 
output_df = output_dataframe(aspect)

# Gets hate speech example and counter narrative response from the chosen generation model
hs = df['HATE_SPEECH'].tolist()
cn = df[generation_model].tolist()

# Gets evaluation prompt based on evaluation aspect being used 
instruction = prompt(aspect)

# Defines lists to contain generated evaluation scores and explanations that will be used to update output dataframe
score, exp = [], []
total_tokens = 0

for i in range(0, len(df)):
    content = "Hate Speech: " + hs[i] + "\nCounter Narrative: " + cn[i] + "\nScore: "
    print(content  + "\n")
    completion = openai.ChatCompletion.create(
        model=evaluation_model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": content}
        ],
        temperature=0,
        max_tokens=512
    )
    output = completion.choices[0].message.content
    print(output + "\n")

    # Finds first number from model response and stores it as the evaluation score 
    score = [int(i) for i in output.split() if i.isdigit()]
    score.append(score[0])

    # Stores the generated score and explanation from the model 
    exp.append(output)

    # Keeps track of total tokens generated from the model for logging purposes
    total_tokens += completion.usage.total_tokens

print(total_tokens)
finish(aspect)