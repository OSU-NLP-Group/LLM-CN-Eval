import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import pandas as pd
from conversation import get_conv_template

#To use Llama 2 tokenizer, call huggingface-cli to login to account using token from https://huggingface.co/settings/tokens to get authentication to gated repo
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("kaist-ai/Prometheus-13b-v1.0", device_map="auto")

generation_model = "dialoGPT" # generation_model must be dialoGPT, chatGPT, or vicuna
output_df = pd.DataFrame(columns=[generation_model, generation_model + '_opposition_score', generation_model + '_opposition_exp', 
            generation_model + '_relatedness_score', generation_model + '_relatedness_exp', generation_model + '_specificity_score', 
            generation_model + '_specificity_exp', generation_model + '_toxicity_score', generation_model + '_toxicity_exp', generation_model + '_fluency_score',
            generation_model + '_fluency_exp', generation_model + '_overall_score', generation_model + '_overall_exp'])
aspect_list = ["Opposition", "Relatedness", "Specificity", "Toxicity", "Fluency", "Overall"]
opp_score, opp_exp, rel_score, rel_exp, spe_score, spe_exp, tox_score, tox_exp, flu_score, flu_exp, ovr_score, ovr_exp = [], [], [], [], [], [], [], [], [], [], [], []
            
df = pd.read_csv('../all_candidateCNs.csv')
hs = df['HATE_SPEECH'].tolist()
cn = df[generation_model].tolist()

# JSON file containing the score rubrics and reference answers for each CN evaluation aspect
prompt_text = pd.read_json("cn_prometheus.json")

# Task description and feedback portions of the Prometheus evaluation prompt
task_description = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

"""
feedback = """\n\n###Feedback: """
  
for i in range(0, len(df)):
    # Fills template for instruction and response in Prometheus evaluation prompt for each HS/CN pair
    instruction = """###The instruction to evaluate:\nGenerate a counter narrative response to the following hate speech example.\nHate Speech: """ + hs[i]

    response = """\n\n###Response to evaluate:\nCounter Narrative: """ + cn[i]

    for aspect in aspect_list:
        # Fills template for reference answer and score rubric in Prometheus evaluation prompt for each evaluation aspect
        reference_answer = """\n\n###Reference Answer (Score 5): """ + prompt_text["references"][aspect]

        score_rubric = """\n\n###Score Rubrics: """ + prompt_text["rubrics"][aspect]

        input_text = task_description + instruction + response + reference_answer + score_rubric + feedback

        # Defines use of llama-2 conv template for Prometheus instructions
        conv = get_conv_template("llama-2")

        # Sets system message and input prompt used for Prometheus evaluation within the llama-2 conv template
        conv.set_system_message("You are a fair evaluator language model.")
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        outputs = model.generate(input_ids, do_sample=True, temperature=0.1, top_p=0.9, max_new_tokens=256, repetition_penalty=1.03)

        # Pulls score and explanation from llama-2 conv template 
        decoded = tokenizer.decode(outputs[0])
        print(decoded)
        x = decoded.split('[/INST]  ')
        y = x[1].split(' [RESULT] ')
        explanation = y[0]
        if len(y) < 1:
            score = "N/A"
        else:
            z = y[1].split('</s>')
            score = z[0]
        print('Feedback: ' + explanation)
        print('Score: ' + score)

        # Adds each score and explanation to list for each aspect that will be used to update output dataframe 
        if aspect == "Opposition":
            opp_score.append(score)
            opp_exp.append(explanation)
        elif aspect == "Relatedness":
            rel_score.append(score)
            rel_exp.append(explanation)
        elif aspect == "Specificity":
            spe_score.append(score)
            spe_exp.append(explanation)
        elif aspect == "Toxicity":
            tox_score.append(score)
            tox_exp.append(explanation)
        elif aspect == "Fluency":
            flu_score.append(score)
            flu_exp.append(explanation)
        elif aspect == "Overall":
            ovr_score.append(score)
            ovr_exp.append(explanation)

# Updating output dataframe with lists containing each generated evaluation score and explanation
output_df[generation_model] = df[generation_model]
output_df[generation_model + '_opposition_score'] = opp_score
output_df[generation_model + '_opposition_exp'] = opp_exp
    
output_df[generation_model + '_relatedness_score'] = rel_score
output_df[generation_model + '_relatedness_exp'] = rel_exp
        
output_df[generation_model + '_specificity_score'] = spe_score
output_df[generation_model + '_specificity_exp'] = spe_exp

output_df[generation_model + '_toxicity_score'] = tox_score
output_df[generation_model + '_toxicity_exp'] = tox_exp

output_df[generation_model + '_fluency_score'] = flu_score
output_df[generation_model + '_fluency_exp'] = flu_exp

output_df[generation_model + '_overall_score'] = ovr_score
output_df[generation_model + '_overall_exp'] = ovr_exp

output_df.to_csv(generation_model + "_scores_prometheus.csv", index=False)