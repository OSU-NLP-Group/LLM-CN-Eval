import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Parses generated counter narrative from DialoGPT output
def counter_narrative(output):
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    dialogpt_output = decoded.split("  ")

    # If/else statement to check if DialoGPT generated output correctly 
    if len(dialogpt_output) == 2:
        return dialogpt_output[1]
    else:
        dialogpt_output.append("CHECK " + dialogpt_output[0])
        return dialogpt_output[1]

# Loads finetuned DialoGPT model
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
model_state = torch.load('finetuning/dialoGPT-finetuned.pt')
model.load_state_dict(model_state)

tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token

test = pd.read_csv("../../data/multitargetCONAN_test.csv")
sentences = test["HATE_SPEECH"]
targets = test["TARGET"]
labels = test["COUNTER_NARRATIVE"]
df = pd.DataFrame(columns=['HATE_SPEECH', 'COUNTER_NARRATIVE', 'TARGET', 'pk1', 'pk2', 'pk3', 'pk4', 'pk5'])
df["HATE_SPEECH"] = sentences
df['COUNTER_NARRATIVE'] = labels
df["TARGET"] = targets

for i in range(0, len(sentences)):
    inputs = tokenizer([sentences[i]], return_tensors="pt", padding=True)

    # Samples 5 generated counter narratives following the methodology and best decoding mechanism from https://arxiv.org/abs/2204.01440
    pk_outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], do_sample=True, top_k=40, top_p=.92, num_return_sequences=5, max_length=512)
    df['pk1'][i] = counter_narrative(pk_outputs[0])
    df['pk2'][i] = counter_narrative(pk_outputs[1])
    df['pk3'][i] = counter_narrative(pk_outputs[2])
    df['pk4'][i] = counter_narrative(pk_outputs[3])
    df['pk5'][i] = counter_narrative(pk_outputs[4])

df.to_csv('dialoGPT_candidateCNs.csv', index=False)