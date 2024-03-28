from transformers import LlamaForCausalLM, LlamaTokenizer
from fastchat.model.model_adapter import load_model
import pandas as pd

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-33b-v1.3", legacy=False)
model, _ = load_model(model_path="lmsys/vicuna-33b-v1.3", device="cuda", num_gpus=4)

def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)

test = pd.read_csv('../../data/multitargetCONAN_test.csv')
sentences = test['HATE_SPEECH']
task_prefix = "Generate counterspeech to the given hate speech post.\nHate Speech: "

output_df = pd.DataFrame(columns=["HATE_SPEECH", "COUNTER_NARRATIVE", "TARGET", "vicuna"])
output_df['HATE_SPEECH'] = test['HATE_SPEECH']
output_df['COUNTER_NARRATIVE'] = test['COUNTER_NARRATIVE']
output_df['TARGET'] = test['TARGET']

for i in range(0, len(test)):
    instruction = task_prefix + sentences[i] + "\nCounterspeech: "
    inputs = tokenizer([instruction], return_tensors="pt", padding=True)
    input_ids = load2gpu(inputs['input_ids'], 'cuda')
    mask = load2gpu(inputs['attention_mask'], 'cuda')
    output = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=512, temperature=1)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.replace(instruction, "")
    print(decoded + "\n")
    output_df['vicuna'][i] = response

output_df.to_csv('vicuna-v1.3_candidateCNs.csv', index=False)