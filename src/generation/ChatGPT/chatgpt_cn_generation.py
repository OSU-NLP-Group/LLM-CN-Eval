import os
import openai
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

openai.organization = # add your organization ID here
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv('../../data/multitargetCONAN_test.csv')
examples = df['HATE_SPEECH'].tolist()
cns = df['COUNTER_NARRATIVE']
targets = df["TARGET"]

output_df = pd.DataFrame(columns=["HATE_SPEECH", "COUNTER_NARRATIVE", "TARGET", "chatGPT"])
outputs = []
total_tokens = 0

system_message = "Generate counterspeech for the given hate speech example."

for i in range(0, len(examples)):
  content = "Hate Speech: " + examples[i] + "\nCounter Speech: "
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": system_message},
      {"role": "user", "content": content}
    ],
    temperature=1
  )
  outputs.append(completion.choices[0].message.content)
  total_tokens += completion.usage.total_tokens

print(total_tokens)
output_df["HATE_SPEECH"] = examples
output_df["COUNTER_NARRATIVE"] = cns
output_df["TARGET"] = targets
output_df["chatGPT"] = outputs
output_df.to_csv('chatGPT_candidateCNs.csv', index=False)
