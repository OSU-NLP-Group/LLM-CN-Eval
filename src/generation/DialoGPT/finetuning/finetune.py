import torch 
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from torch.optim import Adam

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel

class Dataset(torch.utils.data.Dataset):

  def __init__(self, annotation_file):
    self.df = pd.read_csv(annotation_file)
    self.texts = [text for text in self.df['PAIR']]
    self.labels = [text for text in self.df['PAIR']]

  def classes(self):
        return self.labels

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):
        hate = self.texts[index]
        counter = self.labels[index]

        return hate, counter

class Collator():
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_source_length = 128
        self.max_target_length = 128

    def __call__(self, lst):
        texts = [x[0] for x in lst]
        labels = [x[1] for x in lst]
        train_input = self.tokenizer(
            [text for text in texts],
            padding='max_length', 
            max_length = self.max_source_length, 
            truncation=True,
            return_tensors="pt"
        )
        train_labels = train_input
        return train_input, train_labels
    
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
    
def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)
    collator = Collator()

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True, collate_fn=collator)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4, shuffle=True, collate_fn=collator)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(model.parameters(), lr= learning_rate)

    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")

    if use_cuda:
        model = model.cuda()

    model.train()
    for epoch_num in range(epochs):

        total_loss_train = 0

        for train_inputs, train_labels in tqdm(train_dataloader):

            input_ids = load2gpu(train_inputs.input_ids, device)
            mask = load2gpu(train_inputs.attention_mask, device)
            labels = load2gpu(train_labels.input_ids, device)
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            total_loss_train += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss_val = 0

        with torch.no_grad():

            for val_inputs, val_labels in tqdm(val_dataloader):
                input_ids = load2gpu(val_inputs.input_ids, device)
                mask = load2gpu(val_inputs.attention_mask, device)
                labels = load2gpu(val_labels.input_ids, device)
                labels[labels == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                total_loss_val += loss.item()

        print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f}')

model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
learning_rate = 5e-5
epochs = 2
train(model, 'dialogpt_train.csv', 'dialogpt_val.csv', learning_rate, epochs)

torch.save(model.state_dict(), "dialoGPT-finetuned.pt")