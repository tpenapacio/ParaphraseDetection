# takes about a minute to run each batch, ~5 hours total?
# used colab

from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
import torch
import os
import pandas as pd
import json

model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model1.load_state_dict(torch.load('/content/drive/MyDrive/Research/493/Self-Instruct/best_model.pt'))

model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model2.load_state_dict(torch.load('/content/drive/MyDrive/Research/493/Self-Instruct/best_model2.pt'))
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def is_relevant(input1, input2, m):
    inputs = tokenizer.batch_encode_plus(list(zip(input1, input2)), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = m(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    return prediction

def process_file(file):
    batch = 250
    inputs = []
    output = []
    input2 = []
    output2 = []
    l = []

    for line in file:
        jsonl = json.loads(line)
        inputs.append(jsonl["instruction"] + " " + jsonl["input"])
        input2.append(jsonl["instruction"])
        output.append(jsonl["output"])
        output2.append(jsonl["input"])
        l.append(jsonl)
        if len(inputs) >= 250:
          handle_batch(inputs, output, input2, output2, l)
          inputs = []
          output = []
          input2 = []
          output2 = []
          l = []
    if len(inputs) > 0:
      handle_batch(inputs, output, input2, output2, l)

def handle_batch(inputs, output, input2, output2, l):
    print(1)
    m1 = is_relevant(inputs, output, model1)
    m2 = is_relevant(input2, output2, model2)

    for ln, m1_val, m2_val in zip(l, m1, m2):
      data.append(ln)
      if m1_val.item() == 1:
          data_m1_rel.append(ln)
      if m2_val.item() == 1:
          data_m2_rel.append(ln)
      if m1_val.item() == 1 and m2_val.item() == 1:
          data_m1_m2_rel.append(ln)

data = []
data_m1_rel = []
data_m2_rel = []
data_m1_m2_rel = []
data_m1_not_rel = []
data_m2_not_rel = []
data_m1_m2_not_rel = []


with open('/content/drive/My Drive/' + FOLDERNAME + 'all_instances_82K.jsonl', 'r') as f:
    process_file(f)

data_m1_not_rel = list(set(data).difference(set(data_m1_rel)))
data_m2_not_rel = list(set(data).difference(set(data_m2_rel)))
data_m1_m2_not_rel = list(set(data).difference(set(data_m1_m2_rel)))

print("Length original: " + str(len(data)))
print("Length M1: " + str(len(data_m1_rel)))
print("Length M2: " + str(len(data_m2_rel)))
print("Length M1 + M2: " + str(len(data_m1_m2_rel)))
