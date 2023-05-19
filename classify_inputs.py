from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
import torch
from finetune_BERT import process_file
from finetune_BERT import RelevancyDataset
import os
import pandas as pd
import json

model = BertForSequenceClassification.from_pretrained('model')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
nlp = pipeline('text-classification', model=model, tokenizer=tokenizer)


def is_relevant(input1, input2):
    inputs = tokenizer.encode_plus(input1, input2, return_tensors='pt')
    outputs = model(**inputs)
    p = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(p)
    print(f"Prediction: {prediction.item()}")
    print(f"Probability of Relevance: {p[0][1].item()}")
    return prediction.item()

test_data = []

# hehe when we generate test data
with open('', 'r') as f:
    process_file(f)

df1 = pd.DataFrame(data)

df2 = df1.copy()
df2['output'] = df2['output'].sample(frac=1).reset_index(drop=True)
df2['label'] = 0
test_set = pd.concat([df1, df2], ignore_index=True)
test_set = test_set.sample(frac=1).reset_index(drop=True)

os.makedirs('./eval', exist_ok=True)

df = pd.DataFrame(columms = ['input', 'output', 'label', 'prediction'])

for i, row in df.iterrows():
    pred = is_relevant(test_set['input'], test_set['output'])
    df.append({'input' : test_set['input'], 
                'output' : test_set['output'],
                'label' : test_set['label'],
                'prediction' : pred})

df.to_csv('eval/preds')

# random exploration using generated pairs
# print(is_relevant("Could you describe the relationship between the following analogies?Book : Reading :: Music : Listening","Consider a young boy who has grown up hearing that 'all men should be strong and not show emotions.' This belief can prevent him from expressing his feelings when he's sad, anxious, or scared, causing him to suppress his emotions. Over time, this repression can lead to serious mental health issues like depression and anxiety. Furthermore, it can strain his relationships, as he may struggle to empathize with others or communicate effectively about his own emotions. This stereotype, while seemingly promoting strength, can actually harm individuals and their interpersonal connections."))
# print(is_relevant("Could you describe the relationship between the following analogies?Book : Reading :: Music : Listening", "The relation between the given pairs is that they are opposites."))