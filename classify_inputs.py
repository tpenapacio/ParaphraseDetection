from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
import torch
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

def process_file(file):
    for line in file:
        jsonl = json.loads(line)
        for instance in jsonl['instances']:
            line1 = jsonl['instruction'] + instance['input']
            line2 = instance['output']
            data.append({'input': line1, 'output': line2, 'label': 1})

data = []

# hehe when we generate test data
with open('data/evaluation_tasks.jsonl', 'r') as f:
    process_file(f)

df1 = pd.DataFrame(data)

df2 = df1.copy()
df2['output'] = df2['output'].sample(frac=1).reset_index(drop=True)
df2['label'] = 0
test_set = pd.concat([df1, df2], ignore_index=True)
test_set = test_set.sample(frac=1).reset_index(drop=True)

os.makedirs('./eval', exist_ok=True)


res = []
for i, row in test_set.iterrows():
    pred = is_relevant(row['input'], row['output'])
    res.append({'input' : test_set['input'], 
                'output' : test_set['output'],
                'label' : test_set['label'],
                'prediction' : pred})
df = pd.DataFrame.from_dict(res)
df.to_csv('eval/preds')

# random exploration using generated pairs
# print(is_relevant("Could you describe the relationship between the following analogies?Book : Reading :: Music : Listening","Consider a young boy who has grown up hearing that 'all men should be strong and not show emotions.' This belief can prevent him from expressing his feelings when he's sad, anxious, or scared, causing him to suppress his emotions. Over time, this repression can lead to serious mental health issues like depression and anxiety. Furthermore, it can strain his relationships, as he may struggle to empathize with others or communicate effectively about his own emotions. This stereotype, while seemingly promoting strength, can actually harm individuals and their interpersonal connections."))
# print(is_relevant("Could you describe the relationship between the following analogies?Book : Reading :: Music : Listening", "The relation between the given pairs is that they are opposites."))
