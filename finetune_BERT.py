import pandas as pd
import json
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def process_file(file):
    for line in file:
        jsonl = json.loads(line)
        for instance in jsonl['instances']:
            line1 = jsonl['instruction'] + instance['input']
            line2 = instance['output']
            data.append({'input': line1, 'output': line2, 'label': 1})


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


data = []
with open('data/seed_tasks.jsonl', 'r') as f:
    process_file(f)

# this code is for adding the examples we generated (I named it classifier_training_tasks.jsonl)
with open('data/classifier_training_tasks.jsonl', 'r') as f:
    process_file(f)

df1 = pd.DataFrame(data)

# create irrelevant examples by using same seed task instructions and shuffling outputs
# things to consider:
#   - the shuffling is not gauranteed to produce irrelevant pairings
#   - how does reuse of instructions affect results when there are very few total examples?
#   - is concatenation enough or should the template contain more than that?
df2 = df1.copy()
df2['output'] = df2['output'].sample(frac=1).reset_index(drop=True)
df2['label'] = 0
df = pd.concat([df1, df2], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

train_set, val_test_set = train_test_split(df, test_size=0.3, random_state=0)
val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=0)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_set['input'].tolist(), train_set['output'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_set['input'].tolist(), val_set['output'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_set['input'].tolist(), test_set['output'].tolist(), truncation=True, padding=True)


class RelevancyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = RelevancyDataset(train_encodings, train_set['label'].tolist())
val_dataset = RelevancyDataset(val_encodings, val_set['label'].tolist())
test_dataset = RelevancyDataset(test_encodings, test_set['label'].tolist())

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

os.makedirs('./model', exist_ok=True)
os.makedirs('./events', exist_ok=True)

training_args = TrainingArguments(
    output_dir='./model',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=300,
    weight_decay=0.012,
    logging_dir='./events',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model()

# TODO: hyperparameter tuning (on GPU!!)

