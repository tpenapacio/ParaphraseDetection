from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
import torch

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


# random exploration using generated pairs
print(is_relevant("Could you describe the relationship between the following analogies?Book : Reading :: Music : Listening", "Consider a young boy who has grown up hearing that 'all men should be strong and not show emotions.' This belief can prevent him from expressing his feelings when he's sad, anxious, or scared, causing him to suppress his emotions. Over time, this repression can lead to serious mental health issues like depression and anxiety. Furthermore, it can strain his relationships, as he may struggle to empathize with others or communicate effectively about his own emotions. This stereotype, while seemingly promoting strength, can actually harm individuals and their interpersonal connections."))
print(is_relevant("Could you describe the relationship between the following analogies?Book : Reading :: Music : Listening", "The relation between the given pairs is that they are opposites."))

