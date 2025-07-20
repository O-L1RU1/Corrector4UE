import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
import os
from torch.nn import functional as F

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Fine-tune a model with custom dataset.")
parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model.")
parser.add_argument('--data_path', type=str, required=True, help="Path to the train dataset.")
parser.add_argument('--save_path', type=str, required=True, help="Path to save the fine-tuned model.")
args = parser.parse_args()

model_path = args.model_path
data_path = args.data_path
save_path = args.save_path

os.environ["WANDB_DISABLED"] = "true"

batch_size = 8
metric_name = "f1"

dataset = load_dataset(data_path)

labels = [label for label in dataset['train'].features.keys() if label not in ['text','question','labels']]
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained(model_path)

def preprocess_data(examples):
    texts = examples["text"]
    def process_text(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > 512:
            tokens = tokens[-512:]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    processed_texts = [process_text(text) for text in texts]
    encodings = tokenizer.batch_encode_plus(
        processed_texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(processed_texts), len(labels)), dtype=np.float64)
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = np.array(labels_batch[label], dtype=np.float64)
    encodings["labels"] = labels_matrix.tolist()
    return encodings

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")

config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    problem_type="multi_label_classification",
    num_labels=1,
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    save_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

#04_train.py --model_path ${model_path_train} --data_path ${data_path} --save_path ${save_path}