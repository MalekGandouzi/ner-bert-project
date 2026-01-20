import os
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from preprocessing.load_data import load_conll
from datasets import Dataset   # ✅ OBLIGATOIRE

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "bert-base-cased"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_FILE = os.path.join(BASE_DIR, "data", "raw", "train_split.conll")
VALID_FILE = os.path.join(BASE_DIR, "data", "raw", "valid.conll")

label2id = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6,
    "B-MISC": 7, "I-MISC": 8
}
id2label = {v: k for k, v in label2id.items()}

# -----------------------------
# LOAD DATA
# -----------------------------
train_sentences, train_labels = load_conll(TRAIN_FILE)
valid_sentences, valid_labels = load_conll(VALID_FILE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(sentences, labels):
    tokenized = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True
    )

    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[label[word_id]])
            else:
                label_ids.append(-100)
            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized

# 🔥 CORRECTION CRUCIALE ICI
train_encodings = tokenize_and_align(train_sentences, train_labels)
valid_encodings = tokenize_and_align(valid_sentences, valid_labels)

train_dataset = Dataset.from_dict(train_encodings)
valid_dataset = Dataset.from_dict(valid_encodings)

# -----------------------------
# MODEL
# -----------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# -----------------------------
# TRAINING
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/ner_improved",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="epoch",   # transformers ≥ 4.56
    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        curr_preds = []
        curr_labels = []
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                curr_preds.append(id2label[p_i])
                curr_labels.append(id2label[l_i])
        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)



trainer.train()

# -----------------------------
# SAVE
# -----------------------------
model.save_pretrained("models/ner_improved")
tokenizer.save_pretrained("models/ner_improved")

print("✅ Entraînement amélioré terminé")

