import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

INPUT_FILE = os.path.join(RAW_DIR, "train.conll")
TRAIN_OUT = os.path.join(RAW_DIR, "train_split.conll")
VALID_OUT = os.path.join(RAW_DIR, "valid.conll")

def read_sentences(path):
    sentences = []
    sentence = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                sentence.append(line)
        if sentence:
            sentences.append(sentence)

    return sentences

def write_sentences(path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for line in sent:
                f.write(line)
            f.write("\n")

sentences = read_sentences(INPUT_FILE)
random.shuffle(sentences)

split = int(0.9 * len(sentences))
train_sents = sentences[:split]
valid_sents = sentences[split:]

write_sentences(TRAIN_OUT, train_sents)
write_sentences(VALID_OUT, valid_sents)

print(f"✅ Train: {len(train_sents)} phrases")
print(f"✅ Validation: {len(valid_sents)} phrases")
