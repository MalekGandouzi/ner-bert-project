from transformers import AutoTokenizer

# 1. Choisir le tokenizer
MODEL_NAME = "Davlan/bert-base-multilingual-cased-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Exemple phrase + labels (comme dans train.conll)
words = ["Malek", "Gandouzi", "étudie", "à", "Polytech", "Monastir", "en", "Tunisie", "."]
labels = ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O"]

label2id = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6
}

# 3. Tokenisation
encoding = tokenizer(
    words,
    is_split_into_words=True,
    truncation=True,
    return_offsets_mapping=True
)

# 4. Alignement tokens ↔ labels
aligned_labels = []
previous_word_id = None

for word_id in encoding.word_ids():
    if word_id is None:
        aligned_labels.append(-100)
    elif word_id != previous_word_id:
        aligned_labels.append(label2id[labels[word_id]])
    else:
        aligned_labels.append(-100)
    previous_word_id = word_id

# 5. Résultat
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

print("TOKENS:")
print(tokens)

print("\nLABELS ALIGNÉS:")
print(aligned_labels)
