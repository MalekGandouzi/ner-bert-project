import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "models/ner_improved"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

def ner_predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    words = []
    labels = []

    current_word = ""
    current_label = None

    for token, pred in zip(tokens, predictions):
        if token in ["[CLS]", "[SEP]"]:
            continue

        label = model.config.id2label[pred]

        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                words.append(current_word)
                labels.append(current_label)

            current_word = token
            current_label = label

    if current_word:
        words.append(current_word)
        labels.append(current_label)

    return list(zip(words, labels))


# TEST
sentence = "Polytech Monastir est située en Tunisie"
print(ner_predict(sentence))
