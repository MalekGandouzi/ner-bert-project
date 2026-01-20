from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_PATH = "models/simple_ner"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

text = "Malek Gandouzi étudie à Polytech Monastir en Tunisie."

results = ner(text)

print("Texte :", text)
print("\nEntités détectées :")
for r in results:
    print(f"- {r['word']} → {r['entity_group']}")
