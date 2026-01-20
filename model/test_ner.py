from transformers import pipeline

# Charger un pipeline NER pré-entraîné
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

text = "Malek Gandouzi étudie à Polytech Monastir en Tunisie."

results = ner_pipeline(text)

print("Texte :", text)
print("\nEntités détectées :")
for ent in results:
    print(f"- {ent['word']} → {ent['entity_group']}")
