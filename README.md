# 🧠 Reconnaissance d’Entités Nommées (NER) avec BERT

Ce projet réalise un système de **Reconnaissance d’Entités Nommées (Named Entity Recognition – NER)** en utilisant le modèle **BERT** via un **fine-tuning pour la classification de tokens**.
Il permet d’identifier automatiquement certaines entités nommées dans un texte en français à l’aide d’un modèle entraîné, et de les visualiser via une **interface web interactive développée avec Streamlit**.

Projet réalisé dans un cadre **académique** à **Polytech Monastir** (année universitaire 2025–2026).

---

## 🎯 Objectif du projet

L’objectif principal est de :

* Mettre en œuvre un pipeline complet de **fine-tuning de BERT pour la tâche NER**
* Entraîner le modèle sur un dataset annoté au format CoNLL
* Évaluer les performances à l’aide des métriques classiques (precision, recall, F1-score)
* Déployer une **interface graphique simple et moderne** pour tester le modèle sur des textes libres

---

## 🗂️ Structure du projet

```
ner-bert-project/
│
├── data/
│   └── raw/                # Jeux de données (train / valid au format CoNLL)
│
├── model/
│   └── train_simple.py     # Script d'entraînement du modèle BERT NER
│
├── models/
│   └── ner_improved/       # Modèle entraîné sauvegardé
│
├── preprocessing/
│   └── load_data.py        # Chargement et préparation des données
│
├── app.py                  # Interface Streamlit
├── requirements.txt        # Dépendances Python
└── README.md
```

---

## ⚙️ Technologies utilisées

* **Python 3**
* **PyTorch**
* **Hugging Face Transformers**
* **SeqEval** (évaluation NER)
* **Streamlit** (interface graphique)

---

## 🚀 Installation

1️⃣ Créer et activer un environnement virtuel :

```bash
python -m venv venv
venv\Scripts\activate
```

2️⃣ Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 🧠 Entraînement du modèle

L’entraînement du modèle se fait via le script suivant :

```bash
python model/train_simple.py
```

Ce script :

* Charge les données annotées
* Tokenise les textes avec BERT
* Entraîne un modèle `BertForTokenClassification`
* Évalue le modèle à chaque époque
* Sauvegarde le modèle final

---

## 🎨 Interface graphique (Streamlit)

Une interface web permet de tester le modèle sur des textes personnalisés.

### Lancer l’interface :

```bash
streamlit run app.py
```

### Fonctionnalités :

* Saisie libre de texte en français
* Prédiction des entités nommées
* Mise en évidence visuelle des entités détectées

---

## 📸 Aperçu de l’interface

![Interface Streamlit](assets/streamlit.png)

---

## 📊 Résultats

Le modèle montre une **amélioration progressive des performances** au cours de l’entraînement, avec une diminution de la loss et une augmentation du score F1 sur le jeu de validation.
Les résultats restent dépendants de la taille et de la qualité du dataset utilisé.

---

## 👩‍🎓👨‍🎓 Contexte académique

* **Établissement** : Polytech Monastir
* **Filière** : Data Science & Intelligence Artificielle
* **Niveau** : 4ᵉ année
* **Année universitaire** : 2025–2026

---

## ✍️ Auteur

**Malek Gandouzi**
Étudiant en Data Science & Intelligence Artificielle

---

## 📌 Remarque

Ce projet a été réalisé à des fins **pédagogiques et expérimentales** afin de se familiariser avec :

* le fine-tuning de modèles de langage,
* la tâche NER,
* et le déploiement d’un modèle via une interface simple.
