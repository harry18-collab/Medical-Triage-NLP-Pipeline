# 🏥 Medical Triage NLP Pipeline

> Automatically routing patient descriptions to the right medical specialty using a fine-tuned BERT + BiLSTM ensemble — because the right doctor at the right time matters.

---

## What This Project Does

Ever wondered how hospitals could use AI to figure out which specialist a patient needs — just from what they describe? That's exactly what this project tackles.

Given a patient's clinical transcription or symptom description, this pipeline predicts the correct medical specialty (like Cardiology, Neurology, Orthopedics, etc.) with high accuracy. It's the kind of thing that could help reduce misrouting in busy hospitals or power smarter healthcare chatbots.

---

## How It Works

At the core, there are two models working together:

**1. BiLSTM with Attention**
A Bidirectional LSTM that reads the patient text in both directions and uses an attention mechanism to focus on the most clinically relevant words. Think of it as a model that knows "chest pain" and "shortness of breath" matter more than "the patient said."

**2. Fine-tuned BioBERT**
BioBERT is BERT pre-trained specifically on biomedical literature — so it already understands medical language before we even start training. We fine-tuned all 12 transformer layers on our dataset for maximum accuracy.

**3. Confidence-Gated Ensemble**
The final prediction isn't just one model's opinion. Both models vote, weighted by their confidence. If BioBERT is very confident, it gets more say. If both are unsure, a smart keyword fallback kicks in for edge cases like mental health or ENT.

---

## Results

| Model | Test Accuracy | Weighted F1 |
|-------|--------------|-------------|
| BiLSTM (baseline) | ~81% | ~0.80 |
| Fine-tuned BioBERT | ~85.94% | ~0.85 |
| **Confidence-Gated Ensemble** | **85.94%** | **Best** |

The ensemble outperforms the BiLSTM baseline by **4.82%** — which is meaningful in a medical context where wrong routing has real consequences.

---

## Dataset

- **Primary:** [Medical Transcriptions (Kaggle)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) — 1,660+ real clinical transcriptions
- **Secondary:** [Medical Speech Transcription and Intent (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)

Both datasets were combined, cleaned, and normalized into 6 clinically coherent specialty classes. Administrative categories (discharge summaries, lab reports) were dropped since they don't map to patient symptoms.

---

## Tech Stack

```
Language        : Python 3
Deep Learning   : PyTorch
Transformers    : HuggingFace (BioBERT - dmis-lab/biobert-base-cased-v1.2)
NLP             : NLTK (tokenization, lemmatization, stopwords)
Data            : Pandas, NumPy
Visualization   : Matplotlib, Seaborn
Evaluation      : Scikit-learn (F1, confusion matrix, classification report)
Platform        : Kaggle Notebooks
```

---

## Key Design Decisions (and Why)

- **Why BioBERT over regular BERT?** BioBERT was pre-trained on PubMed and clinical notes — it already knows what "myocardial infarction" means. Regular BERT doesn't.

- **Why full fine-tuning?** Freezing transformer layers hurt recall on minority classes like Neurology. Unfreezing all 12 layers gave the model freedom to adapt fully.

- **Why confidence-gated ensemble?** Neither model is perfect alone. Letting each model contribute based on its own confidence — rather than equal weighting — gave better results on edge cases.

- **Why AdamW + class-weighted loss?** The dataset is imbalanced (lots of Orthopedics, fewer Neurology). Class weighting ensures the model doesn't just predict the majority class.

- **Why lemmatization over stemming?** Clinical text needs real words, not truncated stems. "Prescribed" shouldn't become "prescrib."

---

## Project Structure

```
├── notebook.ipynb          # Full pipeline (EDA → Training → Evaluation)
├── best_bert_model.pt      # Saved BioBERT weights
├── best_lstm_model.pt      # Saved BiLSTM weights
├── vocab.pkl               # Vocabulary for LSTM
├── label_encoder.pkl       # Label encoder (specialty names)
├── class_names.pkl         # List of specialty class names
└── README.md               # This file
```

---

## How to Run

1. Clone the repo and open the notebook on Kaggle or Google Colab
2. Set up your Kaggle API credentials (for dataset download)
3. Run cells sequentially — Cell 1 installs dependencies, restart runtime after
4. Training takes ~30-45 mins on GPU (Kaggle P100 recommended)

```python
# Quick inference example
text = "Patient complains of severe chest pain radiating to left arm with shortness of breath"
predicted_specialty = predict_triage(text)
print(predicted_specialty)  # → Cardiology
```

---

## What I Learned

Building this taught me a lot about real NLP challenges:
- Medical text is noisy, inconsistent, and full of abbreviations
- Class imbalance isn't just a number problem — it breaks models in subtle ways
- Ensemble methods shine when individual models have complementary weaknesses
- Full fine-tuning vs. partial freezing makes a measurable difference in practice

---

## About

Built as part of my exploration into NLP for healthcare applications.  
**Stack:** Python • PyTorch • HuggingFace Transformers • NLTK • Scikit-learn

---
