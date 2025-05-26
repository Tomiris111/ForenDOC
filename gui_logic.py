import os
import json
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tkinter import messagebox
from wordcloud import WordCloud

# Load models once
rf_model = joblib.load("rf_encryption_model.pkl")
bilstm_encryption_model = load_model("bilstm_encryption_model_v2.keras")
bilstm_classifier_model = load_model("bilstm_topic_classifier.keras")

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

def run_ml_classification(report_path="forensic_report.json"):
    if not os.path.exists(report_path):
        raise FileNotFoundError("forensic_report.json not found")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    names = list(report.keys())
    texts = [report[name].get("extracted_text", "") for name in names]

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=300)

    def extract_entropy_features(text):
        byte_data = text.encode("utf-8", errors="ignore")
        counter = Counter(byte_data)
        total = len(byte_data)
        entropy = -sum((c / total) * np.log2(c / total) for c in counter.values()) if total > 0 else 0
        uniformity = len(counter) / 256
        printable_ratio = sum(c in range(32, 127) for c in byte_data) / total if total > 0 else 0
        return [entropy, uniformity, printable_ratio]

    X_entropy = np.array([extract_entropy_features(text) for text in texts])
    rf_preds = rf_model.predict(X_entropy)

    bilstm_enc_probs = bilstm_encryption_model.predict(padded)
    bilstm_enc_labels = (bilstm_enc_probs > 0.5).astype(int).flatten()

    bilstm_cls_probs = bilstm_classifier_model.predict(padded)
    topic_labels = {
        0: "Legal", 1: "Fraud", 2: "Medical",
        3: "Darknet", 4: "Religious", 5: "Economic"
    }
    bilstm_cls_labels = np.argmax(bilstm_cls_probs, axis=1)
    bilstm_cls_conf = np.max(bilstm_cls_probs, axis=1)

    for i, name in enumerate(names):
        report[name]["rf_encryption"] = "Encrypted" if rf_preds[i] == 1 else "Not Encrypted"
        report[name]["bilstm_encryption"] = {
            "label": "Encrypted" if bilstm_enc_labels[i] == 1 else "Not Encrypted",
            "probability": round(float(bilstm_enc_probs[i][0]), 3)
        }
        report[name]["bilstm_classification"] = {
            "category": topic_labels[bilstm_cls_labels[i]],
            "confidence": round(float(bilstm_cls_conf[i]) * 100, 2)
        }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

def show_keyword_cloud(json_path="forensic_report.json", bg_color="#e2f5d6"):
    if not os.path.exists(json_path):
        raise FileNotFoundError("forensic_report.json not found")

    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    text = " ".join(info.get("extracted_text", "") for info in report.values())
    if not text.strip():
        messagebox.showinfo("Info", "No extracted text found for word cloud.")
        return

    wc = WordCloud(width=800, height=400, background_color=bg_color, colormap="viridis").generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("Keyword Cloud")
    plt.show()

def show_slot_chart(slots_stats):
    if not slots_stats or slots_stats[1] == 0:
        messagebox.showerror("Error", "No slot data available or total is zero.")
        return

    found, total = slots_stats
    empty = max(0, total - found)

    sizes = [max(0, found), max(0, empty)]
    if sum(sizes) == 0:
        messagebox.showerror("Error", "Both valid and empty slots are zero.")
        return

    labels = ['Valid .docx Slots', 'Empty Slots']
    colors = ['#4CAF50', '#F44336']

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title("DOCX Signature Density in Dump")
    plt.axis("equal")
    plt.show()
