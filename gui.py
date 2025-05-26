from custom_models import LogisticRegressionCustom
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import joblib
import numpy as np
import tensorflow as tf
from analysis import analyze_files
from slot_scanner import scan_and_create_accurate_fragment
from slot_recovery import recover_docx_from_fragment
from slot_image_recovery import extract_images_from_all_docx
from gui_logic import show_keyword_cloud, show_slot_chart
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Global state
log_box = None
selected_dump_path = ""
slots_stats = (0, 0)

# Styling (light theme)
BG_COLOR = "#f2f2f2"
FG_COLOR = "#000000"
BUTTON_COLOR = "#4CAF50"
FONT = ("Segoe UI", 10)

def browse_file(entry):
    global selected_dump_path
    selected_dump_path = filedialog.askopenfilename(filetypes=[("Dump Files", "*.001 *.bin *")])
    entry.delete(0, tk.END)
    entry.insert(0, selected_dump_path)
    msg = f"Selected dump file: {selected_dump_path}"
    log_box.insert(tk.END, f"[\u2713] {msg}\n")

def extract_fragments():
    global slots_stats
    if not selected_dump_path:
        messagebox.showerror("Error", "No dump file selected")
        return
    valid, total = scan_and_create_accurate_fragment(selected_dump_path)
    slots_stats = (valid, total)
    msg = f"Extracted {valid} valid fragments from {total} scanned slots."
    log_box.insert(tk.END, f"[\u2713] {msg}\n")
    messagebox.showinfo("Done", msg)

def browse_fragment():
    fragment = filedialog.askopenfilename(title="Select Fragment (.bin)", filetypes=[("Fragment", "*.bin")])
    if not fragment:
        return
    log_box.insert(tk.END, f"[\u2713] Selected fragment: {fragment}\n")
    recovered_files = recover_docx_from_fragment(fragment)
    msg = f"Recovered {len(recovered_files)} valid .docx files from fragment."
    log_box.insert(tk.END, f"[\u2713] {msg}\n")
    messagebox.showinfo("Recovery", msg)

def run_analysis():
    recovered_path = "recovered_docs_from_fragment"
    if not os.path.exists(recovered_path):
        messagebox.showerror("Missing", "Recover documents before analysis")
        return
    files = [os.path.join(recovered_path, f) for f in os.listdir(recovered_path)
             if f.endswith(".docx") or f.endswith(".doc")]
    report = analyze_files(files)
    with open("forensic_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    extract_images_from_all_docx()
    msg = "Forensic analysis completed and report saved"
    log_box.insert(tk.END, f"[\u2713] {msg}\n")

def run_ml():
    try:
        rf_model = joblib.load("rf_encryption_model.pkl")
        model_encryption = load_model("bilstm_encryption_model_v2.keras")
        model_classifier = load_model("bilstm_topic_classifier.keras")

        with open("forensic_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)

        names = list(report.keys())
        texts = [report[name].get("extracted_text", "") for name in names]

        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=300)

        def extract_entropy_features(text):
            from collections import Counter
            import math
            byte_data = text.encode("utf-8", errors="ignore")
            counter = Counter(byte_data)
            total = len(byte_data)
            entropy = -sum((c / total) * math.log2(c / total) for c in counter.values()) if total > 0 else 0
            uniformity = len(counter) / 256
            printable_ratio = sum(c in range(32, 127) for c in byte_data) / total if total > 0 else 0
            return [entropy, uniformity, printable_ratio]

        X_entropy = np.array([extract_entropy_features(t) for t in texts])
        rf_preds = rf_model.predict(X_entropy)

        bilstm_preds = model_encryption.predict(padded)
        bilstm_labels = (bilstm_preds > 0.5).astype(int).flatten()

        topic_probs = model_classifier.predict(padded)
        topic_indices = np.argmax(topic_probs, axis=1)
        topic_labels_dict = {
            0: "Legal", 1: "Fraud", 2: "Medical",
            3: "Darknet", 4: "Religious", 5: "Economic"
        }

        for i, name in enumerate(names):
            report[name]["rf_encryption"] = "Encrypted" if rf_preds[i] else "Not Encrypted"
            report[name]["bilstm_encryption"] = "Encrypted" if bilstm_labels[i] else "Not Encrypted"
            report[name]["predicted_topic"] = topic_labels_dict[topic_indices[i]]
            report[name]["topic_confidence"] = f"{round(float(np.max(topic_probs[i])) * 100, 2)}%"

        with open("forensic_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        msg = "ML classification updated and saved"
        log_box.insert(tk.END, f"[\u2713] {msg}\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_keywords():
    try:
        show_keyword_cloud()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_chart():
    try:
        show_slot_chart(slots_stats)
    except Exception as e:
        messagebox.showerror("Chart Error", str(e))

def show_security_policy_window():
    policy_text = """
Security Policy for ForenDOC

This application complies with internationally recognized digital forensics and cybersecurity principles, based on standards such as ISO/IEC 27001, ISO/IEC 27037:2012, and NIST SP 800-101 Rev.1, to ensure responsible handling of recovered data, operational safety, and evidence integrity.

1. Deployment and Execution

The application runs locally via terminal on Windows systems.
No installation is required beyond a Python environment.
No internet connection is used.
All machine learning models are embedded and work fully offline.

2. Read-Only Forensic Operation

The tool operates in read-only mode throughout analysis and recovery.
Memory dumps or disk images are scanned without alteration.
No editing, rewriting, or direct changes to source evidence are permitted.
Aligned with ISO/IEC 27037:2012.

3. Fragment Recovery and Processing

The system identifies and extracts .docx documents based on file signatures.
From recovered .docx files, it extracts embedded images including .jpg, .jpeg, .png, .gif, and .bmp.
This process ensures segregation of evidence and avoids contamination.
Aligned with ISO/IEC 27041:2015.

4. No Persistent Logging or User Tracking

No logs, tracking data, or operator identifiers are stored, either temporarily or permanently.
This ensures that the investigator's identity or actions are not exposed or misused.
Aligned with ISO/IEC 27001 and 27002.

5. Local Machine Learning-Based Classification

The system uses embedded ML models to analyze document content without uploading any data.
A prohibited keyword scanner evaluates possible sensitive topics and generates a local JSON report.
Aligned with ISO/IEC 27018.

6. Secure Output and Export

Users can export results as JSON files, saved only to user-specified folders.
No output is stored outside of user control or beyond the session.

Referenced Standards

ISO/IEC 27001: Information Security Management Systems
ISO/IEC 27002: Information Security Controls
ISO/IEC 27037:2012: Guidelines for handling digital evidence
ISO/IEC 27041:2015: Ensuring forensic method suitability
NIST SP 800-101 Rev.1: Guidelines for mobile and memory device forensics

If you have questions about our policy or wish to report a security issue, please contact us.
"""
    window = tk.Toplevel()
    window.title("Security Policy & Contact")
    window.geometry("700x500")
    window.configure(bg=BG_COLOR)

    tk.Label(window, text="Security Policy", font=("Segoe UI", 14, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(pady=(15, 5))
    frame = tk.Frame(window, bg=BG_COLOR)
    frame.pack(expand=True, fill="both", padx=20, pady=(0, 10))

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    text_widget = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set, bg="white", fg="black", font=FONT)
    text_widget.insert(tk.END, policy_text)
    text_widget.config(state="disabled")
    text_widget.pack(side="left", fill="both", expand=True)

    scrollbar.config(command=text_widget.yview)
    tk.Label(window, text="Contact Us", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(pady=(5, 0))
    tk.Label(window, text="📧 tommyzh531@gmail.com", font=FONT, bg=BG_COLOR, fg="blue").pack(pady=(0, 10))

def download_report():
    report_file = "forensic_report.json"
    if not os.path.exists(report_file):
        messagebox.showerror("Error", "No forensic report found")
        return

    save_path = filedialog.asksaveasfilename(
        title="Download Forensic Report",
        defaultextension=".json",
        filetypes=[("JSON Files", "*.json")]
    )
    if save_path:
        with open(report_file, "r", encoding="utf-8") as src:
            content = src.read()
        with open(save_path, "w", encoding="utf-8") as dst:
            dst.write(content)
        messagebox.showinfo("Success", f"Report saved to: {save_path}")

def launch_gui():
    global log_box
    root = tk.Tk()
    root.title("ForenDOC - Document Recovery Forensics")
    root.configure(bg=BG_COLOR)

    top_frame = tk.Frame(root, bg=BG_COLOR)
    top_frame.pack(pady=10)
    tk.Label(top_frame, text="Disk dump path:", bg=BG_COLOR, fg=FG_COLOR, font=FONT).pack(side=tk.LEFT)
    path_entry = tk.Entry(top_frame, width=60)
    path_entry.pack(side=tk.LEFT, padx=5)
    tk.Button(top_frame, text="Browse", command=lambda: browse_file(path_entry), bg="darkgreen", fg="white", font=FONT).pack(side=tk.LEFT)

    center = tk.Frame(root, bg=BG_COLOR)
    center.pack(pady=5)

    def add_main_button(text, command):
        tk.Button(center, text=text, command=command, font=FONT, bg=BUTTON_COLOR, fg="white", width=40, height=2).pack(pady=4)

    add_main_button("1. Extract DOCX Signatures", extract_fragments)
    add_main_button("2. Recover from Fragment", browse_fragment)
    add_main_button("3. Run Forensic Analysis", run_analysis)
    add_main_button("4. Run ML Classification", run_ml)

    side = tk.Frame(root, bg=BG_COLOR)
    side.place(x=0, y=0)

    def add_side_button(label, command):
        tk.Button(side, text=label, command=command, font=FONT, bg="#cccccc", fg="black", width=15).pack(pady=1, anchor="w")

    add_side_button("Show Keywords", show_keywords)
    add_side_button("Show Chart", show_chart)
    add_side_button("Download Report", download_report)
    add_side_button("Security Policy", show_security_policy_window)
    


    log_box_frame = tk.Frame(root, bg=BG_COLOR)
    log_box_frame.pack(pady=10)
    log_box = tk.Text(log_box_frame, height=10, width=100, bg="white", fg="black")
    log_box.pack()

    root.mainloop()

if __name__ == "__main__":
    launch_gui()