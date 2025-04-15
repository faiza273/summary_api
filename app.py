import os
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

EXTRACT_MODEL_DIR = "distilbert_summary_model"
ABS_MODEL_DIR = "t5_summary_model"

# Google Drive folders
EXTRACT_MODEL_DRIVE_URL = "https://drive.google.com/drive/folders/1fvSmJHQgARznfeUNIujzX2d-eCU7pUwJ"
ABS_MODEL_DRIVE_URL = "https://drive.google.com/drive/folders/1icLqy1IVVuwXi37B6ji_Tk4BrlCLdrkt"

# Download model folders from Google Drive if not found locally
if not os.path.exists(EXTRACT_MODEL_DIR):
    print("📥 Downloading distilBERT model from Drive...")
    gdown.download_folder(EXTRACT_MODEL_DRIVE_URL, output=EXTRACT_MODEL_DIR, quiet=False, use_cookies=False)

if not os.path.exists(ABS_MODEL_DIR):
    print("📥 Downloading T5 model from Drive...")
    gdown.download_folder(ABS_MODEL_DRIVE_URL, output=ABS_MODEL_DIR, quiet=False, use_cookies=False)

# Load tokenizers
bert_tokenizer = AutoTokenizer.from_pretrained("bert_tokenizer")
t5_tokenizer = T5Tokenizer.from_pretrained("t5_tokenizer")

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

try:
    extract_model = AutoModelForSequenceClassification.from_pretrained(EXTRACT_MODEL_DIR).to(device)
    abs_model = T5ForConditionalGeneration.from_pretrained(ABS_MODEL_DIR).to(device)
except Exception as e:
    print(f"❌ Model loading error: {e}")
    raise

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Extractive summary (simulate sentence selection)
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = extract_model(**inputs).logits
    scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    if len(text.split(".")) < 2:
        extracted_sentences = [text]
    else:
        sentences = text.split(".")
        scored_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
        extracted_sentences = [s.strip() for s, _ in scored_sentences[:2]]

    extracted_text = ". ".join(extracted_sentences)

    # Abstractive summary
    t5_input = f"summarize: {extracted_text}"
    t5_tokens = t5_tokenizer(t5_input, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = abs_model.generate(t5_tokens.input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({
        "extractive": extracted_text,
        "abstractive": summary
    })

if __name__ == "__main__":
    app.run(debug=True)
