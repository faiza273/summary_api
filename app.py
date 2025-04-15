from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, T5Tokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from pathlib import Path

# Setup base path
BASE_DIR = Path(__file__).absolute().parent

# Load tokenizers
try:
    extract_tok = AutoTokenizer.from_pretrained(str(BASE_DIR / "bert_tokenizer"), local_files_only=True)
    abs_tok = T5Tokenizer.from_pretrained(str(BASE_DIR / "t5_tokenizer"), local_files_only=True)
    print("✅ Tokenizers loaded")
except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    raise

# Download required NLTK models
nltk.download("punkt", quiet=True)

# Initialize Flask
app = Flask(__name__)
CORS(app)

print("🚀 Starting summarization service...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

# Load PyTorch models
try:
    extract_model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_DIR / "distilbert_summary_model"),
        local_files_only=True
    ).to(device).eval()

    abs_model = T5ForConditionalGeneration.from_pretrained(
        str(BASE_DIR / "t5_summary_model"),
        local_files_only=True
    ).to(device).eval()

    print("✅ Models loaded")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    raise

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        # Extractive Summarization
        sentences = sent_tokenize(text)
        inputs = extract_tok(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = extract_model(**inputs)
            logits = outputs.logits
            scores = torch.softmax(logits, dim=1)[:, 1]
            topk_indices = torch.topk(scores, k=min(3, len(sentences))).indices.tolist()

        extracted = [sentences[i] for i in sorted(topk_indices)]

        # Abstractive Summarization
        prompt = "summarize: " + " ".join(extracted)
        inputs = abs_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            output_ids = abs_model.generate(**inputs, max_length=150)

        summary = abs_tok.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({
            "extractive": extracted,
            "abstractive": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
 