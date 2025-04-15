from transformers import AutoTokenizer, T5Tokenizer

# 1) DistilBERT tokenizer
extract_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
extract_tok.save_pretrained("bert_tokenizer")

# 2) T5 tokenizer
abs_tok = T5Tokenizer.from_pretrained("t5-small")
abs_tok.save_pretrained("t5_tokenizer")

print("✅ Tokenizers saved to bert_tokenizer/ and t5_tokenizer/")
