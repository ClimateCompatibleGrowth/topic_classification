from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

language_model_name = "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"
emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(language_model_name, truncate=True)


def tokenize(seq, **kwargs):
    """
    Function to tokenize text using model tokenizer.

    Input:
    seq: string of text

    Output:
    tok_data: dictionary of tokenized text
    """
    tok_data = tokenizer(
        seq, max_length=512, truncation=True, padding="max_length", **kwargs
    )
    return [tok_data["input_ids"], tok_data["attention_mask"]]
