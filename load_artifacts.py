import os
import pickle

model_path = "model"


with open(os.path.join(model_path, "target_vocab.pkl"), "rb") as f:
    target_vocab = pickle.load(f)

print("Loaded target vocab")

with open(os.path.join(model_path, "inv_target_vocab.pkl"), "rb") as f:
    inv_target_vocab = pickle.load(f)

print("Loaded inverse target vocab")

with open(os.path.join(model_path, "citation_feature_vocab.pkl"), "rb") as f:
    citation_feature_vocab = pickle.load(f)

print("Loaded citation features vocab.")

with open(os.path.join(model_path, "gold_to_id_mapping_dict.pkl"), "rb") as f:
    gold_to_label_mapping = pickle.load(f)

print("Loaded gold citation mapping")

with open(os.path.join(model_path, "gold_citations_dict.pkl"), "rb") as f:
    gold_dict = pickle.load(f)

print("Loaded gold citation L1")

with open(os.path.join(model_path, "non_gold_citations_dict.pkl"), "rb") as f:
    non_gold_dict = pickle.load(f)

print("Loaded non-gold citation L1")
