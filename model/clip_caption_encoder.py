import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPCaptionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", out_dim=324):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, out_dim),
            nn.ReLU(True)
        )
        # spaCy cho preprocessing
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def preprocess(self, captions):
        """
        captions: list[str]
        return: list[str] đã clean
        """
        clean_captions = []
        for cap in captions:
            doc = self.spacy_nlp(cap.lower())
            tokens = [tok.lemma_ for tok in doc if not tok.is_stop and tok.is_alpha]
            clean_captions.append(" ".join(tokens))
        return clean_captions

    def forward(self, captions):
        device = next(self.parameters()).device
        clean_captions = self.preprocess(captions)

        inputs = self.tokenizer(clean_captions, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():  # freeze CLIP text encoder
            outputs = self.text_model(**inputs)
            text_emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling

        return self.mlp(text_emb)
