# -*- coding: utf-8 -*-
import os
import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from mockvox.config import PRETRAINED_PATH
from typing import Optional

class CNHubert(nn.Module):
    def __init__(self):
        super().__init__()
        model_dir = os.path.join(PRETRAINED_PATH, "GPT-SoVITS")
        base_path = os.path.join(model_dir, "chinese-hubert-base")
        if os.path.exists(base_path):...
        else:raise FileNotFoundError(base_path)
        self.model = HubertModel.from_pretrained(base_path, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_path, local_files_only=True
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats