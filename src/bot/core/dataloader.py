import os
from typing import Optional
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForMaskedLM

class TextNormalization:
    def __init__(self, 
                bert_model="chinese-roberta-wwm-ext-large",
                device: Optional[str] = None
        ):
        """
            先下载GPT-SoVITS预训练模型：
            ~bash$ modelscope download AI-ModelScope/GPT-SoVITS
        """
        model_dir = snapshot_download("AI-ModelScope/GPT-SoVITS", local_files_only=True)
        self.tokenizer = AutoTokenizer(model_dir)
        self.mlm = AutoModelForMaskedLM(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = self.mlm(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T