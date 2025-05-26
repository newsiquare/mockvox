# -*- coding: utf-8 -*-
"""文本到语义"""
import os
from typing import Optional, List
import json
from pathlib import Path
import torch

from mockvox.models.v2 import SynthesizerTrn
from mockvox.config import ASR_PATH, PROCESS_PATH, SOVITS_MODEL_CONFIG, PRETRAINED_S2G_FILE
from mockvox.utils import get_hparams_from_file, MockVoxLogger
from .asr import load_asr_data

class TextToSemantic:
    def __init__(
            self,
            device: Optional[str] = None  # 指定计算设备
        ):
        self.hps = get_hparams_from_file(SOVITS_MODEL_CONFIG)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model            
        ).to(self.device)

        self.vq_model.eval()

        try:
            self.vq_model.load_state_dict(
                torch.load(PRETRAINED_S2G_FILE, map_location="cpu")["weight"], strict=False
            )
        except FileNotFoundError:
            MockVoxLogger.error(f"Pretrained model not found: {PRETRAINED_S2G_FILE}")
    
    def process(self, file_id: str, model_id: str) -> List:
        results = []
        # 路径配置
        asr_dir = Path(ASR_PATH) / file_id
        processed_dir = Path(PROCESS_PATH) / model_id
        semantic_file = processed_dir / "text2semantic.json"
        # 已处理
        if semantic_file.exists():
            MockVoxLogger.info(
                "Text to semantic has been done",
                extra={
                    "action": "text_to_semantic",
                    "file_id": file_id,
                    "json_file": semantic_file
                }
            )
            return None

        hubert_dir = processed_dir / "cnhubert"

        # 处理文本转语义
        asr_data = load_asr_data(asr_dir)
        lines = asr_data["results"]    

        for line in lines:
            hubert_file = hubert_dir / f"{line['key']}.pt"
            if not hubert_file.exists():
                MockVoxLogger.warning(f"Feature file not found: {hubert_file}")
                continue

            ssl_content = torch.load(hubert_file, map_location="cpu").to(self.device)
            codes = self.vq_model.extract_latent(ssl_content)
            semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
            result_item = {
                "key": line['key'],
                "semantic": semantic
            }
            results.append(result_item)

        with open(semantic_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        MockVoxLogger.info(
            "Text to semantic done",
            extra={
                "action": "text_to_semantic",
                "file_id": file_id,
                "json_file": semantic_file
            }
        )
        return results
      
if __name__ == '__main__':
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='processed file id.')
    parser.add_argument('model', type=str, help='model id.')
    args = parser.parse_args()

    t2s = TextToSemantic()
    results = t2s.process(args.file, args.model)
    print(results)
