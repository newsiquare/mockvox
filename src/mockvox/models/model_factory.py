import threading
import time
import os
import torch
import numpy as np
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model
from io import BytesIO

from mockvox.models import CNHubert
from mockvox.models.v2.SynthesizerTrn import SynthesizerTrn,Generator
from mockvox.models.v2.t2s_model import Text2SemanticDecoder
from mockvox.models.v4.synthesizer import SynthesizerTrnV3
from mockvox.config import PRETRAINED_VOCODER_FILE,PRETRAINED_S2GV4_FILE,PRETRAINED_PATH
from mockvox.utils import MockVoxLogger
from mockvox.nn import mel_spectrogram_torch


# 伪模型类（实际项目中替换为真实的模型类）
class PyTorchModel:
    MODEL_MAPPING = {
        "zh": "GPT-SoVITS/chinese-roberta-wwm-ext-large",
        "en": "FacebookAI/roberta-large",
        "ja": "tohoku-nlp/bert-large-japanese-v2",
        "ko": "klue/roberta-large",
        "can": "GPT-SoVITS/chinese-roberta-wwm-ext-large"
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def __init__(self, name: str):
        self.name = name
        self._load_model()
    
    def _load_model(self):

        """加载模型到GPU"""
        MockVoxLogger.info(f"Loading model '{self.name}' to GPU...")
        # 模拟加载大型模型到GPU
        if self.name == "hifigan_model":
            hifigan_model = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,is_bias=True
            )
            hifigan_model.eval()
            hifigan_model.remove_weight_norm()
            state_dict_g = torch.load(PRETRAINED_VOCODER_FILE, map_location="cpu")
            hifigan_model.load_state_dict(state_dict_g)
            self.tensor = hifigan_model.half().to(self.device)
        if self.name[:15] == "bert_tokenizer_":
            bert_path = os.path.join(PRETRAINED_PATH,self.MODEL_MAPPING.get(self.name[15:], "GPT-SoVITS/chinese-roberta-wwm-ext-large"))
            self.tensor = AutoTokenizer.from_pretrained(bert_path)
        if self.name[:11] == "bert_model_":
            bert_path = os.path.join(PRETRAINED_PATH,self.MODEL_MAPPING.get(self.name[11:], "GPT-SoVITS/chinese-roberta-wwm-ext-large"))
            bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
            self.tensor = bert_model.half().to(self.device)
        if self.name[:15] == "zero_wav_torch_":
            zero_wav = np.zeros(
                int(float(self.name[15:]) * 0.3),
                dtype=np.float16,
            )        
            zero_wav_torch = torch.from_numpy(zero_wav)
            self.tensor = zero_wav_torch.half().to(self.device)
        if self.name == "ssl_model":
            ssl_model = CNHubert()
            ssl_model.eval()
            self.tensor = ssl_model.half().to(self.device)
        if self.name[:10] == "gpt_model_":
            dict_s1 = torch.load(self.name[10:], map_location="cpu")
            self.config = dict_s1["config"]
            self.max_sec = self.config["data"]["max_sec"]
            t2s_model = Text2SemanticDecoder(config=self.config, top_k=3)
            
            state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v 
                    for k, v in dict_s1["weight"].items()}
            t2s_model.load_state_dict(state_dict)
            t2s_model = t2s_model.half()
            self.tensor = t2s_model.to(self.device)
            self.tensor.eval()
            del dict_s1
        if self.name[:13] == "sovits_model_":
            dict_s2, if_lora_v3 = self._load_sovits_new(self.name[13:])
            self.hps = dict_s2["config"]
            self.hps = DictToAttrRecursive(self.hps)
            self.hps.model.semantic_frame_rate = "25hz"
            if self.hps.model.version == "v4":
                self.tensor = SynthesizerTrnV3(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length_v4,
                    n_speakers=self.hps.data.n_speakers,
                    **self.hps.model
                )
            else:
                self.tensor = SynthesizerTrn(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    n_speakers=self.hps.data.n_speakers,
                    **self.hps.model
                )
                if_lora_v3=False
            self.tensor = self.tensor.half().to(self.device)
            self.tensor.eval()
            if if_lora_v3 == False:
                self.tensor.load_state_dict(dict_s2["weight"], strict=False)
            else:
                gv4_model,if_lora_v3 = self._load_sovits_new(PRETRAINED_S2GV4_FILE)
                self.tensor.load_state_dict(gv4_model["weight"], strict=False)
                lora_rank = self.hps["train"]["lora_rank"]
                lora_config = LoraConfig(
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    init_lora_weights=True,
                )
                self.tensor.cfm = get_peft_model(self.tensor.cfm, lora_config)
                self.tensor.load_state_dict(dict_s2["weight"], strict=False)
                self.tensor.cfm = self.tensor.cfm.merge_and_unload()
                self.tensor.eval()
            self.mel_fn_v4 = lambda x: mel_spectrogram_torch(
                x,
                **{
                    "n_fft": self.hps["data"]["filter_length"],
                    "win_size": self.hps["data"]["win_length"],
                    "hop_size": self.hps["data"]["hop_length_v4"],
                    "num_mels": self.hps["data"]["n_mel_channels_v4"],
                    "sampling_rate": self.hps["data"]["sampling_rate"],
                    "fmin": self.hps["data"]["mel_fmin"],
                    "fmax": self.hps["data"]["mel_fmax"],
                    "center": False,
                },
            )
        
        self.is_loaded = True
        
    def release(self):
        """释放模型资源"""
        if hasattr(self, 'tensor') and self.is_loaded:
            MockVoxLogger.info(f"Releasing model '{self.name}' from GPU")
            if self.name[:13] == "sovits_model_":
                del self.mel_fn_v4
                del self.tensor
            else:
                del self.tensor
            torch.cuda.empty_cache()
            self.is_loaded = False
    
    def _load_sovits_new(self, path_sovits):
        f = open(path_sovits, "rb")
        if_lora_v3 = False
        meta = f.read(2)
        if meta != "PK":
            if_lora_v3 = True
            data = b"PK" + f.read()
            bio = BytesIO()
            bio.write(data)
            bio.seek(0)
            return torch.load(bio, map_location="cpu"),if_lora_v3
        return torch.load(path_sovits, map_location="cpu"),if_lora_v3

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

# 模型工厂
class ModelFactory:
    _models: Dict[str, Any] = {}
    _lock = threading.Lock()
    _preloaded_models = ["bert_tokenizer_zh", "bert_tokenizer_ja","bert_tokenizer_ko","bert_model_zh","bert_model_ja","bert_model_ko", "ssl_model", "hifigan_model"]
    
    @classmethod
    def load_model(cls, model_name: str):
        """创建模型实例"""
        with cls._lock:
            if model_name not in cls._models:
                MockVoxLogger.info(f"Creating model instance: {model_name}")
                model = PyTorchModel(model_name)
                cls._models[model_name] = {
                    "instance": model,
                    "last_used": time.time(),
                    "ref_count": 0,
                    "is_loaded": True
                }
            return cls._models[model_name]["instance"]
    
    @classmethod
    def get_model(cls, model_name: str):
        """获取模型实例并更新使用信息"""
        with cls._lock:
            # 如果模型不存在则创建
            if model_name not in cls._models:
                MockVoxLogger.info(f"Model '{model_name}' not preloaded, loading now")
                model = PyTorchModel(model_name)
                cls._models[model_name] = {
                    "instance": model,
                    "last_used": time.time(),
                    "ref_count": 1,
                    "is_loaded": True
                }
                return model
            
            model_data = cls._models[model_name]
            model_data["last_used"] = time.time()
            model_data["ref_count"] += 1
            
            # 如果模型已卸载，重新加载
            if not model_data["instance"].is_loaded:
                MockVoxLogger.info(f"Reloading model '{model_name}' as it was unloaded")
                model_data["instance"]._load_model()
                model_data["is_loaded"] = True
            
            return model_data["instance"]
    
    @classmethod
    def release_model(cls, model_name: str):
        """释放模型引用计数"""
        with cls._lock:
            if model_name in cls._models:
                cls._models[model_name]["ref_count"] -= 1
    
    @classmethod
    def cleanup_unused_models(cls):
        """清理1分钟未使用的模型"""
        current_time = time.time()
        models_to_release = []
        
        with cls._lock:
            for model_name, data in cls._models.items():
                # 预加载模型不释放，只卸载GPU资源
                if model_name in cls._preloaded_models:
                    # 检查条件：引用计数为0且1分钟未使用
                    if data["ref_count"] == 0 and current_time - data["last_used"] > 60:
                        if data["instance"].is_loaded:
                            MockVoxLogger.info(f"Marking preloaded model '{model_name}' for GPU release")
                            models_to_release.append(model_name)
                else:
                    # 动态加载的模型直接删除实例
                    if data["ref_count"] == 0 and current_time - data["last_used"] > 60:
                        MockVoxLogger.info(f"Removing dynamically loaded model '{model_name}'")
                        models_to_release.append(model_name)
            
            for model_name in models_to_release:
                # 预加载模型只释放GPU资源，保留实例
                if model_name in cls._preloaded_models:
                    if model_name in cls._models and cls._models[model_name]["instance"].is_loaded:
                        cls._models[model_name]["instance"].release()
                        cls._models[model_name]["is_loaded"] = False
                else:
                    # 动态加载模型完全删除
                    data = cls._models.pop(model_name)
                    data["instance"].release()
        
        return len(models_to_release)