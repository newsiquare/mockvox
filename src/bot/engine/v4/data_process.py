# -*- coding: utf-8 -*-
"""
数据处理器模块，主要负责文本特征提取和预处理
"""
import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
import torch
from typing import List
import json

from bot.config import PRETRAINED_PATH, PROCESS_PATH, ASR_PATH
from bot.text import Normalizer, symbols
from bot.utils import BotLogger
from bot.engine.v2.asr import load_asr_data

# 特殊符号处理配置，格式：(原符号，语言，替换符号)
special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3")
]

class DataProcessor:
    MODEL_MAPPING = {
        "zh": "GPT-SoVITS/chinese-roberta-wwm-ext-large",
        "en": "FacebookAI/roberta-large",
        "ja": "tohoku-nlp/bert-base-japanese-v3",
        "ko": "klue/bert-base",
        "can": "GPT-SoVITS/chinese-roberta-wwm-ext-large"
    }

    def __init__(self, 
                language='zh',
                bert_model: Optional[str]=None,
                device: Optional[str] = None
        ):
        """
        初始化BERT模型处理器
        
        参数:
            bert_model -- 预训练模型名称 (默认: "chinese-roberta-wwm-ext-large")
            device -- 指定运行设备 (默认自动选择GPU/CPU)
        """
        # 加载分词器和语言模型
        if bert_model is None:
            bert_model = self.MODEL_MAPPING.get(language, "GPT-SoVITS/chinese-roberta-wwm-ext-large")

        bert_dir = os.path.join(PRETRAINED_PATH, bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)
        self.mlm = AutoModelForMaskedLM.from_pretrained(bert_dir, local_files_only=True)
        self.language = language
        self.normalizer = Normalizer(language)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mlm.to(self.device)   

    def process(self, file_id, model_id) -> List:
        """
        主处理流程
        
        参数:
            file_id -- 输入文件ID
            model_id -- 输入模型ID
            
        返回:
            List 处理结果列表，元素格式为[key, phones, word2ph, norm_text]
        """
        results = []
        # 路径配置
        asr_dir = os.path.join(ASR_PATH, file_id)
        processed_dir = os.path.join(PROCESS_PATH, model_id)
        bert_dir = os.path.join(processed_dir, "bert")
        Path(bert_dir).mkdir(parents=True, exist_ok=True)
        json_file = os.path.join(processed_dir, 'name2text.json')

         # 已处理
        if os.path.exists(json_file):
            BotLogger.info(
                "Data process has been done",
                extra={
                    "action": "data_processed",
                    "file_id": file_id,
                    "json_file": json_file
                }
            )
            return None  

        # 加载ASR数据
        asr_data = load_asr_data(asr_dir)
        lines = asr_data["results"]        
        # 逐条处理数据
        for line in lines:
            try:
                # 文本清洗
                text = line['text'].replace("%", "-").replace("￥", ",")
                
                # 文本标准化处理
                phones, word2ph, norm_text = self._normalize(text)
                if len(phones)==0: continue
                
                # 保存BERT特征
                bert_file = "%s/%s.pt" % (bert_dir, line['key'])
                bert_feature = self._get_bert_feature(norm_text, word2ph)
                assert bert_feature.shape[-1] == len(phones)
                torch.save(bert_feature, bert_file)
                
                # 构建结果
                result_item = {
                    "key": line['key'],
                    "phones": " ".join(phones),
                    "word2ph": word2ph,  # 保留原始类型 (list/None)
                    "norm_text": norm_text
                }
                results.append(result_item)

            except Exception as e:
                BotLogger.error(
                    f"Data process failed: {file_id} \nException: {str(e)}",
                    extra={"action": "data_process_error"}
                )
                raise RuntimeError(f"Data process failed: {str(e)}") from e
        
        with open(json_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        BotLogger.info(
            "Data process done",
            extra={
                "action": "data_processed",
                "file_id": file_id,
                "json_file": json_file
            }
        )
        return results

    def _normalize(self, text):
        """
        文本标准化处理
        
        参数:
            text -- 原始文本
            language -- 语言类型
            
        返回:
            tuple (phones, word2ph, norm_text)
        """
        # 特殊符号处理
        for special_s, special_l, target_symbol in special:
            if special_s in text and self.language == special_l:
                text = text.replace(special_s, ",")
                norm_text = self.normalizer.do_normalize(text)
                phones = self.normalizer.g2p(norm_text)
                
                # 替换特殊符号
                new_ph = []
                for ph in phones[0]:
                    assert ph in symbols
                    new_ph.append(target_symbol if ph == "," else ph)
                return new_ph, phones[1], norm_text

        # 常规标准化流程
        norm_text = self.normalizer.do_normalize(text)

        # 不同语言的分词处理
        if self.language=="zh" or self.language=="can":
            phones, word2ph = self.normalizer.g2p(norm_text)
            assert len(phones) == sum(word2ph)
            assert len(norm_text) == len(word2ph)
        elif self.language=="en":
            phones = self.normalizer.g2p(norm_text)
            if len(phones) < 4:  # 确保最小长度
                phones = [','] + phones
            word2ph = None
        else:
            phones = self.normalizer.g2p(norm_text)
            word2ph = None
            
        # 未知符号处理
        phones = ['UNK' if ph not in symbols else ph for ph in phones]
        return phones, word2ph, norm_text

    def _get_bert_feature(self, text, word2ph):
        """
        提取文本的BERT特征
        
        参数:
            text -- 输入文本
            word2ph -- 音素到音节的映射关系
            
        返回:
            (Tensor) 手机级别的特征矩阵
        """
        with torch.no_grad():  # 禁用梯度计算
            # 文本编码
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
                
            # 获取隐藏层特征
            res = self.mlm(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        # 验证对齐关系
        assert len(word2ph) == len(text)
        
        # 构建音节重复特征
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        return torch.cat(phone_level_feature, dim=0).T

if __name__ == '__main__':
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='processed file name.')
    args = parser.parse_args()

    processor = DataProcessor()
    results = processor.process(args.file)
    print(results)