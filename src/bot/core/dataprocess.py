# -*- coding: utf-8 -*-
"""
数据处理器模块，主要负责文本特征提取和预处理
"""
import os
import ast
from typing import Optional
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForMaskedLM
from bot.config import PRETRAINED_DIR, PROCESS_PATH, ASR_PATH
from bot.text import Normalizer
from pathlib import Path
import torch
from typing import List
import json

# 特殊符号处理配置，格式：(原符号，语言，替换符号)
special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3")
]

class DataProcessor:
    def __init__(self, 
                bert_model="chinese-roberta-wwm-ext-large",
                device: Optional[str] = None
        ):
        """
        初始化BERT模型处理器
        
        参数:
            bert_model -- 预训练模型名称 (默认: "chinese-roberta-wwm-ext-large")
            device -- 指定运行设备 (默认自动选择GPU/CPU)
        
        注意: 首次使用需通过以下命令下载模型：
            modelscope download --model 'AI-ModelScope/GPT-SoVITS' --local_dir './pretrained/AI-ModelScope/GPT-SoVITS'
        """
        # 从ModelScope下载模型
        model_dir = snapshot_download("AI-ModelScope/GPT-SoVITS", cache_dir=PRETRAINED_DIR)
        
        # 加载分词器和语言模型
        model_dir = os.path.join(model_dir, 'chinese-roberta-wwm-ext-large')
        self.tokenizer = AutoTokenizer(model_dir)
        self.mlm = AutoModelForMaskedLM(model_dir)
        
        # 设备配置（优先使用GPU）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mlm.to(self.device)

    def get_bert_feature(self, text, word2ph):
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
            res = self.mlm(​**​inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        # 验证对齐关系
        assert len(word2ph) == len(text)
        
        # 构建频谱特征
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        return torch.cat(phone_level_feature, dim=0).T

    @staticmethod
    def load_asr_data(asr_file):
        """读取ASR结果文件并还原为列表"""
        result = []
        with open(asr_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    # 使用 ast.literal_eval 安全转换字符串为字典
                    result.append(ast.literal_eval(line))
                except SyntaxError as e:
                    print(f"格式错误的行: {line}\n错误信息: {e}")
        return result

    def process(self, file_name) -> List:
        """
        主处理流程
        
        参数:
            file_name -- 输入文件名（不带后缀）
            
        返回:
            List 处理结果列表，元素格式为[key, phones, word2ph, norm_text]
        """
        results = []
        # 路径配置
        asr_dir = os.path.join(ASR_PATH, file_name)
        processed_dir = os.path.join(PROCESS_PATH, file_name)
        bert_dir = os.path.join(processed_dir, "bert")
        Path(bert_dir).mkdir(parents=True, exist_ok=True)
        json_file = os.path.join(processed_dir, 'name2text.json')

        # 加载ASR数据
        asr_file = os.path.join(asr_dir, 'output.txt')
        lines = self.load_asr_data(asr_file)
        
        # 逐条处理数据
        for line in lines:
            try:
                # 文本清洗
                text = line['text'].replace("%", "-").replace("￥", ",")
                
                # 文本标准化处理
                phones, word2ph, norm_text = self.normalize(text, 'zh')
                
                # 保存BERT特征
                bert_file = "%s/%s.pt" % (bert_dir, line['key'])
                bert_feature = self.get_bert_feature(norm_text, word2ph)
                assert bert_feature.shape[-1] == len(phones)
                torch.save(bert_feature, bert_file)
                
                # 构建结果
                result_item = {
                    "key": line['key'],
                    "phones": " ".join(phones),
                    "word2ph": word2ph,  # 保留原始类型（list/None）
                    "norm_text": norm_text
                }
                results.append(result_item)

            except Exception as e:
                # 错误处理（假设BotLogger已定义）
                BotLogger.error(
                    f"数据处理异常 | 文件: {file_name} | 错误: {str(e)}",
                    extra={"action": "data_process_error"}
                )
                raise RuntimeError(f"数据处理失败: {str(e)}") from e
        
        with open(json_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results

    def normalize(self, text, language):
        """
        文本标准化处理
        
        参数:
            text -- 原始文本
            language -- 语言类型
            
        返回:
            tuple (phones, word2ph, norm_text)
        """
        self.normlizer = Normalizer(language)
        
        # 特殊符号处理
        for special_s, special_l, target_symbol in special:
            if special_s in text and language == special_l:
                text = text.replace(special_s, ",")
                norm_text = self.normalizer.normalize(text)
                phones = self.normlizer.g2p(norm_text)
                
                # 替换特殊符号
                new_ph = []
                for ph in phones[0]:
                    assert ph in symbols
                    new_ph.append(target_symbol if ph == "," else ph)
                return new_ph, phones[1], norm_text

        # 常规标准化流程
        norm_text = self.normlizer.normalize(text)
        
        # 不同语言的分词处理
        if language == "zh" or language=="cant":
            phones, word2ph = self.normlizer.g2p(norm_text)
            assert len(phones) == sum(word2ph)
            assert len(norm_text) == len(word2ph)
        elif language == "en":
            phones = self.normlizer.g2p(norm_text)
            if len(phones) < 4:  # 确保最小长度
                phones = [','] + phones
            word2ph = None
        else:
            phones = self.normlizer.g2p(norm_text)
            word2ph = None
            
        # 未知符号处理
        phones = ['UNK' if ph not in symbols else ph for ph in phones]
        return phones, word2ph, norm_text