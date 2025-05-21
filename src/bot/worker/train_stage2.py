# -*- coding: utf-8 -*-
import traceback
import torch
import torch.multiprocessing as mp
from pathlib import Path
from collections import OrderedDict
import time
from typing import Optional
import gc
from .worker import celeryApp

from bot.utils import BotLogger, i18n
from bot.engine.v2 import (
    DataProcessor as DataProcessorV2,
    FeatureExtractor as FeatureExtractorV2,
    TextToSemantic as TextToSemanticV2
)
from bot.engine.v2.train import (
    SoVITsTrainer as SoVITsTrainerV2, 
    GPTTrainer as GPTTrainerV2
)
from bot.engine.v4 import (
    DataProcessor,
    FeatureExtractor,
    TextToSemantic
)
from bot.engine.v4.train import (
    SoVITsTrainer, 
    GPTTrainer
)  

from bot.config import (
    PROCESS_PATH,
    SOVITS_MODEL_CONFIG,
    GPT_MODEL_CONFIG,
    WEIGHTS_PATH,
    SOVITS_HALF_WEIGHTS_FILE,
    GPT_HALF_WEIGHTS_FILE
)
from bot.utils import get_hparams_from_file

@celeryApp.task(name="train_stage2", bind=True)
def train_task(
    self, 
    file_name: str, 
    sovits_epochs: int,
    gpt_epochs: int,
    version: Optional[str] = 'v4',
    language: Optional[str] = 'zh',
    ifDenoise: Optional[bool] = True
):
    try:
        if(version=='v2'):
            train_v2(file_name, sovits_epochs, gpt_epochs, language, ifDenoise)
        elif(version=='v4'):
            train_v4(file_name, sovits_epochs, gpt_epochs, language, ifDenoise)
        else:
            BotLogger.error(f"Unsupported version: {version}")
            return     
    except Exception as e:
        BotLogger.error(
            f"{i18n('训练过程错误')}: {file_name}\nTraceback:\n{traceback.format_exc()}"
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)
    
def train_v4(file_name, sovits_epochs, gpt_epochs, language, ifDenoise):
    processor = DataProcessor(language)
    processor.process(file_name)
    extractor = FeatureExtractor()
    extractor.extract(file_path=file_name, denoised=ifDenoise)
    t2s = TextToSemantic()
    t2s.process(file_name)
    del processor, extractor, t2s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()

    mp.set_start_method('spawn', force=True)  # 强制使用 spawn 模式
    hps_sovits = get_hparams_from_file(SOVITS_MODEL_CONFIG)
    processed_path = Path(PROCESS_PATH) / file_name
    hps_sovits.data.processed_dir = processed_path
    trainer_sovits = SoVITsTrainer(hparams=hps_sovits)
    trainer_sovits.train(epochs=sovits_epochs)

    del trainer_sovits, hps_sovits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()

    hps_gpt = get_hparams_from_file(GPT_MODEL_CONFIG)
    hps_gpt.data.semantic_path = processed_path / 'name2text.json'
    hps_gpt.data.phoneme_path = processed_path / 'text2semantic.json'
    hps_gpt.data.bert_path = processed_path / 'bert'
    trainer_gpt = GPTTrainer(hparams=hps_gpt)
    trainer_gpt.train(epochs=gpt_epochs)
    del trainer_gpt, hps_gpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()

    sovits_half_weights_path = Path(WEIGHTS_PATH) / file_name / SOVITS_HALF_WEIGHTS_FILE
    gpt_half_weights_path = Path(WEIGHTS_PATH) / file_name / GPT_HALF_WEIGHTS_FILE

    results = OrderedDict()
    results["SoVITs Weight"] = Path(sovits_half_weights_path).name
    results["GPT Weight"] = Path(gpt_half_weights_path).name
    return {
        "status": "success",
        "results": results,
        "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }
    
def train_v2(file_name, sovits_epochs, gpt_epochs, language, ifDenoise):
    processor = DataProcessorV2(language)
    processor.process(file_name)
    extractor = FeatureExtractorV2()
    extractor.extract(file_path=file_name, denoised=ifDenoise)
    t2s = TextToSemanticV2()
    t2s.process(file_name)
    del processor, extractor, t2s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()       

    mp.set_start_method('spawn', force=True)  # 强制使用 spawn 模式
    hps_sovits = get_hparams_from_file(SOVITS_MODEL_CONFIG)
    processed_path = Path(PROCESS_PATH) / file_name
    hps_sovits.data.processed_dir = processed_path
    trainer_sovits = SoVITsTrainerV2(hparams=hps_sovits)
    trainer_sovits.train(epochs=sovits_epochs)

    del trainer_sovits, hps_sovits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()

    hps_gpt = get_hparams_from_file(GPT_MODEL_CONFIG)
    hps_gpt.data.semantic_path = processed_path / 'name2text.json'
    hps_gpt.data.phoneme_path = processed_path / 'text2semantic.json'
    hps_gpt.data.bert_path = processed_path / 'bert'
    trainer_gpt = GPTTrainerV2(hparams=hps_gpt)
    trainer_gpt.train(epochs=gpt_epochs)
    del trainer_gpt, hps_gpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()

    sovits_half_weights_path = Path(WEIGHTS_PATH) / file_name / SOVITS_HALF_WEIGHTS_FILE
    gpt_half_weights_path = Path(WEIGHTS_PATH) / file_name / GPT_HALF_WEIGHTS_FILE

    results = OrderedDict()
    results["SoVITs Weight"] = Path(sovits_half_weights_path).name
    results["GPT Weight"] = Path(gpt_half_weights_path).name
    return {
        "status": "success",
        "results": results,
        "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }