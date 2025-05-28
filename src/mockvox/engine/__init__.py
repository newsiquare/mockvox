import torch
import gc
import torch.multiprocessing as mp
from pathlib import Path
import traceback
from importlib import import_module
import os

from mockvox.config import (
    PROCESS_PATH,
    WEIGHTS_PATH,
    SOVITS_HALF_WEIGHTS_FILE,
    GPT_HALF_WEIGHTS_FILE,
    SOVITS_MODEL_CONFIG,
    GPT_MODEL_CONFIG,
    ASR_PATH
)
from mockvox.utils import (
    get_hparams_from_file, 
    MockVoxLogger, 
    i18n,
    generate_unique_filename
)
from mockvox.engine.v2 import load_asr_data, DataProcessor, FeatureExtractor


class VersionDispatcher:
    """版本分发器工厂类"""
    @staticmethod
    def get_module(version):
        """动态获取对应版本模块"""
        if version not in ['v2', 'v4']:
            version = 'v4'
        return import_module(f'mockvox.engine.{version}')

    @classmethod
    def create_components(cls, version):
        """创建版本相关组件"""
        module = cls.get_module(version)
        return (
            module.TextToSemantic,
            module.SoVITsTrainer,
            module.GPTTrainer
        )

class TrainingPipeline:
    """训练流程抽象基类"""
    def __init__(self, args, components):
        self.args = args
        self.modelID = Path(generate_unique_filename(args.fileID)).stem
        
        # 初始化版本相关组件
        (self.TextToSemantic, self.SoVITsTrainer, 
         self.GPTTrainer) = components
        
        # 公共路径
        self.processed_path = Path(PROCESS_PATH) / self.modelID
        self.sovits_weights = Path(WEIGHTS_PATH)/ self.modelID / SOVITS_HALF_WEIGHTS_FILE
        self.gpt_weights = Path(WEIGHTS_PATH)/ self.modelID / GPT_HALF_WEIGHTS_FILE

    def _cleanup(self):
        """资源清理公共方法"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()

    def _prepare_data(self):
        # 从ASR结果中读取language信息
        asr_file = os.path.join(ASR_PATH, self.args.fileID)
        asr_data = load_asr_data(asr_file)
        
        processor = DataProcessor(language=asr_data['language'])
        processor.process(self.args.fileID, self.modelID)
        
        extractor = FeatureExtractor()
        extractor.extract(self.args.fileID, self.modelID, 
                         denoised=self.args.denoise)
        
        t2s = self.TextToSemantic()
        t2s.process(self.args.fileID, self.modelID)
        
        # 清理中间对象
        del processor, extractor, t2s
        self._cleanup()

    def _train_sovits(self):
        """SoVITS训练阶段"""
        mp.set_start_method('spawn', force=True)
        hps = get_hparams_from_file(SOVITS_MODEL_CONFIG)
        hps.data.processed_dir = self.processed_path
        
        trainer = self.SoVITsTrainer(hparams=hps)
        trainer.train(epochs=self.args.epochs_sovits)
        
        del trainer, hps
        self._cleanup()

    def _train_gpt(self):
        """GPT训练阶段"""
        hps = get_hparams_from_file(GPT_MODEL_CONFIG)
        hps.data.semantic_path = self.processed_path / 'name2text.json'
        hps.data.phoneme_path = self.processed_path / 'text2semantic.json'
        hps.data.bert_path = self.processed_path / 'bert'
        
        trainer = self.GPTTrainer(hparams=hps)
        trainer.train(epochs=self.args.epochs_gpt)
        
        del trainer, hps
        self._cleanup()

    def execute(self):
        """执行完整训练流程"""
        try:
            self._prepare_data()
            self._train_sovits()
            self._train_gpt()
            
            MockVoxLogger.info(
                f"{i18n('训练完成')}.\n"
                f"Model ID: {self.modelID}\n"
                f"SoVITS checkpoint: {self.sovits_weights}\n"
                f"GPT checkpoint: {self.gpt_weights}"
            )
            return self.modelID
        
        except Exception as e:
            MockVoxLogger.error(
                f"{i18n('训练过程错误')}: {self.modelID}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

class ResumingPipeline:
    """继续训练流程抽象基类"""
    def __init__(self, args, components):
        self.args = args
        self.modelID = args.modelID
        
        # 初始化版本相关组件
        (self.TextToSemantic, self.SoVITsTrainer, 
         self.GPTTrainer) = components
        
        # 公共路径
        self.processed_path = Path(PROCESS_PATH) / self.modelID
        self.sovits_weights = Path(WEIGHTS_PATH) / self.modelID / SOVITS_HALF_WEIGHTS_FILE
        self.gpt_weights = Path(WEIGHTS_PATH) / self.modelID / GPT_HALF_WEIGHTS_FILE

    def _cleanup(self):
        """资源清理公共方法"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()

    def _train_sovits(self):
        """SoVITS训练阶段"""
        mp.set_start_method('spawn', force=True)
        hps = get_hparams_from_file(SOVITS_MODEL_CONFIG)
        hps.data.processed_dir = self.processed_path
        
        trainer = self.SoVITsTrainer(hparams=hps)
        trainer.train(epochs=self.args.epochs_sovits)
        
        del trainer, hps
        self._cleanup()

    def _train_gpt(self):
        """GPT训练阶段"""
        hps = get_hparams_from_file(GPT_MODEL_CONFIG)
        hps.data.semantic_path = self.processed_path / 'name2text.json'
        hps.data.phoneme_path = self.processed_path / 'text2semantic.json'
        hps.data.bert_path = self.processed_path / 'bert'
        
        trainer = self.GPTTrainer(hparams=hps)
        trainer.train(epochs=self.args.epochs_gpt)
        
        del trainer, hps
        self._cleanup()

    def execute(self):
        """执行继续训练流程"""
        try:
            self._train_sovits()
            self._train_gpt()
            
            MockVoxLogger.info(
                f"{i18n('训练完成')}.\n"
                f"Model ID: {self.modelID}\n"
                f"SoVITS checkpoint: {self.sovits_weights}\n"
                f"GPT checkpoint: {self.gpt_weights}"
            )
        except Exception as e:
            MockVoxLogger.error(
                f"{i18n('训练过程错误')}: {self.modelID}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise