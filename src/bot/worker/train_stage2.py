import traceback
import torch.multiprocessing as mp
from pathlib import Path
from collections import OrderedDict
import time
from .worker import celeryApp
from bot.utils import BotLogger
from bot.core import (
    DataProcessor,
    FeatureExtractor,
    TextToSemantic,
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
    ifDenoise: bool
):
    try:
        processor = DataProcessor()
        processor.process(file_name)
        extractor = FeatureExtractor()
        extractor.extract(file_path=file_name, denoised=ifDenoise)
        t2s = TextToSemantic()
        t2s.process(file_name)

        mp.set_start_method('spawn', force=True)  # 强制使用 spawn 模式
        hps_sovits = get_hparams_from_file(SOVITS_MODEL_CONFIG)
        processed_path = Path(PROCESS_PATH) / file_name
        hps_sovits.data.processed_dir = processed_path
        trainer_sovits = SoVITsTrainer(hparams=hps_sovits)
        trainer.train(epochs=sovits_epochs)

        hps_gpt = get_hparams_from_file(GPT_MODEL_CONFIG)
        hps_gpt.data.semantic_path = processed_path / 'name2text.json'
        hps_gpt.data.phoneme_path = processed_path / 'text2semantic.json'
        hps_gpt.data.bert_path = processed_path / 'bert'
        trainer_gpt = GPTTrainer(hparams=hps_gpt)
        trainer_gpt.train(epochs=gpt_epochs)

        sovits_half_weights_path = Path(WEIGHTS_PATH) / file_name / SOVITS_HALF_WEIGHTS_FILE
        gpt_half_weights_path = Path(WEIGHTS_PATH) / file_name / GPT_HALF_WEIGHTS_FILE

        results = OrderedDict()
        results["SoVITs Weight"] = sovits_half_weights_path
        results["GPT Weight"] = gpt_half_weights_path
        return {
            "status": "success",
            "results": results,
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }

    except Exception as e:
        BotLogger.error(
            f"训练失败 | 文件: {file_name} | 错误跟踪:\n{traceback.format_exc()}"
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)  