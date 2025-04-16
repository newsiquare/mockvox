from pathlib import Path
from typing import Optional

from bot.utils import get_hparams_from_file
from bot.config import MODEL_CONFIG_FILE
from bot.models import TextAudioSpeakerLoader, DistributedBucketSampler

class Trainer:
    def __init__(
        self,
        processed_path,
        device: Optional[str] = None  # 指定计算设备
    ):
        self.hps = get_hparams_from_file(MODEL_CONFIG_FILE)
        self.hps.data.processed_dir = processed_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = TextAudioSpeakerLoader(hps.data)

