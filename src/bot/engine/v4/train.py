# -*- coding: utf-8 -*-
""" Trainer """
from pathlib import Path
from typing import Optional
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from bot.utils import (
    get_hparams_from_file,
    load_checkpoint,
    save_checkpoint,
    save_checkpoint_half_latest,
    BotLogger,
    CustomTQDM
)
from bot.config import (
    PRETRAINED_S2GV4_FILE, 
    PRETRAINED_T2SV4_FILE,
    WEIGHTS_PATH,
    SOVITS_G_WEIGHTS_FILE,
    SOVITS_D_WEIGHTS_FILE,
    SOVITS_HALF_WEIGHTS_FILE,
    GPT_WEIGHTS_FILE,
    GPT_HALF_WEIGHTS_FILE
)
from bot.nn.AR import (
    ScaledAdam,
    WarmupCosineLRSchedule
)
from bot.models.v2 import (
    Text2SemanticDecoder
)

from bot.models.v4 import (
    TextAudioSpeakerDataset, 
    TextAudioSpeakerCollate, 
    SoVITsBucketSampler,
    SynthesizerTrnV3,
    Text2SemanticDataset,
    GPTBucketSampler
)

class GPTTrainer:
    """
    传入超参数(hparams)时，必须具备以下参数:
        hparams.data.semantic_path - name2text.json 文件的完整路径
        hparams.data.phoneme_path - text2semantic.json 文件的完整路径 
        hparams.data.bert_path - bert 子目录  
    """
    def __init__(
        self,
        hparams,
        device: Optional[str] = None
    ):
        self.hparams = hparams
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        dataset = Text2SemanticDataset(self.hparams.data)
        sampler = GPTBucketSampler(
            dataset,
            batch_size=self.hparams.train.batch_size,
            shuffle=True,
            bucket_width=2.0
        )
        self.dataloader = DataLoader(
            dataset,
            num_workers=0,
            shuffle=False,
            collate_fn=dataset.collate,
            batch_sampler=sampler,
            persistent_workers=False,
            prefetch_factor=None
        )

        self.model = Text2SemanticDecoder(self.hparams).to(self.device)
        self.scaler = GradScaler(enabled=(self.hparams.train.precision=='16-mixed'))
        self.optimizer, self.scheduler = self._configure_optimizers()

        self.file_name = Path(self.hparams.data.semantic_path).parent.name
        (Path(WEIGHTS_PATH) / self.file_name).mkdir(parents=True, exist_ok=True)
        self.gpt_weights_path = Path(WEIGHTS_PATH) / self.file_name / GPT_WEIGHTS_FILE
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/gpt.pth
        self.gpt_half_weights_path = Path(WEIGHTS_PATH) / self.file_name / GPT_HALF_WEIGHTS_FILE

    def train(self, epochs: Optional[int]=100):
        """ 执行训练 """
        epochs = epochs or self.hparams.train.epochs
        # 如果训练过, 尝试加载最后一次模型参数
        epoch_done = self._resume()
        if not epoch_done:
            epoch_done = 0
            self._load_pretrained()
        elif epochs<=epoch_done:
            BotLogger.info(f"GPT已训练轮次 {epoch_done} >= {epochs}, 训练终止.")
            return

        saved = False
        BotLogger.info(f"启动GPT训练 |  路径: {self.file_name} | 时间: {datetime.now().isoformat()}")

        for epoch in range(epoch_done+1, epochs+1):
            saved=False
            BotLogger.info(f"GPT训练轮次: {epoch}")
            self._do_train(epoch)
            # self.scheduler.step()

            if epoch % self.hparams.train.save_interval == 0:
                save_checkpoint(
                    self.model,
                    self.hparams,
                    self.optimizer,
                    None,
                    epoch,
                    self.gpt_weights_path
                )
                saved = True
        
        if not saved:
            save_checkpoint(
                self.model,
                self.hparams,
                self.optimizer,
                None,
                epochs,
                self.gpt_weights_path
            )
        save_checkpoint_half_latest(self.model, self.hparams, epochs, self.gpt_half_weights_path)        

        BotLogger.info(
            f"GPT模型训练完成. \n \
            GPT参数: {self.gpt_weights_path} \n \
            半精度GPT推理参数: {self.gpt_half_weights_path} \n \
            时间: {datetime.now().isoformat()}"
        )

    def _do_train(self, epoch):
        self.model.train()
        for batch_idx, batch in CustomTQDM(enumerate(self.dataloader)):
            with autocast(enabled=(self.hparams.train.precision=='16-mixed')):
                loss, acc = self.model.forward(
                    batch["phoneme_ids"].to(self.device),
                    batch["phoneme_ids_len"].to(self.device),
                    batch["semantic_ids"].to(self.device),
                    batch["semantic_ids_len"].to(self.device),
                    batch["bert_feature"].to(self.device)
                )
            
            self.scaler.scale(loss).backward()
            if batch_idx > 0 and batch_idx % 4 == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
    
    def _load_pretrained(self) -> bool:
        """ 加载预训练模型 """
        if not Path(PRETRAINED_T2SV4_FILE).exists(): return False
        try:
            ckpt = torch.load(
                PRETRAINED_T2SV4_FILE, 
                map_location="cpu",
                weights_only=False
            )
            ckpt = {k.replace("model.", ""): v for k,v in ckpt["weight"].items()}
            self.model.load_state_dict(
                ckpt,
                strict=False
            )
        except Exception as e:
            BotLogger.error(
                f"预训练模型加载异常 | 错误: {str(e)}"
            )
            raise RuntimeError("预训练模型加载失败")

        return True

    def _resume(self):
        """Check if resume checkpoint exists"""
        if self.hparams.train.resume:
            if not self.gpt_weights_path.exists(): return None
            try:                
                self.model, self.optimizer, _, epoch = load_checkpoint(
                    self.gpt_weights_path,
                    self.model,
                    self.optimizer)
                for _ in range(epoch):
                    self.scheduler.step()
            except Exception as e:
                BotLogger.error(
                    f"模型参数加载异常 | 文件: {self.file_name} | 错误: {str(e)}"
                )
                return None
        return epoch
    
    def _configure_optimizers(self):
        # 获取模型参数名称列表
        parameters_names = [
            [name for name, _ in self.model.named_parameters()]
        ]

        # 创建ScaledAdam优化器
        optimizer = ScaledAdam(
            params=self.model.parameters(),
            lr=0.01,  # 初始占位值，实际由scheduler控制
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            clipping_update_period=1000,
            show_dominant_parameters=False
        )

        # 创建学习率调度器
        scheduler = WarmupCosineLRSchedule(
            optimizer=optimizer,
            init_lr=self.hparams.optimizer.lr_init,
            peak_lr=self.hparams.optimizer.lr,
            end_lr=self.hparams.optimizer.lr_end,
            warmup_steps=self.hparams.optimizer.warmup_steps,
            total_steps=self.hparams.optimizer.decay_steps
        )

        return optimizer, scheduler

class SoVITsTrainer:
    """
    传入超参数(hparams)时，必须具备以下参数:
        hparams.data.processed_dir - 存放数据处理结果的完整路径 
    """
    def __init__(
        self,
        hparams,                            # 配置信息
        device: Optional[str] = None    # 指定计算设备
    ):
        self.hparams = hparams
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        dataset = TextAudioSpeakerDataset(self.hparams.data)
        sampler = SoVITsBucketSampler(
            dataset, 
            batch_size=self.hparams.train.batch_size,
            boundaries=[
                32, 300, 400, 500, 600, 700, 800, 900, 
                1000
            ],
            shuffle=True
        )
        collate_fn = TextAudioSpeakerCollate(self.device)
        self.dataloader = DataLoader(
            dataset,
            num_workers=0,
            shuffle=False,
            # pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=sampler,
            persistent_workers=False,
            prefetch_factor=None
        )
        
        # SoVITs Generator
        self.net_g = SynthesizerTrnV3(
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            n_speakers = self.hparams.data.n_speakers,
            **self.hparams.model,
        ).to(self.device)

        lora_rank = int(self.hparams.train.lora_rank)
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )        
        self.net_g.cfm = get_peft_model(self.net_g.cfm, lora_config)

        self.optim_g = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.net_g.parameters()), ###默认所有层lr一致
            self.hparams.train.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps,
        )

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=self.hparams.train.lr_decay, last_epoch=-1
        )
        self.scaler = GradScaler(enabled=self.hparams.train.fp16_run)

        self.file_name = Path(self.hparams.data.processed_dir).name
        (Path(WEIGHTS_PATH) / self.file_name).mkdir(parents=True, exist_ok=True)
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/gen.pth
        self.generator_weights_path = Path(WEIGHTS_PATH) / self.file_name / SOVITS_G_WEIGHTS_FILE
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/sovits.pth
        self.sovits_weights_path = Path(WEIGHTS_PATH) / self.file_name / SOVITS_HALF_WEIGHTS_FILE

    def train(self, epochs: Optional[int]=100):
        """ 执行训练 """
        epochs = epochs or self.hparams.train.epochs
        # 如果训练过, 尝试加载最后一次模型参数
        epoch_done = self._resume()
        if epoch_done:
            for _ in range(epoch_done):
                self.scheduler_g.step()
        else:
            epoch_done=0
            self._load_pretrained()

        if epochs<=epoch_done:
            BotLogger.info(f"SoVITS已训练轮次 {epoch_done} >= {epochs}, 训练终止.")
            return

        saved = False
        BotLogger.info(f"启动SoVITS训练 |  路径: {self.file_name} | 时间: {datetime.now().isoformat()}")
        for epoch in range(epoch_done+1, epochs+1):
            saved = False
            BotLogger.info(f"SoVITS训练轮次: {epoch}")
            self._do_train(epoch)
            self.scheduler_g.step()

            if epoch % self.hparams.train.save_interval == 0:
                save_checkpoint(
                    self.net_g,
                    self.hparams,
                    self.optim_g,
                    self.hparams.train.learning_rate,
                    epoch,
                    self.generator_weights_path
                )
                saved = True
        
        # save latest
        if not saved:
            save_checkpoint(
                self.net_g,
                self.hparams,
                self.optim_g,
                self.hparams.train.learning_rate,
                epochs,
                self.generator_weights_path
            )
        
        save_checkpoint_half_latest(self.net_g, self.hparams, epochs, self.sovits_weights_path)        

        BotLogger.info(
            f"模型训练完成 \n \
            生成器参数: {self.generator_weights_path} \n \
            半精度SoVITs推理参数: {self.sovits_weights_path} \n \
            时间: {datetime.now().isoformat()}"
        )

    def _do_train(self, epoch):
        # self.dataloader.batch_sampler.set_epoch(epoch)
        self.net_g.train()

        for batch_idx, (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            mel,
            mel_lengths,
            text,
            text_lengths       
        ) in CustomTQDM(enumerate(self.dataloader)):
            ssl = ssl.to(self.device)
            ssl.requires_grad = False
            spec, spec_lengths = spec.to(self.device), spec_lengths.to(self.device)
            mel, mel_lengths = mel.to(self.device), mel_lengths.to(self.device)
            text, text_lengths = text.to(self.device), text_lengths.to(self.device)

            with autocast(enabled=self.hparams.train.fp16_run):
                cfm_loss = self.net_g(
                    ssl,
                    spec,
                    mel,
                    ssl_lengths,
                    spec_lengths,
                    text,
                    text_lengths,
                    mel_lengths,
                    use_grad_ckpt=self.hparams.train.grad_ckpt,
                )
                loss_gen_all = cfm_loss            
                
            self.optim_g.zero_grad()
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim_g)
            # grad_norm_g = clip_grad_value_(self.net_g.parameters(), None)
            self.scaler.step(self.optim_g)
            self.scaler.update()

    def _resume(self):
        """Check if resume checkpoint exists"""
        if self.hparams.train.resume:
            if not self.generator_weights_path.exists(): return None
            try:                
                self.net_g, self.optim_g, self.hparams.train.learning_rate, epoch = load_checkpoint(
                    self.generator_weights_path,
                    self.net_g,
                    self.optim_g)
            except Exception as e:
                BotLogger.error(
                    f"模型参数加载异常 | 文件: {self.file_name} | 错误: {str(e)}"
                )
                return None
        return epoch

    def _load_pretrained(self) -> bool:
        """ 加载预训练模型 """
        if not Path(PRETRAINED_S2GV4_FILE).exists(): return False
        try:
            if hasattr(self.net_g, "module"):
                self.net_g.module.load_state_dict(
                    torch.load(PRETRAINED_S2GV4_FILE, map_location="cpu")["weight"],
                    strict=False
                )
            else:
                self.net_g.load_state_dict(
                    torch.load(PRETRAINED_S2GV4_FILE, map_location="cpu")["weight"],
                    strict=False
                )

        except Exception as e:
            BotLogger.error(
                f"预训练模型加载异常 | 错误: {str(e)}"
            )
            raise RuntimeError("预训练模型加载失败")

        return True

if __name__ == '__main__':
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='processed file name.')
    args = parser.parse_args()
    
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # 强制使用 spawn 模式

    device="cuda" if torch.cuda.is_available() else "cpu"
    from bot.config import PROCESS_PATH, SOVITS_MODEL_CONFIG

    hparams = get_hparams_from_file(SOVITS_MODEL_CONFIG)
    processed_path = Path(PROCESS_PATH) / args.file
    hparams.data.processed_dir = processed_path
    trainer = SoVITsTrainer(hparams=hparams, device=device)
    trainer.train(epochs=1)

    from bot.config import GPT_MODEL_CONFIG
    hparams = get_hparams_from_file(GPT_MODEL_CONFIG)
    hparams.data.semantic_path = processed_path / 'name2text.json'
    hparams.data.phoneme_path = processed_path / 'text2semantic.json'
    hparams.data.bert_path = processed_path / 'bert'
    trainer = GPTTrainer(hparams=hparams,device=device)
    trainer.train(epochs=1)