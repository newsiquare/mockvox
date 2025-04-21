# -*- coding: utf-8 -*-
""" Trainer """
from pathlib import Path
from typing import Optional
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from bot.utils import (
    get_hparams_from_file,
    load_checkpoint,
    save_checkpoint,
    BotLogger,
    CustomTQDM
)
from bot.config import (
    PRETRAINED_S2G_FILE, 
    PRETRAINED_S2D_FILE,
    PRETRAINED_GPT_FILE,
    WEIGHTS_PATH,
    SOVITS_G_WEIGHTS_FILE,
    SOVITS_D_WEIGHTS_FILE,
    GPT_WEIGHTS_FILE
)
from bot.models import (
    TextAudioSpeakerLoader, 
    TextAudioSpeakerCollate, 
    SoVITsBucketSampler,
    Text2SemanticDataset,
    GPTBucketSampler,
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    spec_to_mel_torch,
    mel_spectrogram_torch,
    slice_segments,
    discriminator_loss,
    generator_loss,
    feature_loss,
    kl_loss,
    clip_grad_value_
)
from bot.models.AR import Text2SemanticDecoder
from bot.models.AR.nn import ScaledAdam, WarmupCosineLRSchedule

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
            num_workers=self.hparams.data.num_workers,
            shuffle=False,
            collate_fn=dataset.collate,
            batch_sampler=sampler,
            persistent_workers=True,
            prefetch_factor=8
        )

        self.model = Text2SemanticDecoder(self.hparams).to(self.device)
        self.scaler = GradScaler(enabled=(self.hparams.train.precision=='16-mixed'))
        self.optimizer, self.scheduler = self._configure_optimizers()

        self.file_name = Path(self.hparams.data.semantic_path).parent.name
        (Path(WEIGHTS_PATH) / self.file_name).mkdir(parents=True, exist_ok=True)
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/gpt.pth
        self.gpt_weights_path = Path(WEIGHTS_PATH) / self.file_name / GPT_WEIGHTS_FILE

    def train(self, epochs: Optional[int]=100):
        """ 执行训练 """
        epochs = epochs or self.hparams.train.epochs
        # 如果训练过, 尝试加载最后一次模型参数
        epoch_done = self._resume()
        if not epoch_done:
            epoch_done = 0
            self._load_pretrained()

        saved = False
        BotLogger.info(f"启动GPT训练 |  路径: {self.file_name} | 时间: {datetime.now().isoformat()}")

        for epoch in range(epoch_done+1, epochs+1):
            saved=False
            BotLogger.info(f"训练轮次: {epoch}")
            self._do_train(epoch)
            # self.scheduler.step()

            if epoch % self.hparams.train.save_interval == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    None,
                    epoch,
                    self.gpt_weights_path
                )
            saved = True
        
        if not saved:
            save_checkpoint(
                self.model,
                self.optimizer,
                None,
                epochs,
                self.gpt_weights_path
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
        if not Path(PRETRAINED_GPT_FILE).exists(): return False
        try:
            self.model.load_state_dict(
                torch.load(PRETRAINED_GPT_FILE, map_location='cpu')["weight"],
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

        dataset = TextAudioSpeakerLoader(self.hparams.data)
        sampler = SoVITsBucketSampler(
            dataset, 
            batch_size=self.hparams.train.batch_size,
            boundaries=[
                32, 300, 400, 500, 600, 700, 800, 900, 
                1000, 1100, 1200, 1300, 1400, 1500, 
                1600, 1700, 1800, 1900
            ],
            shuffle=True
        )
        collate_fn = TextAudioSpeakerCollate(self.device)
        self.dataloader = DataLoader(
            dataset,
            num_workers=4,
            shuffle=False,
            # pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=sampler,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # SoVITs Generator
        self.net_g = SynthesizerTrn(
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            n_speakers = self.hparams.data.n_speakers,
            **self.hparams.model,
        ).to(self.device)

        # SoVITs Discriminator
        self.net_d = MultiPeriodDiscriminator(self.hparams.model.use_spectral_norm).to(self.device)

        te_p = list(map(id, self.net_g.enc_p.text_embedding.parameters()))
        et_p = list(map(id, self.net_g.enc_p.encoder_text.parameters()))
        mrte_p = list(map(id, self.net_g.enc_p.mrte.parameters()))
        base_params = filter(
            lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
            self.net_g.parameters(),
        )
        self.optim_g = torch.optim.AdamW(
            # filter(lambda p: p.requires_grad, net_g.parameters()), ###默认所有层lr一致
            [
                {"params": base_params, "lr": self.hparams.train.learning_rate},
                {
                    "params": self.net_g.enc_p.text_embedding.parameters(),
                    "lr": self.hparams.train.learning_rate * self.hparams.train.text_low_lr_rate,
                },
                {
                    "params": self.net_g.enc_p.encoder_text.parameters(),
                    "lr": self.hparams.train.learning_rate * self.hparams.train.text_low_lr_rate,
                },
                {
                    "params": self.net_g.enc_p.mrte.parameters(),
                    "lr": self.hparams.train.learning_rate * self.hparams.train.text_low_lr_rate,
                },
            ],
            self.hparams.train.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps,
        )

        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.hparams.train.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps,
        )

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=self.hparams.train.lr_decay, last_epoch=-1
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=self.hparams.train.lr_decay, last_epoch=-1
        )
    
        self.scaler = GradScaler(enabled=self.hparams.train.fp16_run)

        self.file_name = Path(self.hparams.data.processed_dir).name
        (Path(WEIGHTS_PATH) / self.file_name).mkdir(parents=True, exist_ok=True)
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/gen.pth
        self.generator_weights_path = Path(WEIGHTS_PATH) / self.file_name / SOVITS_G_WEIGHTS_FILE
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/disc.pth
        self.discriminator_weights_path = Path(WEIGHTS_PATH) / self.file_name / SOVITS_D_WEIGHTS_FILE

    def train(self, epochs: Optional[int]=100):
        """ 执行训练 """
        epochs = epochs or self.hparams.train.epochs
        # 如果训练过, 尝试加载最后一次模型参数
        epoch_done = self._resume()
        if epoch_done:
            for _ in range(epoch_done):
                self.scheduler_g.step()
                self.scheduler_d.step()
        else:
            epoch_done=0
            self._load_pretrained()

        saved = False
        BotLogger.info(f"启动SoVITs训练 |  路径: {self.file_name} | 时间: {datetime.now().isoformat()}")
        for epoch in range(epoch_done+1, epochs+1):
            saved = False
            BotLogger.info(f"训练轮次: {epoch}")
            self._do_train(epoch)
            self.scheduler_g.step()
            self.scheduler_d.step()

            if epoch % self.hparams.train.save_interval == 0:
                save_checkpoint(
                    self.net_g,
                    self.optim_g,
                    self.hparams.train.learning_rate,
                    epoch,
                    self.generator_weights_path
                )
                save_checkpoint(
                    self.net_d,
                    self.optim_d,
                    self.hparams.train.learning_rate,
                    epoch,
                    self.discriminator_weights_path
                )
                saved = True
        
        # save latest
        if not saved:
            save_checkpoint(
                self.net_g,
                self.optim_g,
                self.hparams.train.learning_rate,
                epochs,
                self.generator_weights_path
            )
            save_checkpoint(
                self.net_d,
                self.optim_d,
                self.hparams.train.learning_rate,
                epochs,
                self.discriminator_weights_path
            )
        
        BotLogger.info(
            f"模型训练完成 | 生成器参数: {self.generator_weights_path} | 分类器参数: {self.discriminator_weights_path} | \
                时间: {datetime.now().isoformat()}"
        )

    def _do_train(self, epoch):
        # self.dataloader.batch_sampler.set_epoch(epoch)
        self.net_g.train()
        self.net_d.train()

        for batch_idx, (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            text,
            text_lengths       
        ) in CustomTQDM(enumerate(self.dataloader)):
            spec = spec.to(self.device)
            spec_lengths = spec_lengths.to(self.device)

            y, y_lengths = y.to(self.device), y_lengths.to(self.device)
            ssl = ssl.to(self.device)
            ssl.requires_grad = False
            text, text_lengths = text.to(self.device), text_lengths.to(self.device)

            with autocast(enabled=self.hparams.train.fp16_run):
                (
                    y_hat,
                    kl_ssl,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    stats_ssl,
                ) = self.net_g(ssl, spec, spec_lengths, text, text_lengths)
            
                mel = spec_to_mel_torch(
                    spec,
                    self.hparams.data.filter_length,
                    self.hparams.data.n_mel_channels,
                    self.hparams.data.sampling_rate,
                    self.hparams.data.mel_fmin,
                    self.hparams.data.mel_fmax,
                )
                y_mel = slice_segments(
                    mel, ids_slice, self.hparams.train.segment_size // self.hparams.data.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    self.hparams.data.filter_length,
                    self.hparams.data.n_mel_channels,
                    self.hparams.data.sampling_rate,
                    self.hparams.data.hop_length,
                    self.hparams.data.win_length,
                    self.hparams.data.mel_fmin,
                    self.hparams.data.mel_fmax
                )
                y = slice_segments(
                    y, ids_slice * self.hparams.data.hop_length, self.hparams.train.segment_size
                ) 

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc

            self.optim_d.zero_grad()
            self.scaler.scale(loss_disc_all).backward()
            self.scaler.unscale_(self.optim_d)
            # grad_norm_d = clip_grad_value_(self.net_d.parameters(), None)
            self.scaler.step(self.optim_d)

            # Generator
            with autocast(enabled=self.hparams.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

            self.optim_g.zero_grad()
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim_g)
            grad_norm_g = clip_grad_value_(self.net_g.parameters(), None)
            self.scaler.step(self.optim_g)
            self.scaler.update()

    def _resume(self):
        """Check if resume checkpoint exists"""
        if self.hparams.train.resume:
            if not self.generator_weights_path.exists(): return None
            if not self.discriminator_weights_path.exists(): return None
            try:                
                self.net_d, self.optim_d, _, _ = load_checkpoint(
                    self.discriminator_weights_path,
                    self.net_d,
                    self.optim_d)
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
        if not Path(PRETRAINED_S2G_FILE).exists(): return False
        if not Path(PRETRAINED_S2D_FILE).exists(): return False
        try:
            if hasattr(self.net_g, "module"):
                self.net_g.module.load_state_dict(
                    torch.load(PRETRAINED_S2G_FILE, map_location="cpu")["weight"],
                    strict=False
                )
            else:
                self.net_g.load_state_dict(
                    torch.load(PRETRAINED_S2G_FILE, map_location="cpu")["weight"],
                    strict=False
                )

            if hasattr(self.net_g, "module"):
                self.net_g.module.load_state_dict(
                    torch.load(PRETRAINED_S2G_FILE, map_location="cpu")["weight"],
                    strict=False
                )
            else:
                self.net_g.load_state_dict(
                    torch.load(PRETRAINED_S2G_FILE, map_location="cpu")["weight"],
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

    from bot.config import PROCESS_PATH, SOVITS_MODEL_CONFIG

    hparams = get_hparams_from_file(SOVITS_MODEL_CONFIG)
    processed_path = Path(PROCESS_PATH) / args.file
    hparams.data.processed_dir = processed_path
    trainer = SoVITsTrainer(hparams=hparams)
    trainer.train(epochs=10)

    from bot.config import GPT_MODEL_CONFIG
    hparams = get_hparams_from_file(GPT_MODEL_CONFIG)
    hparams.data.semantic_path = processed_path / 'name2text.json'
    hparams.data.phoneme_path = processed_path / 'text2semantic.json'
    hparams.data.bert_path = processed_path / 'bert'
    trainer = GPTTrainer(hparams=hparams)
    trainer.train(epochs=10)