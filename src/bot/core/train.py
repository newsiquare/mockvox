# -*- coding: utf-8 -*-
""" Trainer """
from pathlib import Path
from typing import Optional
import datetime
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from bot.utils import get_hparams_from_file, load_checkpoint, save_checkpoint, BotLogger
from bot.config import (
    MODEL_CONFIG_FILE, 
    PRETRAINED_S2G_FILE, 
    PRETRAINED_S2D_FILE,
    WEIGHTS_PATH,
    SOVITS_G_WEIGHTS_FILE,
    SOVITS_D_WEIGHTS_FILE
)
from bot.models import (
    TextAudioSpeakerLoader, 
    TextAudioSpeakerCollate, 
    DistributedBucketSampler,
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

class SoVITsTrainer:
    def __init__(
        self,
        processed_path,                 # 处理之后的数据存放地址
        device: Optional[str] = None    # 指定计算设备
    ):
        self.hps = get_hparams_from_file(MODEL_CONFIG_FILE)
        self.hps.data.processed_dir = processed_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = TextAudioSpeakerLoader(self.hps.data)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method='env://',
            world_size=1,
            rank=0            
        )
        self.sampler = DistributedBucketSampler(
            self.dataset, 
            self.hps.train.batch_size,
            [
                32,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1100,
                1200,
                1300,
                1400,
                1500,
                1600,
                1700,
                1800,
                1900,
            ],
            shuffle=True        
        )
        self.collate_fn = TextAudioSpeakerCollate()
        self.dataloader = DataLoader(
            self.dataset,
            num_workers=6,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_sampler=self.sampler,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4
        )
        
        # SoVITs Generator
        self.net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(self.device)

        # SoVITs Discriminator
        self.net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(self.device)

        te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
        et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
        mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
        base_params = filter(
            lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
            net_g.parameters(),
        )
        self.optim_g = torch.optim.AdamW(
            # filter(lambda p: p.requires_grad, net_g.parameters()), ###默认所有层lr一致
            [
                {"params": base_params, "lr": hps.train.learning_rate},
                {
                    "params": net_g.enc_p.text_embedding.parameters(),
                    "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
                },
                {
                    "params": net_g.enc_p.encoder_text.parameters(),
                    "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
                },
                {
                    "params": net_g.enc_p.mrte.parameters(),
                    "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
                },
            ],
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=hps.train.lr_decay, last_epoch=-1
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=hps.train.lr_decay, last_epoch=-1
        )
    
        self.scaler = GradScaler(enabled=self.hps.train.fp16_run)

        file_name = Path(self.hps.data.processed_dir).name
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/generator.pth
        self.generator_weights_path = Path(WEIGHTS_PATH) / file_name / SOVITS_G_WEIGHTS_FILE
        # 类似 ./data/weights/20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878/discriminator.pth
        self.discriminator_weights_path = Path(WEIGHTS_PATH) / file_name / SOVITS_D_WEIGHTS_FILE

    def train(self, epochs: Optional[int]=100):
        """ 执行训练 """
        epochs = epochs or self.hps.train.epochs
        # 如果训练过, 尝试加载最后一次模型参数
        epoch_done = self.resume()
        if epoch_done:
            for _ in range(epoch):
                self.scheduler_g.step()
                self.scheduler_d.step()
        else:
            epoch_done=0
            self.load_pretrained()

        saved = False
        for epoch in range(epoch_done+1, epochs+1):
            saved = False
            self._do_train(epoch)
            self.scheduler_g.step()
            self.scheduler_d.step()

            if epoch % self.hps.train.save_interval == 0:
                save_checkpoint(
                    self.net_g,
                    self.optim_g,
                    self.hps.train.learning_rate,
                    epoch,
                    self.generator_weights_path
                )
                save_checkpoint(
                    self.net_d,
                    self.optim_d,
                    self.hps.train.learning_rate,
                    epoch,
                    self.discriminator_weights_path
                )
                saved = True
        
        # save latest
        if not saved:
            save_checkpoint(
                self.net_g,
                self.optim_g,
                self.hps.train.learning_rate,
                epoch,
                self.generator_weights_path
            )
            save_checkpoint(
                self.net_d,
                self.optim_d,
                self.hps.train.learning_rate,
                epoch,
                self.discriminator_weights_path
            )
        
        BotLogger.info(
            f"模型训练完成 --- 生成器参数: {self.generator_weights_path}; 分类器参数: {self.discriminator_weights_path}, \
                时间: {datetime.now().isoformat()}"
        )

    def _do_train(epoch):
        self.dataloader.batch_sampler.set_epoch(epoch)
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
        ) in enumerate(self.dataloader):
            spec = spec.to(self.device)
            spec_lengths = spec_lengths.to(self.device)
            y, y_lengths = y.to(self.device), y_lengths.to(self.device)
            ssl = ssl.to(self.device)
            ssl.requires_grad = False
            text, text_lengths = text.to(self.device), text_lengths.to(self.device)

            with autocast(enabled=self.hps.train.fp16_run):
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
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax,
            )
            y_mel = slice_segments(
                mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax
            )
            y = slice_segments(
                y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size
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
            with autocast(enabled=self.hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

            self.optim_g.zero_grad()
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim_g)
            grad_norm_g = clip_grad_value_(self.net_g.parameters(), None)
            self.scaler.step(self.optim_g)
            self.scaler.update()

    def resume(self):
        """Check if resume checkpoint exists"""
        if self.hps.train.resume:
            if not self.generator_weights_path.exists(): return None
            if not self.discriminator_weights_path.exists(): return None
            try:                
                self.net_d, self.optim_d, _, _ = load_checkpoint(
                    self.discriminator_weights_path,
                    self.net_d,
                    self.optim_d)
                self.net_g, self.optim_g, self.hps.train.learning_rate, epoch = load_checkpoint(
                    self.generator_weights_path,
                    self.net_g,
                    self.optim_g)
            except Exception as e:
                BotLogger.error(
                    f"模型参数加载异常 | 文件: {file_name} | 错误: {str(e)}"
                )
                return None
        return epoch

    def load_pretrained(self) -> bool:
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
    from bot.config import PROCESS_PATH
    processed_path = Path(PROCESS_PATH) / "20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878"
    trainer = SoVITsTrainer(processed_path)
    trainer.train(epochs=2)