# -*- coding: utf-8 -*-
import traceback
import torch
import torch.multiprocessing as mp
from pathlib import Path
from collections import OrderedDict
import time
from typing import Optional
import gc
import argparse

from .worker import celeryApp
from bot.utils import BotLogger, i18n, get_hparams_from_file

from bot.engine import TrainingPipeline, ResumingPipeline, VersionDispatcher
from bot.config import (
    WEIGHTS_PATH,
    GPT_WEIGHTS_FILE,
    SOVITS_G_WEIGHTS_FILE
)

@celeryApp.task(name="train", bind=True)
def train_task(
    self, 
    file_id: str, 
    sovits_epochs: int,
    gpt_epochs: int,
    version: Optional[str] = 'v4',
    ifDenoise: Optional[bool] = True
):
    try:
        args = argparse.Namespace()
        args.fileID = file_id
        args.version = version
        args.epochs_sovits = sovits_epochs
        args.epochs_gpt = gpt_epochs
        if ifDenoise:
            args.denoise = True
        else:
            args.denoise = False

        components = VersionDispatcher.create_components(args.version)
        pipeline = TrainingPipeline(args, components)
        modelID = pipeline.execute()
        results = OrderedDict()
        results["model id"] = modelID
        return {
            "status": "success",
            "results": results,
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }

    except Exception as e:
        BotLogger.error(
            f"{i18n('训练过程错误')}: {args.fileID} | Traceback :\n{traceback.format_exc()}"
        )
        return {
            "status": "fail",
            "results": {},
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }

@celeryApp.task(name="resume", bind=True)
def resume_task(
    self,
    model_id: str,
    sovits_epochs: int,
    gpt_epochs: int
):
    try:
        args = argparse.Namespace()
        args.modelID = model_id
        args.epochs_sovits = sovits_epochs
        args.epochs_gpt = gpt_epochs

        gpt_weights = Path(WEIGHTS_PATH) / args.modelID / GPT_WEIGHTS_FILE
        gpt_ckpt = torch.load(gpt_weights, map_location="cpu")
        version = gpt_ckpt["config"]["model"]["version"]
        
        sovits_weights = Path(WEIGHTS_PATH) / args.modelID / SOVITS_G_WEIGHTS_FILE
        sovits_ckpt = torch.load(sovits_weights, map_location="cpu")
        
        gpt_epoch = gpt_ckpt["iteration"]
        sovits_epoch = sovits_ckpt["iteration"]
        BotLogger.info(f"Model Version: {version}\n"
                       f"SOVITS trained epoch: {sovits_epoch}\n"
                       f"GPT trained epoch: {gpt_epoch}")

        components = VersionDispatcher.create_components(version)
        pipeline = ResumingPipeline(args, components)
        pipeline.execute() 

        results = {}
        return {
            "status": "success",
            "results": results,
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    
    except Exception as e:
        BotLogger.error(
            f"{i18n('训练过程错误')}: {args.modelID} | Traceback :\n{traceback.format_exc()}"
        )