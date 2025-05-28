# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import os
import gc
import torch
import traceback
import time

from mockvox.utils import MockVoxLogger, allowed_file, generate_unique_filename, i18n
from mockvox.config import get_config, SLICED_ROOT_PATH, DENOISED_ROOT_PATH, ASR_PATH
from mockvox.engine.v2 import slice_audio, batch_denoise
from mockvox.engine.v4.inference import Inferencer
from mockvox.engine.v2 import batch_asr
from mockvox.engine import TrainingPipeline, ResumingPipeline, VersionDispatcher
         
from mockvox.config import (
    WEIGHTS_PATH,
    SOVITS_HALF_WEIGHTS_FILE,
    GPT_HALF_WEIGHTS_FILE,
    SOVITS_G_WEIGHTS_FILE,
    GPT_WEIGHTS_FILE,
    OUT_PUT_PATH,
    ASR_PATH
)
import soundfile as sf

CLI_HELP_MSG = f"""
    MockVox command line support the following subcommands:        
"""
cfg = get_config()

def handle_upload(args):
    if not allowed_file(args.file):
        MockVoxLogger.error(i18n("不支持的文件格式"))
    if os.path.getsize(args.file) > cfg.MAX_UPLOAD_SIZE:
        MockVoxLogger.error(i18n("文件大小超过限制"))

    fileID = generate_unique_filename(Path(args.file).name)
    stem = Path(fileID).stem

    try:
        # 文件切割
        sliced_path = os.path.join(SLICED_ROOT_PATH, stem)
        sliced_files = slice_audio(args.file, sliced_path)
        MockVoxLogger.info(f"File sliced: {sliced_path}")

        # 降噪
        if(args.denoise):
            denoised_path = os.path.join(DENOISED_ROOT_PATH, stem)
            denoised_files = batch_denoise(sliced_files, denoised_path)
        MockVoxLogger.info(f"File denoised: {denoised_path}")

        # 语音识别
        asr_path = os.path.join(ASR_PATH, stem)
        if(args.denoise):
            batch_asr(args.language, denoised_files, asr_path)
        else:
            batch_asr(args.language, sliced_files, asr_path)

        MockVoxLogger.info(f"{i18n('ASR完成. 结果已保存在')}: {os.path.join(asr_path, 'output.json')}")
        MockVoxLogger.info(f"File ID: {stem}")

    except Exception as e:
        MockVoxLogger.error(
            f"{i18n('文件处理错误')}: {args.file} | Traceback:\n{traceback.format_exc()}"
        )

def handle_train(args):
    try:
        components = VersionDispatcher.create_components(args.version)
        pipeline = TrainingPipeline(args, components)
        pipeline.execute() 
    except Exception as e:
        MockVoxLogger.error(
            f"{i18n('训练过程错误')}: {args.fileID} | Traceback :\n{traceback.format_exc()}"
        )

def handle_inference(args):
    try:
        gpt_path = Path(WEIGHTS_PATH) / args.modelID / GPT_HALF_WEIGHTS_FILE
        sovits_path = Path(WEIGHTS_PATH) / args.modelID / SOVITS_HALF_WEIGHTS_FILE
        reasoning_result_path = Path(OUT_PUT_PATH)
        if not gpt_path.exists():
            MockVoxLogger.error(i18n("路径错误! 找不到GPT模型"))
            return
        if not sovits_path.exists():
            MockVoxLogger.error(i18n("路径错误! 找不到SOVITS模型"))
            return
        if not reasoning_result_path.exists():
            reasoning_result_path.mkdir(parents=True, exist_ok=True)

        gpt_ckpt = torch.load(gpt_path, map_location="cpu")
        version = gpt_ckpt["config"]["model"]["version"]
        MockVoxLogger.info(f"Model Version: {version}")
        
        inference = Inferencer(gpt_path, sovits_path,version) 
        # Synthesize audio
        synthesis_result = inference.inference(ref_wav_path=args.refWavFilePath,# 参考音频 
                                    prompt_text=args.promptText, # 参考文本
                                    prompt_language=args.promptLanguage, 
                                    text=args.targetText, # 目标文本
                                    text_language=args.targetLanguage, top_p=float(args.top_p), temperature=float(args.temperature), top_k=int(args.top_k), speed=float(args.speed))
        timestamp = str(int(time.time()))
        outputname = reasoning_result_path / Path(timestamp+".WAV")
        if synthesis_result is None:
            return
        else:
            result_list = list(synthesis_result)
            if result_list:
                last_sampling_rate, last_audio_data = result_list[-1]
                sf.write(outputname, last_audio_data, int(last_sampling_rate))
                MockVoxLogger.info(f"Audio saved in {outputname}")
    except Exception as e:
        MockVoxLogger.error(
            f"{i18n('推理过程错误')}: {args.modelID} | Traceback :\n{traceback.format_exc()}"
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()

def handle_resume(args):
    try:
        gpt_weights = Path(WEIGHTS_PATH) / args.modelID / GPT_WEIGHTS_FILE
        gpt_ckpt = torch.load(gpt_weights, map_location="cpu")
        version = gpt_ckpt["config"]["model"]["version"]
        
        sovits_weights = Path(WEIGHTS_PATH) / args.modelID / SOVITS_G_WEIGHTS_FILE
        sovits_ckpt = torch.load(sovits_weights, map_location="cpu")
        
        gpt_epoch = gpt_ckpt["iteration"]
        sovits_epoch = sovits_ckpt["iteration"]
        MockVoxLogger.info(f"Model Version: {version}\n"
                       f"SOVITS trained epoch: {sovits_epoch}\n"
                       f"GPT trained epoch: {gpt_epoch}")

        components = VersionDispatcher.create_components(version)
        pipeline = ResumingPipeline(args, components)
        pipeline.execute() 
    except Exception as e:
        MockVoxLogger.error(
            f"{i18n('训练过程错误')}: {args.modelID} | Traceback :\n{traceback.format_exc()}"
        )

def handle_info(args):
    gpt_weights = Path(WEIGHTS_PATH) / args.modelID / GPT_WEIGHTS_FILE
    sovits_weights = Path(WEIGHTS_PATH) / args.modelID / SOVITS_G_WEIGHTS_FILE
    if not gpt_weights.exists() or not sovits_weights.exists():
        MockVoxLogger.error(f"Model checkpoint not found.")
        return

    gpt_ckpt = torch.load(gpt_weights, map_location="cpu")
    version = gpt_ckpt["config"]["model"]["version"]
    
    sovits_ckpt = torch.load(sovits_weights, map_location="cpu")
    
    gpt_epoch = gpt_ckpt["iteration"]
    sovits_epoch = sovits_ckpt["iteration"]
    MockVoxLogger.info(f"Model Version: {version}\n"
                    f"SOVITS trained epoch: {sovits_epoch}\n"
                    f"GPT trained epoch: {gpt_epoch}")

def main():
    parser = argparse.ArgumentParser(prog='mockvoi', description=CLI_HELP_MSG)
    subparsers = parser.add_subparsers(dest='command', help='')

    # upload 子命令
    parser_upload = subparsers.add_parser('upload', help='Upload specified file.')
    parser_upload.add_argument('file', type=str, help='Full file path to upload.')
    parser_upload.add_argument('--no-denoise', dest='denoise', action='store_false', 
                               help='Disable denoise processing (default: enable denoise).')
    parser_upload.set_defaults(denoise=True)
    parser_upload.add_argument('--language', type=str, default='zh', help='Language code, support zh can en ja ko.')
    parser_upload.set_defaults(func=handle_upload)

    # inference 子命令
    parser_inference = subparsers.add_parser('inference', help='Inference command line')
    parser_inference.add_argument('modelID', type=str, help='Returned train id from train.')
    parser_inference.add_argument('refWavFilePath', type=str, help='Reference file full path.')
    parser_inference.add_argument('promptText', type=str, help='Prompt text.')
    parser_inference.add_argument('targetText', type=str, help='Target text.')
    parser_inference.add_argument('--no-denoise', dest='denoise', action='store_false', 
                               help='Disable denoise processing (default: enable denoise).')
    parser_inference.set_defaults(denoise=True)
    parser_inference.add_argument('--promptLanguage',default='zh', type=str, help='Prompt language.')
    parser_inference.add_argument('--targetLanguage', default='zh',type=str, help='Target Language.')
    parser_inference.add_argument('--top_p', default=1,type=str, help='top_p')
    parser_inference.add_argument('--top_k', default=15,type=str, help="GPT sampling parameters (do not set too low when there is no reference text.) If you don't understand, use the default.")
    parser_inference.add_argument('--temperature', default=1,type=str, help='GPT temperature')
    parser_inference.add_argument('--speed', default=1,type=str, help='speed')
    parser_inference.set_defaults(func=handle_inference)

    # train 子命令
    parser_train = subparsers.add_parser('train', help='Train specified file id.')
    parser_train.add_argument('fileID', type=str, help='Returned file id from upload.')
    parser_train.add_argument('--no-denoise', dest='denoise', action='store_false', 
                               help='Disable denoise processing (default: enable denoise).')
    parser_train.set_defaults(denoise=True)
    parser_train.add_argument('--epochs_sovits', type=int, default=1, help='Train epochs of SoVITS (default:1).')
    parser_train.add_argument('--epochs_gpt', type=int, default=1, help='Train epochs of GPT (default:1).')
    parser_train.add_argument('--version', type=str, default='v4', help='Default version is v4.')
    parser_train.set_defaults(func=handle_train)

    # resume 子命令
    parser_resume = subparsers.add_parser('resume', help='Resume train specified model id.')
    parser_resume.add_argument('modelID', type=str, help='Returned model id from train.')
    parser_resume.add_argument('--epochs_sovits', type=int, default=2, help='Train epochs of SoVITS (default:2).')
    parser_resume.add_argument('--epochs_gpt', type=int, default=2, help='Train epochs of GPT (default:2).')
    parser_resume.set_defaults(func=handle_resume)

    # info 子命令
    parser_info = subparsers.add_parser('info', help='Get specified model info.')
    parser_info.add_argument('modelID', type=str, help='Returned model id from train.')
    parser_info.set_defaults(func=handle_info)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
