# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import os
import gc
import torch
import torch.multiprocessing as mp
import traceback

from bot.utils import BotLogger, allowed_file, generate_unique_filename
from bot.config import get_config, SLICED_ROOT_PATH, DENOISED_ROOT_PATH, ASR_PATH
from bot.engine.v2 import slice_audio, batch_denoise, batch_asr
from bot.engine.v4.inference import Inferencer as v4
from bot.engine.v2.inference import Inferencer as v2
from bot.engine.v2 import (
    DataProcessor,
    FeatureExtractor 
)
from bot.engine.v4 import TextToSemantic
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
    GPT_HALF_WEIGHTS_FILE,
    REASONING_RESULT_PATH,
    REASONING_RESULT_FILE
)
import soundfile as sf
from bot.utils import get_hparams_from_file

CLI_HELP_MSG = f"""
    MockVoi command line support the following subcommands:        
"""
cfg = get_config()

def handle_upload(args):
    if not allowed_file(args.file):
        BotLogger.error("Unsupported file format")
    if os.path.getsize(args.file) > cfg.MAX_UPLOAD_SIZE:
        BotLogger.error("File size exceeds the limit")

    fileID = generate_unique_filename(Path(args.file).name)
    stem = Path(fileID).stem

    try:
        # 文件切割
        sliced_path = os.path.join(SLICED_ROOT_PATH, stem)
        sliced_files = slice_audio(args.file, sliced_path)
        BotLogger.info(f"File sliced: {sliced_path}")

        # 降噪
        if(args.denoise):
            denoised_path = os.path.join(DENOISED_ROOT_PATH, stem)
            denoised_files = batch_denoise(sliced_files, denoised_path)
        BotLogger.info(f"File denoised: {denoised_path}")

        # 语音识别
        asr_path = os.path.join(ASR_PATH, stem)
        if(args.denoise):
            batch_asr(denoised_files, asr_path)
        else:
            batch_asr(sliced_files, asr_path)

        BotLogger.info(f"ASR done. Results saved in: {os.path.join(asr_path, 'output.json')}")

    except Exception as e:
        BotLogger.error(
            f"Failed | File: {args.file} | Traceback:\n{traceback.format_exc()}"
        )

def handle_train(args):
    try:
        processor = DataProcessor()
        processor.process(args.fileID)
        extractor = FeatureExtractor()
        extractor.extract(file_path=args.fileID, denoised=args.denoise)
        t2s = TextToSemantic()
        t2s.process(args.fileID)
        del processor, extractor, t2s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()       

        mp.set_start_method('spawn', force=True)  # 强制使用 spawn 模式
        hps_sovits = get_hparams_from_file(SOVITS_MODEL_CONFIG)
        processed_path = Path(PROCESS_PATH) / args.fileID
        hps_sovits.data.processed_dir = processed_path
        trainer_sovits = SoVITsTrainer(hparams=hps_sovits)
        trainer_sovits.train(epochs=args.epochs_sovits)

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
        trainer_gpt.train(epochs=args.epochs_gpt)
        del trainer_gpt, hps_gpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()

        sovits_half_weights_path = Path(WEIGHTS_PATH) / args.fileID / SOVITS_HALF_WEIGHTS_FILE
        gpt_half_weights_path = Path(WEIGHTS_PATH) / args.fileID / GPT_HALF_WEIGHTS_FILE

        BotLogger.info(f"Train done. \n \
                        Sovits checkpoint saved in: {sovits_half_weights_path} \n \
                        GPT checkpoint saved in: {gpt_half_weights_path}")

    except Exception as e:
        BotLogger.error(
            f"Train failed | File: {args.fileID} | Traceback :\n{traceback.format_exc()}"
        )

def handle_inference(args):
    try:
        gpt_path = Path(WEIGHTS_PATH) / args.fileID / GPT_HALF_WEIGHTS_FILE
        sovits_path = Path(WEIGHTS_PATH) / args.fileID / SOVITS_HALF_WEIGHTS_FILE
        reasoning_result_path = Path(REASONING_RESULT_PATH) / args.fileID
        if not os.path.exists(gpt_path):
            BotLogger.error("路径错误！找不到GPT模型")
        if not os.path.exists(sovits_path):
            BotLogger.error("路径错误！找不到SOVITS模型")
        if not os.path.exists(reasoning_result_path):
            os.makedirs(reasoning_result_path, exist_ok=True)
        if args.version == 'v2':
            inference = v2(gpt_path, sovits_path)
        else:
            inference = v4(gpt_path, sovits_path)
        
        # Synthesize audio
        synthesis_result = inference.inference(ref_wav_path=args.refWavFilePath,# 参考音频 
                                    prompt_text=args.promptText, # 参考文本
                                    prompt_language="中文", 
                                    text=args.targetText, # 目标文本
                                    text_language="中文", top_p=1, temperature=1, top_k=15, speed=1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()
        result_list = list(synthesis_result)
        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            sf.write(reasoning_result_path / REASONING_RESULT_FILE, last_audio_data, last_sampling_rate)
            BotLogger.info(f"Audio saved to {reasoning_result_path / REASONING_RESULT_FILE}")
    except Exception as e:
        BotLogger.error(
            f"inference failed | File: {args.fileID} | Traceback :\n{traceback.format_exc()}"
        )
def main():
    parser = argparse.ArgumentParser(prog='mockvoi', description=CLI_HELP_MSG)
    subparsers = parser.add_subparsers(dest='command', help='')

    # upload 子命令
    parser_upload = subparsers.add_parser('upload', help='upload specified file.')
    parser_upload.add_argument('file', type=str, help='full file path to upload.')
    parser_upload.add_argument('--no-denoise', dest='denoise', action='store_false', 
                               help='disable denoise processing (default: enable denoise).')
    parser_upload.set_defaults(denoise=True)
    parser_upload.set_defaults(func=handle_upload)

    # inference 子命令
    parser_inference = subparsers.add_parser('inference', help='')
    parser_inference.add_argument('fileID', type=str, help='returned train id from train.')
    parser_inference.add_argument('refWavFilePath', type=str, help='reference file full path.')
    parser_inference.add_argument('promptText', type=str, help='prompt text.')
    parser_inference.add_argument('targetText', type=str, help='target text.')
    parser_inference.add_argument('--no-denoise', dest='denoise', action='store_false', 
                               help='disable denoise processing (default: enable denoise).')
    parser_inference.set_defaults(denoise=True)
    parser_inference.add_argument('--version', type=str, default='v4', help='the default version is v4.')
    parser_inference.set_defaults(func=handle_inference)

    # train 子命令
    parser_train = subparsers.add_parser('train', help='train specified file id.')
    parser_train.add_argument('fileID', type=str, help='returned file id from upload.')
    parser_train.add_argument('--no-denoise', dest='denoise', action='store_false', 
                               help='disable denoise processing (default: enable denoise).')
    parser_train.set_defaults(denoise=True)
    parser_train.add_argument('--epochs_sovits', type=int, default=10, help='train epochs of SoVITS (default:10).')
    parser_train.add_argument('--epochs_gpt', type=int, default=10, help='train epochs of GPT (default:10).')
    parser_train.set_defaults(func=handle_train)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()