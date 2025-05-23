# from .worker import app
from bot.engine.v4.inference import Inferencer as v4
from bot.engine.v2.inference import Inferencer as v2
from bot.utils import i18n
import soundfile as sf
import torch
import gc
import os
import time
from pathlib import Path
from bot.config import OUT_PUT_PATH,OUT_PUT_FILE
from .worker import celeryApp
from collections import OrderedDict

@celeryApp.task(name="inference", bind=True)
def inference_task(self,gpt_model_path:str , 
                   soVITS_model_path:str, 
                   ref_audio_path:str , 
                   ref_text:str , 
                   ref_language:str, 
                   target_text:str , 
                   target_language:str,  
                   top_p:float, 
                   top_k:int, 
                   temperature:float, 
                   speed:float,
                   version:str):
    if version == "v2":
        inference = v2(gpt_model_path,soVITS_model_path)
    else:
        inference = v4(gpt_model_path,soVITS_model_path)
    # Synthesize audio
    synthesis_result = inference.inference(ref_wav_path=ref_audio_path,# 参考音频 
                                prompt_text=ref_text, # 参考文本
                                prompt_language=i18n(ref_language), 
                                text=target_text, # 目标文本
                                text_language=i18n(target_language), top_p=top_p, temperature=temperature, top_k=top_k, speed=speed)
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()  
    result_list = list(synthesis_result)
    outputname = os.path.join(Path(OUT_PUT_PATH), OUT_PUT_FILE+"_"+self.request.id+".WAV")
    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        sf.write( outputname,  last_audio_data, last_sampling_rate)
        results = OrderedDict()
        results["task_id"] = self.request.id
        results["path"] = Path(outputname).name
        return {
            "status": "success", 
            "results": results, 
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    

if __name__ == "__main__":
    inference = v4("/home/easyman/zjh/bot/test/gpt.pth","/home/easyman/zjh/bot/test/sovits.pth")
    # Synthesize audio
    synthesis_result = inference.inference(ref_wav_path="/home/easyman/zjh/bot/test/0018652800_0018799360.wav",# 参考音频 
                                prompt_text="我还是走吧，拜身如柳絮，随风摆。", # 参考文本
                                prompt_language=i18n("中文"), 
                                text="咱当兵的，怕过啥？小鬼子的枪炮再厉害，那也得给老子让道！", # 目标文本
                                text_language=i18n("中文"), top_p=0.6, temperature=0.6, top_k=20, speed=1)
    
    result_list = list(synthesis_result)
    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        # output_path = os.path.join("/home/easyman/zjh/bot/", "output.wav")
        sf.write("/home/easyman/zjh/bot/output.wav", last_audio_data, last_sampling_rate)
        print(f"Audio saved to /home/easyman/zjh/bot/output.wav")