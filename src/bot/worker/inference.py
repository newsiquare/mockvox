from .worker import app
from bot.core.inference import Inferener
from bot.utils import i18n
import os
from fastapi import Form
import soundfile as sf

@app.task(name="inference", bind=True)
async def inference_task(GPT_model_path:str = Form(..., description="GPT模型路径"), 
                   SoVITS_model_path:str = Form(..., description="sovits模型路径") , 
                   ref_audio_path:str = Form(..., description="参考音频路径"), 
                   ref_text:str = Form(..., description="参考音频文字"), 
                   ref_language:str = Form(..., description="参考音频语言"), 
                   target_text:str = Form(..., description="要生成的文字"), 
                   target_language:str = Form(..., description="要生成音频语言"), 
                   output_path:str = Form(..., description="结果保存路径"), 
                   top_p:int = Form(..., description="top_p"), 
                   top_k:int = Form(..., description="GPT采样参数(无参考文本时不要太低。不懂就用默认)："), 
                   temperature:int = Form(..., description="温度"), 
                   speed:int = Form(..., description="语速")):
    inference = Inferener(GPT_model_path,SoVITS_model_path)

    # Synthesize audio
    synthesis_result = inference.inference(ref_wav_path=ref_audio_path,# 参考音频 
                                prompt_text=ref_text, # 参考文本
                                prompt_language=i18n(ref_language), 
                                text=target_text, # 目标文本
                                text_language=i18n(target_language), top_p=top_p, temperature=temperature, top_k=top_k, speed=speed)
    
    result_list = list(synthesis_result)
    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        # output_path = os.path.join("/home/easyman/zjh/bot/", "output.wav")
        sf.write(output_path, last_audio_data, last_sampling_rate)
        return f"Audio saved to {output_path}"
    pass