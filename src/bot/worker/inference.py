# from .worker import app
from bot.core.inference import Inferencer
from bot.utils import i18n
import os
from fastapi import Form
import soundfile as sf

# @app.task(name="inference", bind=True)
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
    inference = Inferencer(GPT_model_path,SoVITS_model_path)

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

if __name__ == "__main__":
    inference = Inferencer("/home/easyman/zjh/bot/test/gpt.pth","/home/easyman/zjh/bot/test/sovits.pth")
    # Synthesize audio
    synthesis_result = inference.inference(ref_wav_path="/home/easyman/zjh/bot/test/LIYUNLONG.WAV",# 参考音频 
                                prompt_text="少给老子谈什么清规戒律。说，是不是偷喝我酒了？哈哈哈。你小子嘴还挺硬。那我的酒怎么少了。", # 参考文本
                                prompt_language=i18n("中文"), 
                                text="咱当兵的，怕过啥？小鬼子的枪炮再厉害，那也得给老子让道！战场上子弹不长眼咋地？狭路相逢勇者胜，咱端着刺刀冲上去，就是要让敌人知道，咱中国军人的骨头比钢铁还硬！别跟我扯什么战术战略，关键时刻就得敢打敢拼，犹豫一秒钟，敌人的子弹就钻你脑壳！​咱独立团的兵，个个都是嗷嗷叫的狼崽子！平时训练多苦多累都给我忍着，上了战场才能像老虎一样扑向敌人！要是谁在战场上拉稀摆带，临阵退缩，老子第一个毙了他！记住，咱丢啥都不能丢了咱的精气神，就算剩下最后一个人，也要拉响手榴弹和敌人同归于尽！", # 目标文本
                                text_language=i18n("中文"), top_p=0.6, temperature=0.6, top_k=20, speed=1)
    
    result_list = list(synthesis_result)
    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        # output_path = os.path.join("/home/easyman/zjh/bot/", "output.wav")
        sf.write("/home/easyman/zjh/bot/output.wav", last_audio_data, last_sampling_rate)
        print(f"Audio saved to /home/easyman/zjh/bot/output.wav")