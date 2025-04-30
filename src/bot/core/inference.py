import os, re
import torch
from typing import Optional
from bot.utils import BotLogger
from bot.utils import i18n
from bot.models.SynthesizerTrn import SynthesizerTrn
from time import time as ttime
import numpy as np
import librosa
from bot.models.HuBERT import CNHubert
from bot.text import chinese
from bot.text import cleaned_text_to_sequence
from bot.text import symbols 
from transformers import AutoTokenizer, AutoModelForMaskedLM
import traceback
from bot.models.mel import spectrogram_torch
from bot.models.AR.t2s_model import Text2SemanticDecoder

class Inferener:
    def __init__(
        self,
        gpt_path: Optional[str] = None,
        sovits_path: Optional[str] = None
    ):
        self.device = "cuda:0"
        self.dict_language = {
            i18n("中文"): "all_zh",#全部按中文识别
            i18n("英文"): "en",#全部按英文识别#######不变
            i18n("日文"): "all_ja",#全部按日文识别
            i18n("粤语"): "all_yue",#全部按中文识别
            i18n("韩文"): "all_ko",#全部按韩文识别
            i18n("中英混合"): "zh",#按中英混合识别####不变
            i18n("日英混合"): "ja",#按日英混合识别####不变
            i18n("粤英混合"): "yue",#按粤英混合识别####不变
            i18n("韩英混合"): "ko",#按韩英混合识别####不变
            i18n("多语种混合"): "auto",#多语种启动切分识别语种
            i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
        }
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


        self.punctuation = set(['!', '?', '…', ',', '.', '-'," "])
        self.hz = 50
        self.t2s_model,self.config,self.max_sec = self._change_gpt_weights(gpt_path)
        
        self.vq_model, self.hps = self._change_sovits_weights(sovits_path)
        
    def _change_gpt_weights(self, gpt_path):
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticDecoder(config=config, top_k=3)
        
        state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v 
                for k, v in dict_s1["weight"].items()}
        t2s_model.load_state_dict(state_dict)
        t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        return t2s_model,config,max_sec


    def _change_sovits_weights(self, sovits_path):
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        hps.model.version = "v2"

        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        vq_model = vq_model.half().to(self.device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        return vq_model, hps

    def cut1(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)


    def cut2(self,inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        # print(opts)
        if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)


    def cut3(self, inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return  "\n".join(opts)

    def cut4(self, inp):
        inp = inp.strip("\n")
        opts = re.split(r'(?<!\d)\.(?!\d)', inp.strip("."))
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)


    # contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
    def cut5(self, inp):
        inp = inp.strip("\n")
        punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item for item in mergeitems if not set(item).issubset(punds)]
        return "\n".join(opt)


    def process_text(self, texts):
        _text=[]
        if all(text in [None, " ", "\n",""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in  [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text


    def merge_short_text_in_array(self, texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result


    def clean_text_inf(self,text, language):
        phones, word2ph, norm_text = self.clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text





    def clean_text(self, text, language):
        special = [
        # ("%", "zh", "SP"),
        ("￥", "zh", "SP2"),
        ("^", "zh", "SP3"),
        # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
        ]
        for special_s, special_l, target_symbol in special:
            if special_s in text and language == special_l:
                return self.clean_special(text, language, special_s, target_symbol)
        chinesen = chinese.ChineseNormalizer()
        norm_text = chinesen.do_normalize(text)
        if language == "zh" or language=="yue":##########
            phones, word2ph = chinesen.g2p(norm_text)
            assert len(phones) == sum(word2ph)
            assert len(norm_text) == len(word2ph)
        elif language == "en":
            phones = chinesen.g2p(norm_text)
            if len(phones) < 4:
                phones = [','] + phones
            word2ph = None
        else:
            phones = chinesen.g2p(norm_text)
            word2ph = None
        phones = ['UNK' if ph not in symbols else ph for ph in phones]
        return phones, word2ph, norm_text


    def clean_special(self, text, language, special_s, target_symbol):
        
        """
        特殊静音段sp符号处理
        """
        chinesen = chinese.ChineseNormalizer()
        text = text.replace(special_s, ",")
        norm_text = chinesen.do_normalize(text)
        phones = chinesen.g2p(norm_text)
        new_ph = []
        for ph in phones[0]:
            assert ph in symbols
            if ph == ",":
                new_ph.append(target_symbol)
            else:
                new_ph.append(ph)
        return new_ph, phones[1], norm_text






    def get_bert_feature(self, text, word2ph):
        bert_path = os.environ.get(
        "bert_path", "pretrained/GPT-SoVITS/chinese-roberta-wwm-ext-large"
        )
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        bert_model = bert_model.half().to(self.device)
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T


    def get_phones_and_bert(self, text,language,final=False):
        
        language = language.replace("all_","")
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext,"zh")
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
                bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext,"yue")
        else:
            phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16,
            ).to(self.device)
        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text,language,final=True)

        return phones,bert.to(torch.float16),norm_text


    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16,
            ).to(self.device)

        return bert


    def split(self,todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts


    def get_spepc(self, hps, filename):
        # audio = load_audio(filename, int(hps.data.sampling_rate))
        audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx=audio.abs().max()
        if(maxx>1):audio/=min(2,maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec




    def inference(self,ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("按标点符号切"), top_k=20, top_p=0.6, temperature=0.6, ref_free = False,speed=1,inp_refs=None):
        print(self.hps)
        if ref_wav_path:pass
        else:BotLogger.error(i18n('请上传参考音频'))
        if text:pass
        else:BotLogger.error(i18n('请填入推理文本'))
        t = []
        t0 = ttime()
        prompt_language = self.dict_language[prompt_language]
        text_language = self.dict_language[text_language]
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in self.splits:
                prompt_text += "。" if prompt_language != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
        text = text.strip("\n")

        print(i18n("实际输入的目标文本:"), text)
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16,
        )
        ssl_model = CNHubert()
        ssl_model.eval()
        ssl_model = ssl_model.half().to(self.device)
        zero_wav_torch = torch.from_numpy(zero_wav)
        zero_wav_torch = zero_wav_torch.half().to(self.device)
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                BotLogger.error(i18n("参考音频在3~10秒范围外，请更换！"))
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            
            wav16k = wav16k.half().to(self.device)
            
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

        t1 = ttime()
        t.append(t1-t0)
        if (how_to_cut == i18n("凑四句一切")):
            text = self.cut1(text)
        elif (how_to_cut == i18n("凑50字一切")):
            text = self.cut2(text)
        elif (how_to_cut == i18n("按中文句号。切")):
            text = self.cut3(text)
        elif (how_to_cut == i18n("按英文句号.切")):
            text = self.cut4(text)
        elif (how_to_cut == i18n("按标点符号切")):
            text = self.cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = self.process_text(texts)
        texts = self.merge_short_text_in_array(texts, 5)
        audio_opt = []
        phones1,bert1,norm_text1=self.get_phones_and_bert(prompt_text, prompt_language)
        for i_text,text in enumerate(texts):
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in self.splits): text += "。" if text_language != "en" else "."
            print(i18n("实际输入的目标文本(每句):"), text)
            phones2,bert2,norm_text2=self.get_phones_and_bert(text, text_language)
            print(i18n("前端处理后的文本(每句):"), norm_text2)
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(self.device).unsqueeze(0)

            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            t2 = ttime()
            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            t3 = ttime()
            refers=[]
            if(inp_refs):
                for path in inp_refs:
                    try:
                        refer = self.get_spepc(self.hps, path.name).to(torch.float16).to(self.device)
                        refers.append(refer)
                    except:
                        traceback.print_exc()
            if(len(refers)==0):refers = [self.get_spepc(self.hps, ref_wav_path).to(torch.float16).to(self.device)]
            audio = self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refers,speed=speed)[0, 0]
            
            max_audio=torch.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio=audio/max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav_torch)
            t4 = ttime()
            t.extend([t2 - t1,t3 - t2, t4 - t3])
            t1 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
        audio_opt = torch.cat(audio_opt, 0)
        sr=self.hps.data.sampling_rate 
        yield sr, (audio_opt.cpu().detach().numpy()* 32767).astype(np.int16)


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

