import os, re
import torch
from typing import Optional
from mockvox.utils import MockVoxLogger
from mockvox.utils import i18n
from time import time as ttime
import numpy as np
import librosa
from mockvox.models import CNHubert
from mockvox.text import normalizer as nl
from mockvox.text import Normalizer
from mockvox.text import symbols 
from mockvox.config import (
    PRETRAINED_PATH,
    PRETRAINED_S2GV4_FILE,
    PRETRAINED_VOCODER_FILE)
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torchaudio
from mockvox.nn.mel import spectrogram_torch
from mockvox.models.v2.t2s_model import Text2SemanticDecoder
from io import BytesIO
from mockvox.models.v4.synthesizer import SynthesizerTrnV3
from mockvox.models.v2.SynthesizerTrn import Generator,SynthesizerTrn
from peft import LoraConfig, get_peft_model
from mockvox.nn import mel_spectrogram_torch
from mockvox.text.LangSegmenter import LangSegmenter
import traceback

class Inferencer:
    MODEL_MAPPING = {
        "zh": "GPT-SoVITS/chinese-roberta-wwm-ext-large",
        "en": "FacebookAI/roberta-large",
        "ja": "tohoku-nlp/bert-large-japanese-v2",
        "ko": "klue/roberta-large",
        "can": "GPT-SoVITS/chinese-roberta-wwm-ext-large"
        }  
    def __init__(
        self,
        gpt_path: Optional[str] = None,
        sovits_path: Optional[str] = None,
        version: Optional[str] = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }     
        self.punctuation = set(['!', '?', '…', ',', '.', '-'," "])
        self.hz = 50
        self.t2s_model,self.config,self.max_sec = self._change_gpt_weights(gpt_path)
        self.vq_model, self.hps,self.mel_fn_v4 = self._change_sovits_weights(sovits_path)
        if version=="v4":
            self.resample_transform_dict={}
            self.hifigan_model = self._init_hifigan()

    def _init_hifigan(self):
        hifigan_model = Generator(
            initial_channel=100,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 6, 2, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 12, 4, 4, 4],
            gin_channels=0,is_bias=True
        )
        hifigan_model.eval()
        hifigan_model.remove_weight_norm()
        state_dict_g = torch.load(PRETRAINED_VOCODER_FILE, map_location="cpu")
        MockVoxLogger.info(f"loading vocoder {hifigan_model.load_state_dict(state_dict_g)}")

        return hifigan_model.half().to(self.device)
        
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

        dict_s2, if_lora_v3 = self._load_sovits_new(sovits_path)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if hps.model.version == "v4":
            vq_model = SynthesizerTrnV3(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length_v4,
                n_speakers=hps.data.n_speakers,
                **hps.model
            )
        else:
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model
            )
            if_lora_v3=False
        vq_model = vq_model.half().to(self.device)
        vq_model.eval()
        if if_lora_v3 == False:
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
        else:
            gv4_model,if_lora_v3 = self._load_sovits_new(PRETRAINED_S2GV4_FILE)
            vq_model.load_state_dict(gv4_model["weight"], strict=False)
            lora_rank = hps["train"]["lora_rank"]
            # lora_rank = 32
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
            MockVoxLogger.info(f"loading sovits_v4_lora{lora_rank}")
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
            vq_model.cfm = vq_model.cfm.merge_and_unload()
            vq_model.eval()
        mel_fn_v4 = lambda x: mel_spectrogram_torch(
            x,
            **{
                "n_fft": hps["data"]["filter_length"],
                "win_size": hps["data"]["win_length"],
                "hop_size": hps["data"]["hop_length_v4"],
                "num_mels": hps["data"]["n_mel_channels_v4"],
                "sampling_rate": hps["data"]["sampling_rate"],
                "fmin": hps["data"]["mel_fmin"],
                "fmax": hps["data"]["mel_fmax"],
                "center": False,
            },
        )
        return vq_model, hps,mel_fn_v4
    
    def _load_sovits_new(self, path_sovits):
        f = open(path_sovits, "rb")
        if_lora_v3 = False
        meta = f.read(2)
        if meta != "PK":
            if_lora_v3 = True
            data = b"PK" + f.read()
            bio = BytesIO()
            bio.write(data)
            bio.seek(0)
            return torch.load(bio, map_location="cpu"),if_lora_v3
        return torch.load(path_sovits, map_location="cpu"),if_lora_v3

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
        if len(opts) > 1 and len(opts[-1]) < 50:  
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
    
    def cut5(self, inp):
        inp = inp.strip("\n")
        punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…','《','》'}
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
        language = language.replace("all_", "")
        phones, word2ph, norm_text = self.clean_text(text, language)
        phones = Normalizer.cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    def clean_text(self, text, language):
        special = [
        ("￥", "zh", "SP2"),
        ("^", "zh", "SP3"),
        ]
        normalizer = nl.Normalizer(language)
        for special_s, special_l, target_symbol in special:
            if special_s in text and language == special_l:
                return self.clean_special(text, special_s, target_symbol,normalizer)
        
        norm_text = normalizer.do_normalize(text)
        if language == "zh" or language=="can":##########
            phones, word2ph = normalizer.g2p(norm_text)
            assert len(phones) == sum(word2ph)
            assert len(norm_text) == len(word2ph)
        elif language in ["en", "ja", "ko"]:
            phones,word2ph = normalizer.g2p(norm_text)
        else:
            phones = normalizer.g2p(norm_text)
            word2ph = None
        phones = ['UNK' if ph not in symbols else ph for ph in phones]
        return phones, word2ph, norm_text

    def clean_special(self, text, special_s, target_symbol, normalizer):        
        """
        特殊静音段sp符号处理
        """
        text = text.replace(special_s, ",")
        norm_text = normalizer.do_normalize(text)
        phones = normalizer.g2p(norm_text)
        new_ph = []
        for ph in phones[0]:
            assert ph in symbols
            if ph == ",":
                new_ph.append(target_symbol)
            else:
                new_ph.append(ph)
        return new_ph, phones[1], norm_text

    def get_bert_feature(self, text, word2ph,language):
        
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        if language == "zh":
            assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_phones_and_bert(self, text,language,final=False):        
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_can"}:
            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            normalizer = nl.Normalizer(language.replace("all_", ""))
            if language == "all_zh":
                if re.search(r"[A-Za-z]", formattext):
                    formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                    formattext = normalizer.do_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh")
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
                    bert = self.get_bert_feature(norm_text, word2ph,language).to(self.device)
            elif language == "all_can" and re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = normalizer.do_normalize(formattext)
                return self.get_phones_and_bert(formattext, "can")
            elif language in {"all_ja", "en", "all_ko"}:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
                bert = self.get_bert_feature(norm_text, word2ph,language).to(self.device)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "can", "auto", "auto_can"}:
            textlist = []
            langlist = []
            if language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_can":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "can"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)
        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, final=True)

        return phones, bert.to(torch.float16), norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh" or language == "ja":
            bert = self.get_bert_feature(norm_text, word2ph,language).to(self.device)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16,
            ).to(self.device)
        return bert

    def split(self,todo_text):
        todo_text = str(todo_text).replace("……", "。").replace("——", "，")
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
        if hps.model.version == "v4":
            hot_length = hps.data.hop_length_v4
        else:
            hot_length = hps.data.hop_length
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hot_length,
            hps.data.win_length,
            center=False,
        )
        return spec

    def inference(self,ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("凑四句一切"), top_k=15, top_p=1,inp_refs=None, temperature=1,speed=1,is_stream=False):
        if ref_wav_path:
            pass
        else:
            MockVoxLogger.error(i18n('请上传参考音频'))
            return
        if text:
            pass
        else:
            MockVoxLogger.error(i18n('请填入推理文本'))
            return
        # t = []
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
            
        # t0 = ttime()

        dict_language = [
            "all_zh",#全部按中文识别
            "en",#全部按英文识别#######不变
            "all_ja",#全部按日文识别
            "all_can",#全部按中文识别
            "all_ko",#全部按韩文识别
            "zh",#按中英混合识别####不变
            "ja",#按日英混合识别####不变
            "can",#按粤英混合识别####不变
            "ko",#按韩英混合识别####不变
            "auto",#多语种启动切分识别语种
             "auto_can",#多语种启动切分识别语种
        ]
        if prompt_language not in dict_language or text_language not in dict_language:
            MockVoxLogger.error(i18n('语言不存在'))
            return
        
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in self.splits:
            prompt_text += "。" if prompt_language != "en" else "."
        MockVoxLogger.info(i18n("实际输入的参考文本:")+prompt_text)
        text = text.strip("\n")

        MockVoxLogger.info(i18n("实际输入的目标文本:")+text)
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16,
        )        
        
        ssl_model = CNHubert()
        ssl_model.eval()
        ssl_model = ssl_model.half().to(self.device)
        bert_language = ""
        if "zh" in text_language or "zh" in prompt_language:
            bert_language="zh"
        if "ja" in text_language or "ja" in prompt_language:
            bert_language="ja"
        if "ko" in text_language or "ko" in prompt_language:
            bert_language="ko"
        bert_path = os.path.join(PRETRAINED_PATH,self.MODEL_MAPPING.get(bert_language, "GPT-SoVITS/chinese-roberta-wwm-ext-large"))
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert_model = self.bert_model.half().to(self.device)
        zero_wav_torch = torch.from_numpy(zero_wav)
        zero_wav_torch = zero_wav_torch.half().to(self.device)

        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                MockVoxLogger.error(i18n("参考音频在3~10秒范围外，请更换！"))
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            
            wav16k = wav16k.half().to(self.device)
            
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

        # t1 = ttime()
        # t.append(t1-t0)
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
        
        phones1,bert1,norm_text1=self.get_phones_and_bert(prompt_text, prompt_language)
        audio_opt = []
        # MockVoxLogger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
        for i_text,text in enumerate(texts):
            
            if len(text.strip()) == 0:
                continue
            # 解决输入目标文本的空行导致报错的问题
            
            if text[-1] not in self.splits: 
                text += "。" if text_language != "en" else "."
            MockVoxLogger.info(i18n("实际输入的目标文本(每句):")+text)
            phones2,bert2,norm_text2=self.get_phones_and_bert(text, text_language)
            MockVoxLogger.info(i18n("前端处理后的文本(每句):")+norm_text2)
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            # t2 = ttime()
            
            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    
            # t3 = ttime()
            if self.hps.model.version == "v4":
                refer = self.get_spepc(self.hps, ref_wav_path).to(self.device).to(torch.float16)
                phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
                fea_ref, ge = self.vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                ref_audio, sr = torchaudio.load(ref_wav_path)
                ref_audio = ref_audio.to(self.device).float()
                if ref_audio.shape[0] == 2:
                    ref_audio = ref_audio.mean(0).unsqueeze(0)
                tgt_sr= 32000
                if sr != tgt_sr:
                    ref_audio = self.resample(ref_audio, sr,tgt_sr)
                mel2 = self.mel_fn_v4(ref_audio)
                mel2 = self.norm_spec(mel2)
                T_min = min(mel2.shape[2], fea_ref.shape[2])
                mel2 = mel2[:, :, :T_min]
                fea_ref = fea_ref[:, :, :T_min]
                Tref= 500
                Tchunk= 1000
                if T_min > Tref:
                    mel2 = mel2[:, :, -Tref:]
                    fea_ref = fea_ref[:, :, -Tref:]
                    T_min = Tref
                chunk_len = Tchunk - T_min
                mel2 = mel2.to(torch.float16)
                fea_todo, ge = self.vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
                cfm_resss = []
                idx = 0
                while 1:
                    fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                    if fea_todo_chunk.shape[-1] == 0:
                        break
                    idx += chunk_len
                    fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                    # 如果有点电可以调整下面的参数{8,16,32,64,128}
                    cfm_res = self.vq_model.cfm.inference(
                        fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, 32, inference_cfg_rate=0
                    )
                    cfm_res = cfm_res[:, :, mel2.shape[2] :]
                    mel2 = cfm_res[:, :, -T_min:]
                    fea_ref = fea_todo_chunk[:, :, -T_min:]
                    cfm_resss.append(cfm_res)
                cfm_res = torch.cat(cfm_resss, 2)
                cfm_res = self.denorm_spec(cfm_res)
                if self.hifigan_model == None:
                    self._init_hifigan()
                vocoder_model=self.hifigan_model
                with torch.inference_mode():
                    wav_gen = vocoder_model(cfm_res)
                    audio = wav_gen[0][0]
            else:
                refers=[]
                if inp_refs:
                    for path in inp_refs:
                        try:
                            refer = self.get_spepc(self.hps, path.name).to(torch.float16).to(self.device)
                            refers.append(refer)
                        except:
                            traceback.print_exc()
                if(len(refers)==0):
                    refers = [self.get_spepc(self.hps, ref_wav_path).to(torch.float16).to(self.device)]
                audio = self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refers,speed=speed)[0, 0]
            audio = torch.clamp(audio, -1.0, 1.0)
            # max_audio=torch.abs(audio).max()
            # if max_audio>1:
            #     audio=audio/max_audio
            # audio = self.soft_clip(audio)
            audio_opt.append(audio)
            audio_opt.append(zero_wav_torch)
            if is_stream and len(audio_opt) == 2:
                audio_opt = torch.cat(audio_opt, 0)
                if self.hps.model.version == "v4":
                    opt_st = 48000
                else:
                    opt_st = self.hps.data.sampling_rate
                audio_opt = audio_opt.cpu().detach().numpy()
                yield opt_st, (audio_opt* 32767).astype(np.int16)
                audio_opt = []
            
        if len(audio_opt) > 0:
            audio_opt = torch.cat(audio_opt, 0)
            if self.hps.model.version == "v4":
                opt_st = 48000
            else:
                opt_st = self.hps.data.sampling_rate
            audio_opt = audio_opt.cpu().detach().numpy()
            yield opt_st, (audio_opt* 32767).astype(np.int16)

    def soft_clip(self, x, threshold=0.9):
        scale = torch.abs(x) - threshold
        scale = torch.clamp(scale, min=0)
        return torch.sign(x) * torch.where(
            torch.abs(x) > threshold,
            threshold + (1 - threshold) * torch.tanh(scale / (1 - threshold)),
            torch.abs(x)
    )

    def norm_spec(self,x):
        spec_min = -12
        spec_max = 2
        return (x - spec_min) / (spec_max - spec_min) * 2 - 1

    def denorm_spec(self,x):
        spec_min = -12
        spec_max = 2
        return (x + 1) / 2 * (spec_max - spec_min) + spec_min

    def resample(self,audio_tensor, sr0,sr1):        
        key="%s-%s"%(sr0,sr1)
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(self.device)
        return self.resample_transform_dict[key](audio_tensor)

    
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
