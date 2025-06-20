import logging
import re

# jieba静音
import jieba

jieba.setLogLevel(logging.CRITICAL)

SPECIAL_CHARS = r"0-9〜~,.;:!?，。！？；：、·([{<（【《〈「『“‘)\]}>）】》〉」』”’\"-_——\#$%&……￥'*+<=>?@[\]^_`{|}~ "

from split_lang import LangSplitter


def full_en(text):
    pattern = r"^[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$"
    return bool(re.match(pattern, text))


def full_cjk(text):
    # 来自wiki
    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DB5),  # CJK Extension A
        (0x20000, 0x2A6DD),  # CJK Extension B
        (0x2A700, 0x2B73F),  # CJK Extension C
        (0x2B740, 0x2B81F),  # CJK Extension D
        (0x2B820, 0x2CEAF),  # CJK Extension E
        (0x2CEB0, 0x2EBEF),  # CJK Extension F
        (0x30000, 0x3134A),  # CJK Extension G
        (0x31350, 0x323AF),  # CJK Extension H
        (0x2EBF0, 0x2EE5D),  # CJK Extension H
    ]

    pattern = rf"^[{SPECIAL_CHARS}]+$"

    cjk_text = ""
    for char in text:
        code_point = ord(char)
        in_cjk = any(start <= code_point <= end for start, end in cjk_ranges)
        if in_cjk or re.match(pattern, char):
            cjk_text += char
    return cjk_text


def split_jako(tag_lang, item):
    if tag_lang == "ja":
        pattern = rf"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[{SPECIAL_CHARS}]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
    else:
        pattern = rf"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[{SPECIAL_CHARS}]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"

    lang_list: list[dict] = []
    tag = 0
    for match in re.finditer(pattern, item["text"]):
        if match.start() > tag:
            lang_list.append({"lang": item["lang"], "text": item["text"][tag : match.start()]})

        tag = match.end()
        lang_list.append({"lang": tag_lang, "text": item["text"][match.start() : match.end()]})

    if tag < len(item["text"]):
        lang_list.append({"lang": item["lang"], "text": item["text"][tag : len(item["text"])]})

    return lang_list


def merge_lang(lang_list, item):
    if lang_list and item["lang"] == lang_list[-1]["lang"]:
        lang_list[-1]["text"] += item["text"]
    else:
        lang_list.append(item)
    return lang_list


class LangSegmenter:
    # 默认过滤器, 基于gsv目前四种语言
    DEFAULT_LANG_MAP = {
        "zh": "zh",
        "yue": "zh",  # 粤语
        "wuu": "zh",  # 吴语
        "zh-cn": "zh",
        "zh-tw": "x",  # 繁体设置为x
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }

    def getTexts(text):
        lang_splitter = LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)
        substr = lang_splitter.split_by_lang(text=text)

        lang_list: list[dict] = []

        for _, item in enumerate(substr):
            dict_item = {"lang": item.lang, "text": item.text}

            # 处理短英文被识别为其他语言的问题
            if full_en(dict_item["text"]):
                dict_item["lang"] = "en"
                lang_list = merge_lang(lang_list, dict_item)
                continue

            # 处理非日语夹日文的问题(不包含CJK)
            ja_list: list[dict] = []
            if dict_item["lang"] != "ja":
                ja_list = split_jako("ja", dict_item)

            if not ja_list:
                ja_list.append(dict_item)

            # 处理非韩语夹韩语的问题(不包含CJK)
            ko_list: list[dict] = []
            temp_list: list[dict] = []
            for _, ko_item in enumerate(ja_list):
                if ko_item["lang"] != "ko":
                    ko_list = split_jako("ko", ko_item)

                if ko_list:
                    temp_list.extend(ko_list)
                else:
                    temp_list.append(ko_item)

            # 未存在非日韩文夹日韩文
            if len(temp_list) == 1:
                # 未知语言检查是否为CJK
                if dict_item["lang"] == "x":
                    cjk_text = full_cjk(dict_item["text"])
                    if cjk_text:
                        dict_item = {"lang": "zh", "text": cjk_text}
                        lang_list = merge_lang(lang_list, dict_item)
                    continue
                else:
                    lang_list = merge_lang(lang_list, dict_item)
                    continue

            # 存在非日韩文夹日韩文
            for _, temp_item in enumerate(temp_list):
                # 未知语言检查是否为CJK
                if temp_item["lang"] == "x":
                    cjk_text = full_cjk(dict_item["text"])
                    if cjk_text:
                        dict_item = {"lang": "zh", "text": cjk_text}
                        lang_list = merge_lang(lang_list, dict_item)
                else:
                    lang_list = merge_lang(lang_list, temp_item)
        return lang_list


if __name__ == "__main__":
    text = "MyGO?,你也喜欢まいご吗？"
    print(LangSegmenter.getTexts(text))

    text = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。"
    print(LangSegmenter.getTexts(text))

    text = r"こんにちは안녕하세요你好！🍵今日の天気はどうですか？오늘 날씨 어때? 今天天气如何？" \
             r"12345@#$%^&*()_+-=[]{}\|;:',.<>/?！？、。〜～【】《》「」『』“”‘’¥£€💵" \
             r"日本語と한국어と中文を混ぜました！피자食べたい? 我想吃披萨🍕！✈️旅行に行きましょう~" \
             r"한글(韓文), 漢字(汉字), ひらがな(Hiragana) ★★★重要！！"
    print(LangSegmenter.getTexts(text))
    
