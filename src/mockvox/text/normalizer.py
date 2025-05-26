from .symbols import symbols
from .chinese import ChineseNormalizer
from .cantonese import CantoneseNormalizer
from .english import EnglishNormalizer
from .japanese import JapaneseNormalizer
from .korean import KoreanNormalizer

symbol_to_id = {s: i for i, s in enumerate(symbols)}

class NormalizerFactory:
    # 定义语言码与类的映射关系
    NORMALIZER_MAP = {
        'zh': ChineseNormalizer,
        'en': EnglishNormalizer,
        'ja': JapaneseNormalizer,
        'ko': KoreanNormalizer,
        'can': CantoneseNormalizer  # 粤语
    }

    @classmethod
    def get_normalizer(cls, language_code, *args, **kwargs):
        """根据语言码返回归一化实例"""
        normalizer_class = cls.NORMALIZER_MAP.get(language_code)
        if not normalizer_class:
            raise ValueError(f"Unsupported language code: {language_code}")
        return normalizer_class(*args, **kwargs)

class Normalizer:
    '''
    每个语言的Normalizer类, 都需要实现 do_normalize 和 g2p 方法
    '''
    def __init__(self, language, *args, **kwargs):
        self.normalizer = NormalizerFactory.get_normalizer(language, *args, **kwargs)

    @staticmethod
    def cleaned_text_to_sequence(text):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
            text: string to convert to a sequence
        Returns:
            List of integers corresponding to the symbols in the text
        '''
        phones = [symbol_to_id[symbol] for symbol in text]
        return phones
    
    def do_normalize(self, text):
        return self.normalizer.do_normalize(text)
    
    def g2p(self, text):
        return self.normalizer.g2p(text)
