import os
import cn2an
from pypinyin import G2PWPinyin

normalizer = lambda x: cn2an.transform(x, "an2cn")

pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(os.path.dirname(__file__), "opencpop-strict.txt")).readlines()
}

g2pw = G2PWPinyin()