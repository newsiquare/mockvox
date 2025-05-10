import re
import os
import hashlib

import pyopenjtalk

def get_hash(fp: str) -> str:
    hash_md5 = hashlib.md5()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

current_file_path = os.path.dirname(__file__)
USERDIC_CSV_PATH = os.path.join(current_file_path, "userdict.csv")
USERDIC_BIN_PATH = os.path.join(current_file_path, "user.dict")
USERDIC_HASH_PATH = os.path.join(current_file_path, "userdict.md5")

if __name__ == "__main__":
    # 如果没有用户词典，就生成一个；如果有，就检查md5，如果不一样，就重新生成
    if os.path.exists(USERDIC_CSV_PATH):
        if (
            not os.path.exists(USERDIC_BIN_PATH)
            or get_hash(USERDIC_CSV_PATH) != open(USERDIC_HASH_PATH, "r", encoding="utf-8").read()
        ):
            pyopenjtalk.mecab_dict_index(USERDIC_CSV_PATH, USERDIC_BIN_PATH)
            with open(USERDIC_HASH_PATH, "w", encoding="utf-8") as f:
                f.write(get_hash(USERDIC_CSV_PATH))

    if os.path.exists(USERDIC_BIN_PATH):
        pyopenjtalk.update_global_jtalk_with_user_dict(USERDIC_BIN_PATH)