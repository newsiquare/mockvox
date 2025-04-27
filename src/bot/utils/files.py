import uuid
from hashlib import md5
from datetime import datetime
import os
import json
import copy
from pathlib import Path, PosixPath
from collections import OrderedDict
from io import BytesIO
import torch
from bot.utils import BotLogger

def save_checkpoint(model, hps, optimizer, learning_rate, iteration, checkpoint_path):
    BotLogger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(
        {
            "weight": state_dict,
            "config": hps.as_dict(),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "date": datetime.now().isoformat(),
            "author": "联宇创新(Lianyu Co,L.T.D)"
        },
        checkpoint_path
    )

def save_checkpoint_half_latest(model, hps, iteration, checkpoint_path):
    BotLogger.info(
        f"Saving latest half model state at iteration {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        ckpt = model.module.state_dict()
    else:
        ckpt = model.state_dict()
    
    half_ckpt = OrderedDict()
    for key in ckpt.keys():
        if "enc_q" in key:
            continue
        half_ckpt[key] = ckpt[key].half()
    
    torch.save(
        {
            "weight": half_ckpt,
            "config": hps.as_dict(),
            "epoch": iteration,
            "date": datetime.now().isoformat(),
            "author": "联宇创新(Lianyu Co,L.T.D)"
        },
        checkpoint_path
    )

def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["weight"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            traceback.print_exc()
            BotLogger.error(
                "error, %s is not in the checkpoint" % k
            )  # shape不对也会，比如text_embedding当cleaner修改时
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    
    BotLogger.info(f"加载模型参数: '{checkpoint_path}' (轮次: {iteration})")
    return model, optimizer, learning_rate, iteration

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = HParams(**value)
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def as_dict(self):
        """将 HParams 对象转换为纯字典, Path/PosixPath 转为字符串"""
        def _to_dict(obj):
            if isinstance(obj, HParams):
                return {k: _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (Path, PosixPath)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            else:
                return obj
        return _to_dict(self.__dict__)

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        """反序列化时重建嵌套 HParams"""
        def _to_hparams(d):
            if isinstance(d, dict):
                return HParams(**{k: _to_hparams(v) for k, v in d.items()})
            else:
                return d
        self.__dict__.update(_to_hparams(state))

    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建 HParams 对象（兼容性方法）"""
        return cls(**config_dict)

    def __repr__(self):
        return self.as_dict().__repr__()

    def __len__(self):
        return len(self.__dict__)

def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

def generate_unique_filename(original_name: str, user_id: str = None) -> str:
    """生成全局唯一文件名
    
    参数：
        original_name: 原始文件名（用于保留扩展名）
        user_id: 用户ID（可选，增加哈希熵值）
        
    返回：
        [时间戳].[哈希基名].[UUID4].[扩展名]
    """
    # 提取扩展名（兼容隐藏文件）
    file_ext = original_name.split('.')[-1] if '.' in original_name else 'data'
    
    # 生成四元组唯一因子
    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")  # 包含微秒
    random_str = uuid.uuid4().hex[:6]  # 取UUID前6位
    system_pid = os.getpid()  # 进程ID
    network_addr = uuid.getnode()  # 网卡MAC地址哈希
    
    # 构建哈希种子
    hash_seed = f"{timestamp}-{random_str}-{system_pid}-{network_addr}"
    if user_id:
        hash_seed += f"-{user_id}"
        
    # 生成128位哈希基名
    hash_base = md5(hash_seed.encode()).hexdigest()[:8]  # 取前8位
    
    # 组合最终文件名
    return f"{timestamp}.{hash_base}.{uuid.uuid4().hex}.{file_ext}"