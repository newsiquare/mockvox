import uuid
from hashlib import md5
from datetime import datetime
import os

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