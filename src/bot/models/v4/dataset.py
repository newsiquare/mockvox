import random
from pathlib import Path
import math
import traceback
import json
from typing import List, Dict, Iterator
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from bot.text import Normalizer
from bot.utils import BotLogger, get_hparams_from_file, load_audio
from bot.nn import spectrogram_torch, mel_spectrogram_torch
from bot.config import SOVITS_MODEL_CONFIG

class Text2SemanticDataset(torch.utils.data.Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(self, hparams) -> None:
        """ 
        传入的超参数(hparams)中, 必须包含以下参数：
            semantic_path - name2text.json 文件的完整路径
            phoneme_path - name2semantic.json 文件的完整路径 
            bert_path - bert 子目录
            max_sample - 最大样本量. 建议 None
            max_sec - 最大长度(秒). 建议 100
            max_ps_ratio - max value of phoneme/sec. 建议25
            min_ps_ratio - min value of phoneme/sec. 建议3
            pad_value - 建议1024
        """
        super().__init__()
        self.hparams = hparams

        assert Path(self.hparams.semantic_path).exists()
        assert Path(self.hparams.phoneme_path).exists()

        try:
            with open(self.hparams.semantic_path, 'r', encoding='utf8') as f:
                self.phoneme_data = json.load(f)
            with open(self.hparams.phoneme_path, 'r', encoding='utf8') as f:
                self.semantic_data = json.load(f)
        except FileNotFoundError:
            BotLogger.error(f"语义文件不存在: {Path(self.hparams.semantic_path).name} or \
                音素文件不存在: {Path(self.hparams.phoneme_path).name}")

        hps = get_hparams_from_file(SOVITS_MODEL_CONFIG)
        self.hz = int(hps.model.semantic_frame_rate[:-2])        

        if self.hparams.max_sample is not None:
            self.semantic_data = self.semantic_data[:max_sample]
        
        self._init_batch()

    def _init_batch(self):
        self.semantic_phoneme = []
        self.item_names = []
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0

        max_sec = self.hparams.max_sec
        max_ps_ratio = self.hparams.max_ps_ratio
        min_ps_ratio = self.hparams.min_ps_ratio
        hz = self.hz

        phoneme_dict = {item["key"]: item for item in self.phoneme_data}

        for item_semantic in self.semantic_data:
            key = item_semantic['key']

            # 1. 检查key对齐
            if key not in phoneme_dict:
                num_not_in += 1
                continue

            item_phoneme = phoneme_dict[key]        
            semantic_ids = [int(x) for x in item_semantic["semantic"].split()]

            # 2. 检查音频时长
            if len(semantic_ids) > max_sec * hz:
                num_deleted_bigger += 1
                continue

            phonemes = item_phoneme["phones"].split()
            try:
                phoneme_ids = Normalizer.cleaned_text_to_sequence(phonemes)
            except Exception:
                num_not_in += 1
                continue

            # 3. 检查phoneme长度限制
            if len(phoneme_ids) > max_sec * hz / 2.5:
                num_deleted_ps += 1
                continue

            # 4. 检查phoneme/sec比例
            duration = len(semantic_ids) / hz
            if duration == 0:
                continue
            ps_ratio = len(phoneme_ids) / duration
            
            if not (min_ps_ratio <= ps_ratio <= max_ps_ratio):
                num_deleted_ps += 1
                continue

            # 保存有效数据
            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            self.item_names.append(key)

        # 数据增强：当有效数据不足时复制样本
        min_num = 100
        copies = 1
        current_len = len(self.semantic_phoneme)
        if current_len < min_num and current_len > 0:
            copies = max(2, min_num // current_len)
        self.semantic_phoneme = self.semantic_phoneme * copies
        self.item_names = self.item_names * copies

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        item_name = self.item_names[idx]
        phoneme_ids_len = len(phoneme_ids)
        # semantic tokens target
        semantic_ids_len = len(semantic_ids)

        flag = 0
        path_bert = Path(self.hparams.bert_path) / f"{item_name}.pt"
        if path_bert.exists():
            bert_feature = torch.load(path_bert, map_location="cpu")
            assert bert_feature.shape[-1] == len(phoneme_ids)
        else:
            bert_feature = None
        
        return {
            "idx": idx,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_ids_len,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_ids_len,
            "bert_feature": bert_feature,
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []
        # return

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        # pad 0
        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.hparams.pad_value)

        # # convert each batch to torch.tensor
        phoneme_ids = torch.tensor(phoneme_ids)
        semantic_ids = torch.tensor(semantic_ids)
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        for idx, item in enumerate(examples):
            bert = item["bert_feature"]
            if bert != None:
                bert_padded[idx, :, : bert.shape[-1]] = bert

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length)
            "phoneme_ids": phoneme_ids,
            # torch.Tensor (B)
            "phoneme_ids_len": phoneme_ids_lens,
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
            # torch.Tensor (B, 1024, max_phoneme_length)
            "bert_feature": bert_padded,
        }

def batch_sequences(sequences: List[np.array], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = (
            [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1)
        )
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch

class GPTBucketSampler(torch.utils.data.Sampler):
    """单卡桶式采样器，支持动态长度分桶和批次洗牌"""
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_width: float = 2.0,
        seed: int = 42
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_width = bucket_width

        # 获取样本长度并分桶
        self.sample_lengths = self._get_sample_lengths()
        self.buckets = self._create_buckets()

    def _get_sample_lengths(self) -> List[tuple]:
        """获取带索引的样本长度列表"""
        return [(i, self.dataset.get_sample_length(i)) for i in range(len(self.dataset))]

    def _create_buckets(self) -> List[List[int]]:
        """创建长度分桶"""
        # 按长度排序
        sorted_samples = sorted(self.sample_lengths, key=lambda x: x[1])
        
        # 动态分桶
        buckets = []
        current_bucket = []
        current_max = self.bucket_width
        
        for idx, length in sorted_samples:
            if length <= current_max:
                current_bucket.append(idx)
            else:
                buckets.append(current_bucket)
                current_bucket = [idx]
                current_max += self.bucket_width
        if current_bucket:
            buckets.append(current_bucket)
        return buckets

    def _generate_batches(self) -> List[List[int]]:
        """生成洗牌后的批次索引"""
        # 桶内洗牌
        rng = random.Random(self.seed)
        shuffled_buckets = [rng.sample(b, len(b)) for b in self.buckets]

        # 平铺所有样本
        all_indices = [idx for bucket in shuffled_buckets for idx in bucket]

        # 切分批次
        batch_indices = []
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i+self.batch_size]
            if not self.drop_last or len(batch) == self.batch_size:
                batch_indices.append(batch)

        # 批次级洗牌
        if self.shuffle:
            rng.shuffle(batch_indices)
        return batch_indices

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._generate_batches()

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def set_epoch(self, epoch: int):
        """设置随机种子 (保持API兼容性)"""
        self.seed = epoch

class TextAudioSpeakerDataset(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """
    def __init__(self, hparams, val=False):
        processed_dir = hparams.processed_dir
        self.n2t = Path(processed_dir) / 'name2text.json'
        self.cnhubert = Path(processed_dir) / 'cnhubert'
        self.wav32k = Path(processed_dir) / 'wav32k'

        # 验证路径
        assert self.cnhubert.is_dir(), f"Directory required: {self.cnhubert}"
        assert self.wav32k.is_dir(), f"Directory required: {self.wav32k}"
        assert self.n2t.exists(), f"File {self.n2t} not found"

        with open(self.n2t, 'r', encoding='utf8') as f:
            n2t_data = json.load(f)
        
        self.phoneme_data = {item["key"]: [item["phones"]] for item in n2t_data}
        names4 = {f.stem for f in self.cnhubert.glob("*.pt")}     # 去除.pt后缀
        names5 = {f.stem for f in self.wav32k.glob("*.wav")}      # 去除.wav后缀

        valid_keys = set(self.phoneme_data.keys()) & names4 & names5
        self.audiopaths_sid_text = list(valid_keys)

        # 不足100条数据，填充
        min_num = 100
        data_len = max(1, len(self.audiopaths_sid_text))
        repeat_times = max(2, (min_num + data_len - 1) // data_len)
        self.audiopaths_sid_text = (self.audiopaths_sid_text * repeat_times)
        
        # 音频参数设置
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.n_mel_channels = hparams.n_mel_channels
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax
        self.val = val

        self.spec_min = -12
        self.spec_max = 2

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        audiopaths_sid_text_new = []
        lengths = []
        skipped_phone = 0
        skipped_dur = 0

        for audiopath in self.audiopaths_sid_text:
            # 音素数据处理
            try:
                phone_str = self.phoneme_data[audiopath][0]  # 获取phones字段
                phonemes = phone_str.strip().split()
                phoneme_ids = Normalizer.cleaned_text_to_sequence(phonemes)
            except KeyError:
                BotLogger.warn(f"{audiopath} not in phoneme_data!")
                skipped_phone += 1
                continue
            except Exception as e:
                BotLogger.warn(f"Phoneme processing error for {audiopath}: {str(e)}")
                skipped_phone += 1
                continue

            # 音频文件验证
            wav_path = self.wav32k / f"{audiopath}.wav"
            try:
                size = wav_path.stat().st_size
                duration = size / (self.sampling_rate * 2)  # 16-bit mono假设
            except FileNotFoundError:
                BotLogger.warn(f"Audio file missing: {wav_path}")
                skipped_dur += 1
                continue

            # 时长过滤逻辑
            if (0.6 < duration < 54) or self.val:
                audiopaths_sid_text_new.append([audiopath, phoneme_ids])
                lengths.append(int(size // (2 * self.hop_length)))
            else:
                skipped_dur += 1

        assert len(audiopaths_sid_text_new) > 1, "Insufficient valid data"

        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, phoneme_ids = audiopath_sid_text
        text = torch.FloatTensor(phoneme_ids)
        try:
            spec, mel = self.get_audio(audiopath)
            with torch.no_grad():
                ssl = torch.load(self.cnhubert / f"{audiopath}.pt", map_location="cpu")
                # 更鲁棒的特征对齐逻辑
                if ssl.shape[-1] < spec.shape[-1]:
                    pad_length = spec.shape[-1] - ssl.shape[-1]
                    ssl = F.pad(ssl, (0, pad_length), mode="constant")
                elif ssl.shape[-1] > spec.shape[-1]:
                    ssl = ssl[:, :, :spec.shape[-1]]
                # if (ssl.shape[-1] != spec.shape[-1]):
                #     ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(ssl.dtype)
                ssl.requires_grad = False
        except:
            traceback.print_exc()
            spec = torch.zeros(1025, 96)
            mel = torch.zeros(100, 192)
            ssl = torch.zeros(1, 768, 96)
            text = text[-1:]
            BotLogger.warn("load audio or ssl error!", self.cnhubert, audiopath)
        return (ssl, spec, mel, text)

    def get_audio(self, filename):
        file_path = self.wav32k / f"{filename}.wav"
        audio_array = load_audio(file_path, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
        audio = torch.FloatTensor(audio_array)  # /32768
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,  # n_fft=2048
            self.hop_length,     # hop_size=640
            self.win_length,     # win_length=2048
            center=False
        )
        spec = torch.squeeze(spec, 0)
        mel = mel_spectrogram_torch(
            audio_norm, 
            self.filter_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            self.mel_fmin,
            self.mel_fmax
        )
        mel = self.norm_spec(torch.squeeze(mel, 0))
        return spec, mel

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def validate_dataset(self):
        """校验数据集完整性"""
        missing = []
        for path in self.audiopaths_sid_text:
            if not (self.cnhubert/f"{path[0]}.pt").exists():
                missing.append(f"pt: {path[0]}")
            if not (self.wav32k/f"{path[0]}.wav").exists():
                missing.append(f"wav: {path[0]}")
        return missing

class TextAudioSpeakerCollate:
    """ Zero-pads model inputs and targets
    """
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
    
    def __call__(self, batch):
        # 过滤空批次
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        # 按频谱时间步降序排序
        sorted_indices = sorted(
            range(len(batch)),
            key=lambda x: batch[x][1].size(1),
            reverse=True
        )
        sorted_batch = [batch[i] for i in sorted_indices]

        # 动态获取特征维度
        ssl_feat_dim = sorted_batch[0][0].size(1)
        spec_feat_dim = sorted_batch[0][1].size(0)
        mel_feat_dim = sorted_batch[0][2].size(0)

        # 检查维度一致性
        for x in sorted_batch:
            assert x[0].size(1) == ssl_feat_dim, \
                f"SSL特征维度不一致! 样本维度: {x[0].size()}"
            assert x[1].size(0) == spec_feat_dim, \
                f"频谱特征维度不一致! 样本维度: {x[1].size()}"

        # 计算各特征最大时间步（保持偶数）
        def get_even_max(feature_idx, time_dim):
            max_len = max(x[feature_idx].size(time_dim) for x in sorted_batch)
            return max_len if max_len % 2 ==0 else max_len+1

        max_ssl_time = get_even_max(0, 2)   # x[0] 是 SSL，时间步在 dim2
        max_spec_time = get_even_max(1, 1)  # x[1] 是频谱，时间步在 dim1
        max_mel_frame = get_even_max(2, 1)  # x[2] 是mel谱，时间步在 dim1
        max_text_len = max(x[3].size(0) for x in sorted_batch)

        # 初始化填充容器
        ssl_padded = torch.zeros(
            len(sorted_batch), 1, ssl_feat_dim, max_ssl_time,
            device=self.device
        )
        spec_padded = torch.zeros(
            len(sorted_batch), spec_feat_dim, max_spec_time,
            device=self.device
        )
        mel_padded = torch.zeros(
            len(sorted_batch), mel_feat_dim, max_mel_frame,
            device=self.device
        )
        text_padded = torch.zeros(
            len(sorted_batch), max_text_len,
            dtype=torch.long, device=self.device
        )

        # 填充数据
        for i, (ssl, spec, mel, text) in enumerate(sorted_batch):
            # SSL: (1, D, T) -> (1, D, max_T)
            ssl_padded[i, :, :, :ssl.size(2)] = ssl
            
            # Spec: (F, T) -> (F, max_T)
            spec_padded[i, :, :spec.size(1)] = spec
            
            # Mel: (1, L) -> (1, max_L)
            mel_padded[i, :, :mel.size(1)] = mel
            
            # Text: (N,) -> (max_N)
            text_padded[i, :text.size(0)] = text

        # 获取实际长度
        lengths = torch.tensor([
            [x[0].size(2), x[1].size(1), x[2].size(1), x[3].size(0)]
            for x in sorted_batch
        ], device=self.device).T

        return (
            ssl_padded.squeeze(1),  # 移除冗余的通道维度 (B, D, T)
            lengths[0],
            spec_padded,
            lengths[1],
            mel_padded,
            lengths[2],
            text_padded,
            lengths[3]
        )

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

class SoVITsBucketSampler(torch.utils.data.Sampler):
    """单机版分桶采样器（无分布式逻辑）"""
    def __init__(self, dataset, batch_size, boundaries, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lengths = dataset.lengths
        self.boundaries = boundaries
        
        # 创建分桶
        self.buckets = self._create_buckets()
        self._calc_bucket_counts()

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries)-1)]
        for idx, length in enumerate(self.lengths):
            bid = self._bisect(length)
            if bid != -1:
                buckets[bid].append(idx)
        
        # 过滤空桶并调整边界
        valid_buckets = []
        new_boundaries = [self.boundaries[0]]
        for i, bucket in enumerate(buckets):
            if len(bucket) > 0:
                valid_buckets.append(bucket)
                new_boundaries.append(self.boundaries[i+1])
        self.boundaries = new_boundaries
        return valid_buckets

    def _bisect(self, x):
        lo, hi = 0, len(self.boundaries)-1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                hi = mid
            else:
                lo = mid + 1
        return -1

    def _calc_bucket_counts(self):
        self.bucket_counts = [
            (len(bucket) + self.batch_size - 1) // self.batch_size 
            for bucket in self.buckets
        ]
        self.total_batches = sum(self.bucket_counts)

    def __iter__(self):
        if self.shuffle:
            # 每个epoch重新打乱桶内顺序和桶的顺序
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            
            # 先打乱桶的顺序
            bucket_order = torch.randperm(len(self.buckets), generator=g).tolist()
            
            all_batches = []
            for bucket_id in bucket_order:
                bucket = self.buckets[bucket_id]
                # 打乱桶内样本顺序
                indices = torch.randperm(len(bucket), generator=g).tolist()
                # 生成完整批次
                batches = [
                    [bucket[i] for i in indices[i*self.batch_size : (i+1)*self.batch_size]]
                    for i in range((len(indices) + self.batch_size - 1) // self.batch_size)
                ]
                all_batches.extend(batches)
            
            # 最后打乱所有批次顺序
            batch_order = torch.randperm(len(all_batches), generator=g).tolist()
            yield from [all_batches[i] for i in batch_order]
        else:
            # 顺序生成
            for bucket in self.buckets:
                for i in range(0, len(bucket), self.batch_size):
                    yield bucket[i:i+self.batch_size]

    def __len__(self):
        return self.total_batches

if __name__ == '__main__':
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='processed file name.')
    args = parser.parse_args()

    from bot.config import SOVITS_MODEL_CONFIG, PROCESS_PATH
    from torch.utils.data import DataLoader
    from bot.utils import get_hparams_from_file, HParams

    hps = get_hparams_from_file(SOVITS_MODEL_CONFIG)
    processed_path = Path(PROCESS_PATH) / args.file
    hps.data.processed_dir = processed_path

    print(f"Test SoVITs training Dataset --------------------------------------------------")
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TextAudioSpeakerDataset(hps.data)
    collate_fn = TextAudioSpeakerCollate(_device)
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=4)
    for batch_idx, (
        ssl,
        ssl_lengths,
        spec,
        spec_lengths,
        mel,
        mel_lengths,
        text,
        text_lengths       
    ) in enumerate(dataloader):
        print("spec.shape:", spec.shape)
        print("spec_lengths:", spec_lengths)
        print("mel.shape:", mel.shape)
        print("mel_lengths:", mel_lengths)
        break

    print(f"Test GPT training Dataset --------------------------------------------------")
    hps = HParams()
    hps.phoneme_path = processed_path / 'text2semantic.json'
    hps.semantic_path = processed_path / 'name2text.json'
    hps.bert_path = processed_path / 'bert'
    hps.max_sample = None
    hps.max_sec = 100
    hps.max_ps_ratio = 25
    hps.min_ps_ratio = 3
    hps.pad_value = 1024

    dataset_GPT = Text2SemanticDataset(hparams=hps)
    sampler = GPTBucketSampler(
        dataset_GPT,
        batch_size=4,
        shuffle=True,
        bucket_width=2.0
    )
    dataloader_GPT = DataLoader(
        dataset_GPT,
        num_workers=4,
        shuffle=False,
        collate_fn=dataset_GPT.collate,
        batch_sampler=sampler,
        persistent_workers=False,
        prefetch_factor=None
    )
    for idx, batch in enumerate(dataloader_GPT):
        print("batch : ", batch)
        break