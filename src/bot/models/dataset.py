import random
from pathlib import Path
import traceback
import json
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from bot.core import load_audio
from bot.text import Normalizer
from bot.utils import BotLogger
from bot.models import spectrogram_torch

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
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
        self.val = val

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

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, phoneme_ids = audiopath_sid_text
        text = torch.FloatTensor(phoneme_ids)
        try:
            spec, wav = self.get_audio(audiopath)
            with torch.no_grad():
                ssl = torch.load(self.cnhubert / f"{audiopath}.pt", map_location="cpu")
                if (ssl.shape[-1] != spec.shape[-1]):
                    ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(ssl.dtype)
                ssl.requires_grad = False
        except:
            traceback.print_exc()
            spec = torch.zeros(1025, 100)
            wav = torch.zeros(1, 100 * self.hop_length)
            ssl = torch.zeros(1, 768, 100)
            text = text[-1:]
            BotLogger.warn("load audio or ssl error!", self.processed_dir, audiopath)
        return (ssl, spec, wav, text)

    def get_audio(self, filename):
        file_path = self.wav32k / f"{filename}.wav"
        audio_array = load_audio(file_path, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
        audio = torch.FloatTensor(audio_array)  # /32768
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length)
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def random_slice(self, ssl, wav, mel):
        assert abs(ssl.shape[-1] - wav.shape[-1] // self.hop_length) < 3, ( \
            "first", ssl.shape, wav.shape)

        len_mel = mel.shape[1]
        if self.val:
            reference_mel = mel[:, :len_mel // 3]
            return reference_mel, ssl, wav, mel
        r_input = random.randint(0, 1)
        sep_point = random.randint(int(len_mel // 3), int(len_mel // 3 * 2))

        if r_input == 0:
            reference_mel = mel[:, :sep_point]
            ssl = ssl[:, :, sep_point:]
            wav2 = wav[:, sep_point * self.hop_length:]
            mel = mel[:, sep_point:]
        else:
            reference_mel = mel[:, sep_point:]
            ssl = ssl[:, :, :sep_point]
            wav2 = wav[:, :sep_point * self.hop_length]
            mel = mel[:, :sep_point]

        assert abs(ssl.shape[-1] - wav2.shape[-1] // self.hop_length) < 3, ( \
            ssl.shape, wav.shape, wav2.shape, mel.shape, sep_point, self.hop_length, sep_point * self.hop_length, r_input)
        return reference_mel, ssl, wav2, mel

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
        # 按频谱长度降序排序
        sorted_indices = sorted(
            range(len(batch)), 
            key=lambda x: batch[x][1].size(1), 
            reverse=True
        )
        sorted_batch = [batch[i] for i in sorted_indices]

        # 计算最大长度（保持偶数）
        def get_even_max(dim):
            max_len = max(x[dim].size(1) for x in sorted_batch)
            return max_len if max_len % 2 == 0 else max_len + 1
            
        max_ssl_len = get_even_max(0)
        max_spec_len = get_even_max(1)
        max_wav_len = max(x[2].size(1) for x in sorted_batch)
        max_text_len = max(x[3].size(0) for x in sorted_batch)

        # 使用pad_sequence进行向量化填充
        ssl_padded = pad_sequence(
            [x[0].squeeze(0).T for x in sorted_batch],
            batch_first=True,
            padding_value=0
        ).transpose(1, 2).to(self.device)

        spec_padded = pad_sequence(
            [x[1] for x in sorted_batch],
            batch_first=True,
            padding_value=0
        ).to(self.device)

        wav_padded = pad_sequence(
            [x[2] for x in sorted_batch],
            batch_first=True,
            padding_value=0
        ).to(self.device)

        text_padded = pad_sequence(
            [x[3] for x in sorted_batch],
            batch_first=True,
            padding_value=0
        ).to(self.device)

        # 获取实际长度
        lengths = torch.tensor([
            [x[0].size(2), x[1].size(1), x[2].size(1), x[3].size(0)]
            for x in sorted_batch
        ], device=self.device).T

        return (
            ssl_padded, lengths[0],
            spec_padded, lengths[1],
            wav_padded, lengths[2],
            text_padded, lengths[3]
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

class BucketSampler(torch.utils.data.Sampler):
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
    from bot.config import MODEL_CONFIG_FILE, PROCESS_PATH
    from torch.utils.data import DataLoader
    from bot.utils import get_hparams_from_file

    hps = get_hparams_from_file(MODEL_CONFIG_FILE)
    processed_path = Path(PROCESS_PATH) / "20250416212521743916.69ba5a80.e47c25863b0e4d11831e218672ae51c2"
    hps.data.processed_dir = processed_path

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TextAudioSpeakerLoader(hps.data)
    collate_fn = TextAudioSpeakerCollate(_device)
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=4, pin_memory=True)
    for batch_idx, (
        ssl,
        ssl_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        text,
        text_lengths       
    ) in enumerate(dataloader):
        print("spec.shape:", spec.shape)
        print("spec_lengths:", spec_lengths) 