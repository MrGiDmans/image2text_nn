import json
import re
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold

        # служебные токены
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        # фиксированное назначение индексов
        self.word_to_idx = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        self.idx_to_word = {idx: w for w, idx in self.word_to_idx.items()}

        # ДОБАВЛЕНО: удобные прямые ссылки
        self.pad_idx = self.word_to_idx[self.pad_token]
        self.bos_idx = self.word_to_idx[self.bos_token]
        self.eos_idx = self.word_to_idx[self.eos_token]
        self.unk_idx = self.word_to_idx[self.unk_token]

        self.word_freq = Counter()

    def __len__(self):
        return len(self.word_to_idx)

    def _basic_tokenize(self, text: str):
        text = text.lower().strip()
        text = re.sub(r'\s+\.', '.', text)
        return text.split()

    def build_vocabulary(self, captions_list):
        for caption in captions_list:
            tokens = self._basic_tokenize(str(caption))
            self.word_freq.update(tokens)

        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word

        # ОБНОВЛЯЕМ спец-индексы после построения словаря
        self.pad_idx = self.word_to_idx[self.pad_token]
        self.bos_idx = self.word_to_idx[self.bos_token]
        self.eos_idx = self.word_to_idx[self.eos_token]
        self.unk_idx = self.word_to_idx[self.unk_token]

    def numericalize(self, caption: str):
        tokens = self._basic_tokenize(str(caption))
        return [self.word_to_idx.get(t, self.unk_idx) for t in tokens]

    def decode_indexes(self, idxs):
        return [self.idx_to_word.get(int(i), self.unk_token) for i in idxs]

    def save_vocab(self, path: str):
        payload = {
            "word_to_idx": self.word_to_idx,
            "freq_threshold": self.freq_threshold
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        obj = cls(freq_threshold=payload.get("freq_threshold", 5))
        obj.word_to_idx = payload["word_to_idx"]
        obj.idx_to_word = {int(idx): w for w, idx in obj.word_to_idx.items()}

        # восстановить индексы спец-токенов
        obj.pad_idx = obj.word_to_idx[obj.pad_token]
        obj.bos_idx = obj.word_to_idx[obj.bos_token]
        obj.eos_idx = obj.word_to_idx[obj.eos_token]
        obj.unk_idx = obj.word_to_idx[obj.unk_token]

        return obj

