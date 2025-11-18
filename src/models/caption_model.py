import torch.nn as nn

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from models import (EncoderCNN, DecoderRNN)
from utils.vocabulary import Vocabulary

class CaptioningModel(nn.Module):
    """
    Обёртка Encoder + Decoder.
    forward(...) -> logits для обучения.
    generate(...) -> генерирует подпись (greedy или beam).
    """
    def __init__(
        self,
        encoder: EncoderCNN,
        decoder: DecoderRNN,
        vocab: Vocabulary,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab

        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(self, images: torch.Tensor, captions: torch.Tensor, lengths: List[int]):
        """
        Полный проход для обучения.

        Args:
            images: [B, 3, H, W]
            captions: [B, T_max] (индексы, уже с BOS и EOS, padded)
            lengths: list/1D-tensor len B (каждая длина включает BOS/EOS)

        Возвращает:
            logits: [B, T_max, vocab_size] (можно использовать nn.CrossEntropyLoss с игнорированием pad)
        """
        # Пропускаем изображения через энкодер
        # encoder возвращает (features_spatial, global_feat)
        features, global_feat = self.encoder(images)  # features: [B, num_pixels, enc_dim], global_feat: [B, enc_dim]

        # Decoder требует глобальные фичи (encoder_global_feats) и captions+lengths
        logits = self.decoder(global_feat, captions, lengths)  # [B, T_max, vocab_size]
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 20,
        beam_size: int = 3,
        mode: str = "greedy"  # or "beam"
    ) -> List[Tuple[List[int], List[str], float]]:
        """
        Генерация подписи(ей) для батча изображений.

        Возвращает список длины B: (sequence_indexes, sequence_words, score)
        Для beam режим поддерживается только batch_size == 1 (пока).
        Greedy режим поддерживает batch > 1.
        """
        self.eval()
        images = images.to(self.device)

        features, global_feat = self.encoder(images)  # features ignored for simple DecoderRNN, но может быть полезен
        B = images.size(0)

        results = []

        if mode == "beam":
            # текущая реализация beam в DecoderRNN реализована для одного изображения
            if B != 1:
                raise ValueError("beam mode currently supports batch_size == 1. Для батча используйте цикл по изображениям.")
            encoder_global = global_feat[0:1]  # shape [1, enc_dim]
            seq_idx, seq_words, score = self.decoder.beam_search(encoder_global, self.vocab, beam_size=beam_size, max_len=max_len, device=self.device)
            results.append((seq_idx, seq_words, score))
            return results

        # ---------- greedy decode (поддерживает batch > 1) ----------
        # Инициализируем скрытые состояния из global_feat
        h, c = self.decoder.init_hidden_state(global_feat)  # h,c: (1, B, dec_dim)

        # начальные слова = BOS для каждой выборки
        prev_words = torch.full((B,), fill_value=self.vocab.bos_idx, dtype=torch.long, device=self.device)

        sequences_idx = [[self.vocab.bos_idx] for _ in range(B)]
        sequence_scores = torch.zeros(B, device=self.device)
        finished = [False] * B

        for t in range(max_len):
            # шаг декодера: используем decoder.step для одного шага
            # decoder.step ожидает prev_word_idx: [batch] и prev_hidden (h,c) формы (1, batch, hidden)
            log_probs, (h, c) = self.decoder.step(prev_words, (h, c), global_feat)
            # log_probs: [B, vocab_size]
            top1 = log_probs.argmax(dim=1)  # [B]
            top1_scores = log_probs.gather(1, top1.unsqueeze(1)).squeeze(1)  # [B] лог-пробы выбранных слов

            for i in range(B):
                if finished[i]:
                    continue
                token = int(top1[i].item())
                sequences_idx[i].append(token)
                sequence_scores[i] += float(top1_scores[i].item())
                if token == self.vocab.eos_idx:
                    finished[i] = True

            prev_words = top1
            if all(finished):
                break

        # Преобразуем в слова
        for i in range(B):
            idxs = sequences_idx[i]
            words = [self.vocab.idx_to_word.get(int(idx), self.vocab.unk_token) for idx in idxs]
            score = float(sequence_scores[i].item())
            results.append((idxs, words, score))

        return results
