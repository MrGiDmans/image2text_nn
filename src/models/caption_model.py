import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple, Union

from models import DecoderRNN, EncoderCNN
from utils.vocabulary import Vocabulary


class CaptioningModel(nn.Module):
    """
    Обёртка над EncoderCNN + DecoderRNN.
    """

    def __init__(
        self,
        encoder: EncoderCNN,
        decoder: DecoderRNN,
        vocab: Vocabulary,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab

        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(
        self,
        images: Tensor,
        captions: Tensor,
        lengths: Union[List[int], Tensor],
    ) -> Tensor:
        _, global_feat = self.encoder(images)
        logits = self.decoder(global_feat, captions, lengths)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        max_len: int = 20,
        beam_size: int = 3,
        mode: str = "greedy",
    ) -> List[Tuple[List[int], List[str], float]]:
        self.eval()
        images = images.to(self.device)

        _, global_feat = self.encoder(images)
        batch_size = images.size(0)
        results: List[Tuple[List[int], List[str], float]] = []

        if mode == "beam":
            if batch_size != 1:
                raise ValueError(
                    "beam mode поддерживает batch_size == 1. Используйте цикл."
                )
            encoder_global = global_feat[0:1]
            seq_idx, seq_words, score = self.decoder.beam_search(
                encoder_global,
                self.vocab,
                beam_size=beam_size,
                max_len=max_len,
                device=self.device,
            )
            results.append((seq_idx, seq_words, score))
            return results

        h, c = self.decoder.init_hidden_state(global_feat)
        prev_words = torch.full(
            (batch_size,),
            fill_value=self.vocab.bos_idx,
            dtype=torch.long,
            device=self.device,
        )

        sequences_idx = [[self.vocab.bos_idx] for _ in range(batch_size)]
        sequence_scores = torch.zeros(batch_size, device=self.device)
        finished = [False] * batch_size

        for _ in range(max_len):
            log_probs, (h, c) = self.decoder.step(prev_words, (h, c), global_feat)
            top1 = log_probs.argmax(dim=1)
            top1_scores = log_probs.gather(1, top1.unsqueeze(1)).squeeze(1)

            for i in range(batch_size):
                if finished[i]:
                    continue
                token = int(top1[i])
                sequences_idx[i].append(token)
                sequence_scores[i] += float(top1_scores[i])
                if token == self.vocab.eos_idx:
                    finished[i] = True

            prev_words = top1
            if all(finished):
                break

        for i in range(batch_size):
            idxs = sequences_idx[i]
            words = [
                self.vocab.idx_to_word.get(int(idx), self.vocab.unk_token)
                for idx in idxs
            ]
            results.append((idxs, words, float(sequence_scores[i].item())))

        return results
