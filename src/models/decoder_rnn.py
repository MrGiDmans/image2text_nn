import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union

from utils.vocabulary import Vocabulary


class DecoderRNN(nn.Module):
    """
    Простая LSTM-декодер без внимания. Получает глобальные фичи энкодера и
    генерирует распределение по словарю на каждом шаге.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        decoder_dim: int = 512,
        encoder_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_size + encoder_dim,
            hidden_size=decoder_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_state(self, encoder_feats: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.init_h(encoder_feats)  # [B, dec_dim]
        c = self.init_c(encoder_feats)
        return h.unsqueeze(0), c.unsqueeze(0)

    def _concat_encoder_feats(self, encoder_feats: Tensor, steps: int) -> Tensor:
        """
        Повторяем глобальные фичи по временной оси для объединения со входом LSTM.
        """
        return encoder_feats[:, None, :].expand(-1, steps, -1)

    def forward(
        self,
        encoder_feats: Tensor,
        captions: Tensor,
        lengths: Union[Tensor, List[int]],
    ) -> Tensor:
        """
        Teacher-forcing проход.
        """
        embeddings = self.embedding(captions)
        _, T, _ = embeddings.size()
        enc_expanded = self._concat_encoder_feats(encoder_feats, T)
        lstm_input = torch.cat([embeddings, enc_expanded], dim=2)

        h0, c0 = self.init_hidden_state(encoder_feats)
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.cpu().tolist()

        packed = nn.utils.rnn.pack_padded_sequence(
            lstm_input,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.lstm(packed, (h0, c0))
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(self.dropout(out))
        return logits  # [B, Tmax, vocab]

    def step(
        self,
        prev_tokens: Tensor,
        prev_state: Tuple[Tensor, Tensor],
        encoder_feats: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Один шаг авто-регрессионного декодирования (используется в inference).
        """
        emb = self.embedding(prev_tokens).unsqueeze(1)
        enc = encoder_feats.unsqueeze(1)
        lstm_in = torch.cat([emb, enc], dim=2)

        output, state = self.lstm(lstm_in, prev_state)
        logits = self.fc(self.dropout(output.squeeze(1)))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, state

    @torch.no_grad()
    def beam_search(
        self,
        encoder_feat: Tensor,
        vocab: Vocabulary,
        beam_size: int = 3,
        max_len: int = 20,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Мини-beam search для одного изображения.
        """
        bos = vocab.bos_idx
        eos = vocab.eos_idx

        encoder_feat = encoder_feat.to(device)

        h, c = self.init_hidden_state(encoder_feat)  # (1,1,H)
        h = h.repeat(1, beam_size, 1)
        c = c.repeat(1, beam_size, 1)

        encoder_exp = encoder_feat.expand(beam_size, -1)

        beams = [
            [[bos], 0.0, h[:, i : i + 1, :], c[:, i : i + 1, :]]
            for i in range(beam_size)
        ]
        finished: list[tuple[list[int], float]] = []

        for _ in range(max_len):
            new_beams = []
            for seq, score, h_i, c_i in beams:
                if seq[-1] == eos:
                    finished.append((seq, score))
                    continue

                prev = torch.tensor([seq[-1]], device=device)
                log_probs, (h_new, c_new) = self.step(
                    prev, (h_i, c_i), encoder_exp[0:1]
                )
                top_log_probs, top_idx = log_probs.squeeze(0).topk(beam_size)

                for lp, idx in zip(top_log_probs, top_idx):
                    new_seq = seq + [int(idx)]
                    new_beams.append(
                        [new_seq, score + float(lp), h_new, c_new]
                    )

            if len(new_beams) == 0:
                break

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        if len(finished) == 0:
            finished = [(seq, score) for (seq, score, _, _) in beams]

        finished.sort(key=lambda x: x[1], reverse=True)
        best_seq, best_score = finished[0]
        best_words = [vocab.idx_to_word.get(i, vocab.unk_token) for i in best_seq]
        return best_seq, best_words, best_score
