import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from typing import cast

from utils.vocabulary import Vocabulary

class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, embed_size=512, decoder_dim=512,
                 encoder_dim=512, dropout=0.5):
        """
        DecoderRNN is a recurrent neural network decoder for image captioning.
        It generates captions word-by-word based on encoded image features.
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimensionality of the word embeddings.
            decoder_dim (int): Dimensionality of the decoder's LSTM hidden state.
            encoder_dim (int): Dimensionality of the encoder's output features.
            dropout (float): Dropout rate.
        Methods:
            forward(encoder_feats, captions, lengths): Forward pass for training.
            beam_search(encoder_feat, vocab, beam_size, max_len, device): Generate captions using beam search.
        Returns:
            During training, returns logits for each word in the caption.
        """
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
            batch_first=True
        )

        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_state(self, encoder_feats: Tensor):
        h: Tensor = self.init_h(encoder_feats)  # [B, dec_dim]
        c: Tensor = self.init_c(encoder_feats)
        return h.unsqueeze(0), c.unsqueeze(0)

    def forward(self, encoder_feats: Tensor, captions: Tensor, lengths: Tensor):
        embeddings = self.embedding(captions)

        B, T, _ = embeddings.size()

        # расширяем encoder_feats без создания лишней памяти
        enc_exp = encoder_feats[:, None, :].expand(B, T, self.encoder_dim)

        lstm_input = torch.cat([embeddings, enc_exp], dim=2)

        h0, c0 = self.init_hidden_state(encoder_feats)

        packed = nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed, (h0, c0))

        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        return self.fc(self.dropout(out))  # [B, Tmax, vocab]
    
    def step(self, prev_words: Tensor, prev_hidden: Tuple[Tensor, Tensor], encoder_global_feats: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Выполняет один шаг декодирования для батча.

        Args:
            prev_words: [B] - индексы слов, сгенерированных на предыдущем шаге (или BOS).
            prev_hidden: (h, c) - предыдущие скрытые состояния, h,c: [1, B, dec_dim].
            encoder_global_feats: [B, enc_dim] - глобальные признаки энкодера.

        Returns:
            log_probs: [B, vocab_size] - лог-вероятности следующего слова.
            hidden: (h_new, c_new) - новые скрытые состояния, h,c: [1, B, dec_dim].
        """
        B = prev_words.size(0)

        # 1. Embeddings
        # [B] -> [B, embed_size]
        embeddings = self.embedding(prev_words) 
        
        # 2. Подготовка входных данных для LSTM (Concatenate)
        # Вход для forward: [B, T, E + F]. Для step нужно [1, B, E + F]
        
        # Расширяем глобальные фичи до размера батча
        # [B, enc_dim]
        
        # Создаем входной вектор для LSTM на этом шаге: [B, embed_size + enc_dim]
        # Обратите внимание: в вашем forward вы конкатенируете эмбеддинги и enc_exp
        lstm_input_concat = torch.cat([embeddings, encoder_global_feats], dim=1)
        
        # LSTM ожидает вход: [Sequence_len, Batch, Feature_size] или [Batch, Sequence_len, Feature_size]
        # Используем Batch-first: [B, 1, Feature_size]
        lstm_input = lstm_input_concat.unsqueeze(1) # [B, 1, embed_size + encoder_dim]

        # 3. Проход через LSTM
        # prev_hidden: (h, c), где h,c: [1, B, dec_dim] (соответствует (num_layers, B, dec_dim))
        # lstm_out: [B, 1, dec_dim]
        lstm_out, hidden_new = self.lstm(lstm_input, prev_hidden) 
        
        # 4. Прогноз (Logits)
        # out: [B, dec_dim]
        out = lstm_out.squeeze(1)
        
        # Dropout перед финальным полносвязным слоем
        logits = self.fc(self.dropout(out)) # [B, vocab_size]
        
        # 5. Лог-вероятности (требуется для вычисления score в generate)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs, hidden_new

    @torch.no_grad()
    def beam_search(self, encoder_feat: Tensor, vocab: Vocabulary,
                    beam_size=3, max_len=20, device='cpu'):

        bos = vocab.bos_idx
        eos = vocab.eos_idx

        encoder_feat = encoder_feat.to(device)

        # init LSTM hidden state
        h, c = self.init_hidden_state(encoder_feat)  # (1,1,H)

        # расширяем по beam_size
        h = h.repeat(1, beam_size, 1)
        c = c.repeat(1, beam_size, 1)

        encoder_exp = encoder_feat.expand(beam_size, -1)

        # beams: each = (sequence, score, h, c)
        beams = [[ [bos], 0.0, h[:, i:i+1, :], c[:, i:i+1, :] ]
                 for i in range(beam_size)]

        finished = []

        for _ in range(max_len):

            new_beams = []

            for seq, score, h_i, c_i in beams:

                # если EOS — переносим в finished
                if seq[-1] == eos:
                    finished.append((seq, score))
                    continue

                prev = torch.tensor([seq[-1]], device=device)

                emb = self.embedding(prev).unsqueeze(1)  # [1,1,E]
                enc = encoder_exp[0:1].unsqueeze(1)      # [1,1,F]
                lstm_in = torch.cat([emb, enc], dim=2)

                out, (h_new, c_new) = self.lstm(lstm_in, (h_i, c_i))
                logits = self.fc(out.squeeze(1))
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)

                top_log_probs, top_idx = log_probs.topk(beam_size)

                for lp, idx in zip(top_log_probs, top_idx):
                    new_seq = seq + [int(idx)]
                    new_beams.append([
                        new_seq,
                        score + float(lp),
                        h_new,
                        c_new
                    ])

            # если новые ветки закончились
            if len(new_beams) == 0:
                break

            # оставляем top-k
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        # если нет завершённых гипотез — берём текущие лучшие
        if len(finished) == 0:
            finished = [(seq, score) for (seq, score, _, _) in beams]

        finished.sort(key=lambda x: x[1], reverse=True)

        best_seq, best_score = finished[0]
        best_words = [vocab.idx_to_word.get(i, vocab.unk_token)
                      for i in best_seq]

        return best_seq, best_words, best_score
