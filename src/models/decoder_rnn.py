import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=512, decoder_dim=512, encoder_dim=512, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embed_size = embed_size
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        # LSTM принимает (embed_size + encoder_dim) если конкатенируем глобальный фичер с токеном
        self.lstm = nn.LSTM(input_size=embed_size + encoder_dim, hidden_size=decoder_dim, batch_first=True)
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # инициализация h0 из feature
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # инициализация c0 из feature

    def init_hidden_state(self, encoder_global_feats):
        h = self.init_h(encoder_global_feats)  # [B, decoder_dim]
        c = self.init_c(encoder_global_feats)
        return h.unsqueeze(0), c.unsqueeze(0)  # LSTM expects (num_layers, batch, hidden)

    def forward(self, encoder_global_feats, captions, lengths):
        """
        encoder_global_feats: [B, encoder_dim]  (например global_feat из EncoderCNN)
        captions: Tensor [B, max_len] (индексы токенов, включая <bos> и <eos>)
        lengths: list/1D-tensor с длинами (включая bos/eos)
        Возвращает logits: PackedSequence -> распаковать, или распакованные logits [sum(lengths), vocab_size]
        """
        embeddings = self.embedding(captions)  # [B, max_len, embed_size]
        # расширяем encoder_global_feats в time dimension
        encoder_feats_exp = encoder_global_feats.unsqueeze(1).repeat(1, embeddings.size(1), 1)  # [B, max_len, encoder_dim]
        lstm_input = torch.cat([embeddings, encoder_feats_exp], dim=2)  # [B, max_len, embed + encoder_dim]

        h0, c0 = self.init_hidden_state(encoder_global_feats)  # (1, B, dec_dim)
        packed = nn.utils.rnn.pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed, (h0, c0))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)  # [B, T_max, dec_dim]
        logits = self.fc(self.dropout(outputs))  # [B, T_max, vocab_size]
        return logits

    def step(self, prev_word_idx, prev_hidden, encoder_global_feat):
        """
        один шаг генерации (для beam search)
        prev_word_idx: Tensor [beam_size] (индексы)
        prev_hidden: (h, c) each [1, beam_size, dec_dim]
        encoder_global_feat: Tensor [beam_size, encoder_dim]
        Возвращаем: log_probs [beam_size, vocab_size], new_hidden
        """
        emb = self.embedding(prev_word_idx).unsqueeze(1)  # [beam, 1, embed]
        enc = encoder_global_feat.unsqueeze(1)            # [beam, 1, enc_dim]
        inp = torch.cat([emb, enc], dim=2)                # [beam, 1, embed+enc]
        out, new_hidden = self.lstm(inp, prev_hidden)     # out: [beam, 1, dec_dim]
        logits = self.fc(out.squeeze(1))                  # [beam, vocab_size]
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs, new_hidden

    def beam_search(self, encoder_global_feat, vocab, beam_size=3, max_len=20, device='cpu'):
        """
        encoder_global_feat: [1, encoder_dim] single image
        vocab: Vocabulary instance (to map idx->word)
        Возвращает: (best_sequence_idxs, best_sequence_words, score)
        """
        k = beam_size
        encoder_global_feat = encoder_global_feat.to(device)

        # Инициализация
        vocab_bos = vocab.word_to_idx[vocab.bos_token]
        vocab_eos = vocab.word_to_idx[vocab.eos_token]

        # начальные гипотезы: (sequence, score, hidden)
        # подготовим начальное hidden для beam
        h, c = self.init_hidden_state(encoder_global_feat)  # (1, 1, dec_dim)
        # дублируем по beam
        h = h.repeat(1, k, 1)  # (1, k, dec_dim)
        c = c.repeat(1, k, 1)

        sequences = [[vocab_bos] for _ in range(k)]
        scores = torch.zeros(k, device=device)  # log-prob sums
        complete_seqs = []
        complete_scores = []

        # первый шаг: подаём <bos> всем k веткам
        prev_words = torch.tensor([vocab_bos] * k, dtype=torch.long, device=device)
        encoder_feats_k = encoder_global_feat.expand(k, -1)  # [k, enc_dim]

        for step in range(max_len):
            log_probs, (h, c) = self.step(prev_words, (h, c), encoder_feats_k)  # log_probs: [k, vocab]
            # для первого шага: если step==0, у нас k идентичных гипотез -> берем топ k слов из первой ветки
            if step == 0:
                top_log_probs, top_idx = log_probs[0].topk(k, dim=0)  # [k]
                scores = top_log_probs
                sequences = [[vocab_bos, int(idx)] for idx in top_idx]
                prev_words = top_idx.clone()
                # hidden: нужно индексировать h,c по выбранным индексам — но так как все ветки были одинаковы, дубли не нужны
                # h,c уже повторены
            else:
                # расширяем каждую гипотезу и оставляем топ k из k * V
                curr_scores = scores.unsqueeze(1) + log_probs  # [k, vocab]
                curr_scores_flat = curr_scores.view(-1)        # [k * vocab]
                top_scores, top_pos = curr_scores_flat.topk(k, dim=0)  # берём лучшие k
                # вычисляем откуда пришли:
                prev_hyp_idx = top_pos // log_probs.size(1)  # индекс ветки
                next_word_idx = top_pos % log_probs.size(1)

                # обновляем sequences и hidden
                new_sequences = []
                new_h = h[:, prev_hyp_idx, :].contiguous()  # (1, k, dec_dim)
                new_c = c[:, prev_hyp_idx, :].contiguous()
                for i in range(k):
                    seq = sequences[prev_hyp_idx[i].item()].copy()
                    seq.append(int(next_word_idx[i].item()))
                    new_sequences.append(seq)

                sequences = new_sequences
                scores = top_scores
                prev_words = next_word_idx.clone()
                h, c = new_h, new_c

            # проверяем завершённые последовательности (те, где последний токен == <eos>)
            incomplete_idxs = []
            for i, seq in enumerate(sequences):
                if seq[-1] == vocab_eos:
                    complete_seqs.append(seq)
                    complete_scores.append(scores[i].item())
                else:
                    incomplete_idxs.append(i)

            # если все завершены — останавливаемся
            if len(complete_seqs) >= k:
                break

            # если есть неполные — сжимаем до оставшихся k (иначе следующий шаг упадёт)
            if len(incomplete_idxs) > 0:
                keep_k = min(k, len(incomplete_idxs))
                # оставляем первые keep_k из incomplete_idxs
                sel = incomplete_idxs[:keep_k]
                sequences = [sequences[i] for i in sel]
                scores = scores[sel]
                prev_words = prev_words[sel]
                h = h[:, sel, :].contiguous()
                c = c[:, sel, :].contiguous()
                encoder_feats_k = encoder_global_feat.expand(len(sel), -1)

        # если не накопили завершённых, берём текущие лучшие
        if len(complete_seqs) == 0:
            complete_seqs = sequences
            complete_scores = scores.tolist()

        # выбираем лучшую по score (max log-prob)
        best_idx = int(torch.tensor(complete_scores).argmax().item())
        best_seq = complete_seqs[best_idx]

        # перевод в слова
        words = [vocab.idx_to_word.get(int(i), vocab.unk_token) for i in best_seq]
        return best_seq, words, complete_scores[best_idx]
