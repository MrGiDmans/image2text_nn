import torch
from src.utils.vocabulary import Vocabulary

def collate_fn_for_token(batch, vocab: Vocabulary):
    images = []
    captions = []

    for image, caption in batch:
        images.append(image)

        # токенизация + преобразование в индексы
        token_ids = vocab.numericalize(caption)

        # BOS + токены + EOS
        numerical = [vocab.bos_idx] + token_ids + [vocab.eos_idx]
        captions.append(torch.tensor(numerical, dtype=torch.long))

    # Паддинг
    lengths = [len(c) for c in captions]
    max_len = max(lengths)

    padded = torch.full(
        (len(captions), max_len),
        vocab.pad_idx,
        dtype=torch.long
    )

    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    images = torch.stack(images)

    return images, padded, torch.tensor(lengths, dtype=torch.long)


