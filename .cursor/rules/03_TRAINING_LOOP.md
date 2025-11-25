# Обучение модели captioning

## Цикл обучения
1. Извлекаем батч
2. Encoder → features
3. Decoder → predictions
4. CrossEntropyLoss с паддингом
5. Backward
6. Optimizer.step()

---

## Early Stopping
Останавливает, если:
- val_loss ↑ 3 эпохи подряд
- BLEU-score не растёт

---

## Метрики
- BLEU-1/2/3/4
- CIDEr
- METEOR

---

## Логирование
- каждая эпоха сохраняет:
  - train_loss
  - val_loss
  - BLEU
  - время эпохи
- каждые N эпох — checkpoint

---

## Checkpoints
Хранят:
- веса encoder / decoder
- optimizer state
- epoch
- vocabulary
