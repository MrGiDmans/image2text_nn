# Архитектура captioning модели

## 1. EncoderCNN
Отвечает за извлечение признаков из изображения.  
Обычно используется фиксированный CNN:

- ResNet50 / ResNet101
- EfficientNet
- ConvNeXt

### Основные методы
- `forward(images)` → тензор признаков `B × D`
- `freeze()` — отключение градиентов
- `unfreeze()` — тонкая настройка

---

## 2. DecoderRNN
Вариант 1: LSTM  
Вариант 2: GRU  
Вариант 3: TransformerDecoder

Методы:
- `forward(features, captions)` — обучение
- `sample(features)` — инференс (greedy)
- `beam_search(features, k)` — beam search

---

## 3. CaptionModel
Комбинация:
```python
prediction = decoder( encoder(image) )

Интерфейс:
forward(images, captions)
generate(images)
save(path)
load(path)
