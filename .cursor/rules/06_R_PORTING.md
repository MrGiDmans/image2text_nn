# Портирование модели PyTorch → R

## Подходы

### 1. Torch for R
Прямая реализация encoder/decoder слой-за-слоем.

### 2. Reticulate
Вызов Python модели из R.

### 3. ONNX → R
Экспорт PyTorch → ONNX → R ONNXRuntime.

---

## Рекомендации
- избегать нестандартных PyTorch операций;
- encoder + decoder разбивать на отдельные блоки;
- не смешивать Python-логику с обучением модели.

---

## Что нужно перенести
1. Веса (pt → rds или onnx)
2. Структуры словаря
3. Последовательный forward-pass
4. Генерацию caption (greedy/beam-search)

---

## Что переносить НЕ нужно
- загрузку данных
- аугментации
- collate_fn
- DataLoader
