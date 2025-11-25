# Работа с данными: Dataset, DataLoader, Vocabulary

## Vocabulary
- создает маппинг word → index
- поддерживает:
  - <PAD>
  - <SOS>
  - <EOS>
  - <UNK>

Методы:
- `build_vocabulary(captions_list, threshold)`
- `numericalize(text)`
- `decode(indices)`

---

## Transforms
- `resize`
- `normalize`
- `ToTensor`
- аугментации:
  - RandomCrop
  - HorizontalFlip
  - ColorJitter

---

## Dataset
Хранит:
- изображения
- подписи
- словарь

Методы:
- `__getitem__(index)`
- `load_image(path)`
- `load_caption(idx)`

Вывод:
- image_tensor
- caption_tensor
- image_id

---

## Collate Function
Создает batch:
- паддинг caption-последовательностей
- сортировка по длине (опционально)

---

## DataLoader
- управляет num_workers
- возвращает батчи вида:
images: B,3,H,W
captions: B,T
lengths: B