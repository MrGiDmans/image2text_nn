from typing import Tuple, Type
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
import os
import sys

from utils.transforms import create_training_transforms
from utils.collate_fn import collate_fn_for_token
from utils.vocabulary import Vocabulary


class DataLoaderConfig:
    """
    ⚙️ Конфигурация, инкапсулирующая все параметры для инициализации
    Vocabulary и создания DataLoader'ов.

    Класс предназначен для обеспечения модульности и легкой настройки
    конвейера данных (Data Pipeline).

    Attributes:
        dataset_cls (Type[Dataset]): Класс датасета, который будет инстанцирован.
                                     ВАЖНО: Класс должен иметь статический метод 
                                     `get_all_captions(images_dir, caption_file)` 
                                     для извлечения списка подписей.
        
        # --- Параметры путей и данных ---
        images_dir (str): Путь к корневой директории с файлами изображений.
        caption_file (str): Путь к CSV-файлу, содержащему подписи к изображениям.
        vocab_path (str): Путь для сохранения/загрузки сериализованного объекта Vocabulary.
        
        # --- Параметры словаря (Vocabulary) ---
        freq_threshold (int): Минимальная частота появления слова для его включения в словарь.
                              Слова, встречающиеся реже, будут заменены токеном <UNK>.
        
        # --- Параметры трансформации и батчей ---
        image_size (int): Конечный размер (сторона) изображений после применения трансформаций
                          (например, 224x224).
        batch_size (int): Количество образцов данных в одном батче.
        num_workers (int): Количество процессов или потоков для параллельной загрузки данных.
        
        # --- Параметры разбиения (Split) ---
        val_split (float): Доля (от 0.0 до 1.0) всего датасета, используемая для валидации.
        random_state (int): Фиксированное начальное число для генератора случайных чисел, 
                            обеспечивающее воспроизводимость разбиения на train/val.
    """
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        images_dir: str,
        caption_file: str,
        vocab_path: str,
        freq_threshold: int = 5,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        random_state: int = 42,
    ):
        self.dataset_cls = dataset_cls
        self.images_dir = images_dir
        self.caption_file = caption_file
        self.vocab_path = vocab_path
        self.freq_threshold = freq_threshold
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.random_state = random_state

class CaptioningDataPipeline:
    """
    Класс, инкапсулирующий весь процесс подготовки данных для задачи 
    автоматического описания изображений (image captioning).
    
    methods:
        initialize_vocabulary() -> Vocabulary:
            Создает или загружает словарь (Vocabulary) на основе конфигурации.
    
        create_data_loaders() -> Tuple[DataLoader, DataLoader]:
            Создает и возвращает загрузчики данных (DataLoader) для обучения и валидации.
    """
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.vocab = None
        
    def initialize_vocabulary(self) -> Vocabulary:
        cfg = self.config
        
        if os.path.exists(cfg.vocab_path):
            self.vocab = Vocabulary.load(cfg.vocab_path)
            print(f"Loaded existing vocabulary from {cfg.vocab_path}.")
            return self.vocab
        else:
            print("Creating a new Vocabulary...")

            all_captions = cfg.dataset_cls.get_all_captions(cfg.caption_file)
            
            vocab = Vocabulary(freq_threshold=cfg.freq_threshold)
            vocab.build_vocabulary(all_captions) 
            vocab.save_vocab(cfg.vocab_path)
            self.vocab = vocab
            return vocab

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self.vocab is None:
            raise RuntimeError("First, you need to initialize the vocabulary (initialize_vocabulary).")
            
        cfg = self.config
        
        train_transform = create_training_transforms(image_size=cfg.image_size, resize_size=256)
        
        dataset = cfg.dataset_cls(cfg.images_dir, cfg.caption_file, transform=train_transform)

        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=cfg.val_split, 
            random_state=cfg.random_state
        )

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        collate_fn_final = lambda batch: collate_fn_for_token(batch, self.vocab)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn_final
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn_final
        )
        
        return train_loader, val_loader