import os
from typing import List
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):
    """
    Класс Dataset для набора данных Flickr8k, где каждая строка 
    представляет собой пару (изображение, одно описание).
    В самом файле captions.csv для каждого изображения имеется по 5 описаний.
    """
    def __init__(self, images_dir, caption_file, transform=None):
        """
        Инициализация набора данных.
        
        Args:
            images_dir (str): Путь к директории, содержащей изображения.
            caption_file (str): Путь к CSV/TXT файлу с описаниями (image, caption).
            transform (callable, optional): Необязательное преобразование, 
                                            применяемое к изображению.
        """
        self.df = pd.read_csv(caption_file, sep=",", quotechar='"')

        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        """
        Извлекает элемент набора данных по заданному индексу.
        
        Args:
            idx (int): Индекс элемента.
        
        Returns:
            tuple: Кортеж (image, caption).
        """
        row = self.df.iloc[idx]
        
        img_name = row["image"]
        caption  = str(row["caption"])

        img_path = os.path.join(self.images_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден: {img_path}")

        if self.transform:
            image = self.transform(image)

        return image, caption

    @staticmethod
    def get_all_captions(caption_file: str) -> List[str]:
        """Статический метод для загрузки всех подписей из файла."""

        import pandas as pd
        df = pd.read_csv(caption_file) 
        
        return df['caption'].tolist()