import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError # Импортируем UnidentifiedImageError

def max_size_w_h_improved(path_to_dir: str) -> tuple[int, int, int]:
    """
    Находит максимальную площадь (ширина * высота), а также
    соответствующие максимальные ширину и высоту среди изображений в директории.
    
    Args:
        path_to_dir: Путь к директории с изображениями.
    
    Returns:
        Кортеж (max_area, max_width, max_height).
    """
    path = Path(path_to_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    
    max_area = 0
    max_width = 0
    max_height = 0
    min_width = float('inf')
    min_height = float('inf')
    
    # Используем path.iterdir() для работы с Path-объектами
    for entry in path.iterdir():
        # Проверяем, является ли это файлом и имеет ли он нужное расширение
        if entry.is_file() and entry.suffix.lower() in image_extensions:
            try:
                # Открываем изображение
                with Image.open(entry) as img:
                    w, h = img.size
                    area = w * h
                    
                    if w < min_width:
                        min_width = w

                    if h < min_height:
                        min_height = h

                    if area > max_area:
                        max_area = area
                        max_width = w
                        max_height = h
                        
            # Обработка ошибки, если PIL не может открыть файл (например, он поврежден)
            except UnidentifiedImageError:
                print(f"⚠️ Warning: Не удалось открыть файл изображения: {entry.name}")
            except Exception as e:
                print(f"❌ Error processing file {entry.name}: {e}")
                
    # Возвращаем площадь, а не размер файла на диске
    return max_width, max_height, min_width, min_height

if __name__ == "__main__":
    # Убедитесь, что эта директория существует и содержит изображения для теста!
    # path_for_test = "./data/flickr8k/images/" 
    # В качестве примера:
    path_for_test = os.path.join(os.path.dirname(__file__), "./data/flickr8k/images/") 

    # Создайте тестовую директорию, если ее нет
    if not os.path.exists(path_for_test):
        os.makedirs(path_for_test, exist_ok=True)
        print(f"Created test directory: {path_for_test}")
        # Здесь можно добавить код для создания нескольких тестовых изображений
        
    max_w, max_h, min_w, min_h = max_size_w_h_improved(path_for_test)
    print(f"Max Width: {max_w}, Max Height: {max_h}, Min Width: {min_w}, Min Height: {min_h}")