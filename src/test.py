import torch
import pickle
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Импорты ваших модулей (убедитесь, что они доступны)
from utils.transforms import create_training_transforms
from models import EncoderCNN, DecoderRNN, CaptioningModel
from utils.vocabulary import Vocabulary # Assuming Vocabulary is in utils.vocabulary

# --- Константы и Пути ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к файлам (замените на свои актуальные пути)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = PROJECT_ROOT / "vocab.pkl"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "BEST_model_e5.pth.tar" # Предполагаем, что вы сохранили лучший чекпоинт
TEST_IMAGE_PATH = PROJECT_ROOT / "data" / "flickr8k" / "images" / "3737492755_bcfb800ed1.jpg" # Пример тестового изображения

# Параметры модели (должны совпадать с параметрами, использованными при обучении)
EMBED_SIZE = 512
DECODER_DIM = 512
ENCODER_DIM = 512
DROPOUT = 0.5
IMAGE_SIZE = 224


def load_image(image_path, image_size):
    """Загружает и предобрабатывает изображение для инференса."""
    print(f"Загрузка изображения: {image_path}")
    # Трансформации для инференса (должны соответствовать тем, что использовались в DataLoaderConfig)
    transform = create_training_transforms(image_size=image_size, resize_size=256)
    
    image = Image.open(image_path).convert('RGB')
    image_tensor: torch.Tensor = transform(image)
    # Добавление размерности батча (Batch_size = 1)
    return image_tensor.unsqueeze(0).to(DEVICE) 

def load_model(vocab, checkpoint_path):
    """Инициализирует модель и загружает обученные веса."""
    print(f"Инициализация модели и загрузка чекпоинта: {checkpoint_path}")

    # 1. Инициализация подмоделей
    encoder = EncoderCNN(embed_dim=ENCODER_DIM, fine_tune=False)
    decoder = DecoderRNN(
        vocab_size=len(vocab),  
        embed_size=EMBED_SIZE, 
        decoder_dim=DECODER_DIM, 
        encoder_dim=ENCODER_DIM, 
        dropout=DROPOUT
    )

    # 2. Инициализация обертки CaptioningModel
    model = CaptioningModel(encoder, decoder, vocab, device=DEVICE)
    model.eval() # Установка режима инференса (отключение dropout, batch_norm и т.д.)

    # 3. Загрузка весов
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Файл чекпоинта не найден: {checkpoint_path}")

    # Загружаем состояние модели
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Модель успешно загружена. Эпоха обучения: {checkpoint['epoch']}, Лучшая Loss: {checkpoint['best_loss']:.4f}")
    return model

def inference(model: CaptioningModel, image_tensor: torch.Tensor, max_len=25, beam_size=5):
    """Выполняет инференс (генерацию подписи)."""
    print("\n--- Генерация подписей (Инференс) ---")
    
    # 1. Greedy Search
    greedy_results = model.generate(image_tensor, max_len=max_len, mode="greedy")
    
    # 2. Beam Search (если B=1)
    if image_tensor.size(0) == 1:
        beam_results = model.generate(image_tensor, max_len=max_len, beam_size=beam_size, mode="beam")
    else:
        beam_results = None

    print("\n[Результаты Greedy Search]:")
    for idxs, words, score in greedy_results:
        # Удаляем токены BOS/EOS/PAD из вывода, если они есть
        clean_words = [w for w in words if w not in [model.vocab.bos_token, model.vocab.eos_token, model.vocab.pad_token]]
        caption = " ".join(clean_words)
        print(f"Подпись: **{caption}** (Score: {score:.4f})")
        
    if beam_results:
        print(f"\n[Результаты Beam Search (k={beam_size})]:")
        for idxs, words, score in beam_results:
             clean_words = [w for w in words if w not in [model.vocab.bos_token, model.vocab.eos_token, model.vocab.pad_token]]
             caption = " ".join(clean_words)
             print(f"Лучшая Подпись: **{caption}** (Score: {score:.4f})")


if __name__ == "__main__":
    try:
        # 1. Загрузка словаря
        vocab_obj = Vocabulary.load(VOCAB_PATH)
        
        # 2. Загрузка и подготовка изображения (B=1)
        image_input = load_image(TEST_IMAGE_PATH, IMAGE_SIZE)
        
        # 3. Загрузка модели и весов
        caption_model = load_model(vocab_obj, CHECKPOINT_PATH)
        
        # 4. Выполнение инференса
        inference(caption_model, image_input)
        
    except Exception as e:
        print(f"Произошла ошибка во время инференса: {e}")
        print("Убедитесь, что пути к файлам и параметры модели указаны верно и соответствуют обучению.")