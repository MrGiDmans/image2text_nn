import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any

# Мои импорты
from utils.dataSet import Flickr8kDataset
from utils.dataLoader import DataLoaderConfig, CaptioningDataPipeline
from models import EncoderCNN, DecoderRNN, CaptioningModel

# Глобальные настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100 # Количество эпох обучения
LEARNING_RATE = 1e-4

@torch.no_grad()
def validate_model(model, loader, criterion, device):
    model.eval() # Установка режима инференса
    total_loss = 0
    
    # Можно добавить другую метрику, например, счетчик BLEU
    
    for images, captions, lengths_tensor in tqdm(loader, desc="Validation"):
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths_tensor.tolist() 

        # Прямой проход
        logits = model(images, captions, lengths) 

        # Подготовка для Loss
        targets = captions[:, 1:] 
        logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    
    # После валидации, если вы сразу возвращаетесь к обучению
    # model.train()
    
    return avg_loss

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    loss: float, 
    save_dir: Path,  # Принимаем директорию, а не полный путь к файлу
    is_best: bool, 
    best_loss_so_far: Optional[float] = None
):
    """
    Сохраняет контрольную точку (чекпоинт) модели.
    Сохраняет только LAST (последний) и BEST (лучший) чекпоинты.
    
    Args:
        model: Модель PyTorch.
        optimizer: Оптимизатор PyTorch.
        epoch: Номер текущей эпохи.
        loss: Текущая валидационная потеря (loss).
        save_dir: Директория для сохранения файлов.
        is_best: Флаг, указывающий, является ли текущая модель лучшей.
        best_loss_so_far: Лучшая потеря, достигнутая до сих пор.
    """
    # 1. Сбор состояния (включаем лучшую потерю для удобства при возобновлении)
    state: Dict[str, Any] = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss,
        'best_val_loss': best_loss_so_far if best_loss_so_far is not None else loss
    }
    
    # Убедимся, что директория существует
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Сохранение LAST (последней) модели ---
    last_path = save_dir / "last_checkpoint.pth.tar"
    torch.save(state, last_path)
    print(f"✅ LAST Checkpoint сохранен в {last_path.name} (Epoch {epoch}, Loss: {loss:.4f})")

    # --- 3. Сохранение BEST (лучшей) модели ---
    if is_best:
        best_path = save_dir / "best_checkpoint.pth.tar"
        # Сохраняем то же состояние, но под другим именем
        torch.save(state, best_path)
        print(f"⭐ Лучший Checkpoint сохранен в {best_path.name}")


def train_epoch(model, loader, criterion, optimizer, device, vocab):
    model.train()
    total_loss = 0
    
    # tqdm делает цикл более информативным
    for images, captions, lengths_tensor in tqdm(loader, desc="Training"):
        
        # 1. Перенос данных на устройство
        images = images.to(device)
        captions = captions.to(device)
        
        # 2. Подготовка 'lengths' для DecoderRNN.forward
        # pack_padded_sequence требует список int или 1D tensor на CPU
        lengths = lengths_tensor.tolist() 

        # 3. Обнуление градиентов
        optimizer.zero_grad() 

        # 4. Прямой проход
        # logits: [B, Tmax, vocab_size]
        logits = model(images, captions, lengths)

        # 5. Подготовка для Loss
        # Целевые метки: все токены, кроме BOS (индекс 0)
        targets = captions[:, 1:] 
        
        # Сглаживание тензоров для nn.CrossEntropyLoss
        # Logits: [B * (Tmax-1), Vocab_size]
        logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1)) 
        # Targets: [B * (Tmax-1)]
        targets_flat = targets.reshape(-1)

        # 6. Вычисление Loss
        loss = criterion(logits_flat, targets_flat) 
        total_loss += loss.item()

        # 7. Обратное распространение и обновление весов
        loss.backward() 
        optimizer.step()
        
    avg_loss = total_loss / len(loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def main(images_dir, caption_file, vocab_path):
    
    # 1. Инициализация конвейера данных и словаря
    config = DataLoaderConfig(
        dataset_cls=Flickr8kDataset,
        images_dir=images_dir,
        caption_file=caption_file,
        vocab_path=vocab_path,
        freq_threshold=5,
        image_size=224,
        batch_size=32,
        num_workers=4,
        val_split=0.2,
        random_state=42,
    )
    
    pipeline = CaptioningDataPipeline(config)
    vocab = pipeline.initialize_vocabulary()
    train_loader, val_loader = pipeline.create_data_loaders()
    
    print(f"Загрузчики созданы: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
    
    # 2. Инициализация модели, Loss и Оптимизатора
    best_val_loss = float('inf')
    # Параметры модели
    EMBED_SIZE = 512
    DECODER_DIM = 512
    ENCODER_DIM = 512
    DROPOUT = 0.5

    encoder = EncoderCNN(embed_dim=ENCODER_DIM, fine_tune=False)
    decoder = DecoderRNN(
        vocab_size=len(vocab),  
        embed_size=EMBED_SIZE, 
        decoder_dim=DECODER_DIM, 
        encoder_dim=ENCODER_DIM, 
        dropout=DROPOUT
    )

    model = CaptioningModel(encoder, decoder, vocab, device=DEVICE)

    # Функция потерь: CrossEntropy, игнорируем PAD токены
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx).to(DEVICE)
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Основной цикл обучения
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        train_epoch(model, train_loader, criterion, optimizer, DEVICE, vocab)
        is_best = False
        # 1. Валидация
        val_loss = validate_model(model, val_loader, criterion, DEVICE)
        
        # 2. Сохранение (логика чекпоинта)
        if val_loss < best_val_loss:
            is_best = True
            best_val_loss = val_loss
            
        # Сохраняем модель после каждой эпохи (или только лучшую)
        save_checkpoint(
            model, 
            optimizer, 
            epoch, 
            best_val_loss, 
            path=f"./checkpoints/model_e{epoch}.pth.tar",
            is_best=is_best
        )
        
        # Возвращаем модель в режим обучения для следующей эпохи
        model.train()
        
    print("\nОбучение завершено.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")

    # Убедитесь, что здесь нет лишних запятых
    images_dir = str(PROJECT_ROOT / "data" / "flickr8k" / "images")
    caption_file = str(PROJECT_ROOT / "data" / "flickr8k" / "captions.csv")
    vocab_path = str(PROJECT_ROOT / "vocab.pkl")
    
    main(images_dir, caption_file, vocab_path)