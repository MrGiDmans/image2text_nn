from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from models import CaptioningModel, DecoderRNN, EncoderCNN
from utils.vocabulary import Vocabulary


def load_model(
    checkpoint_path: str,
    vocab_path: str,
    device: torch.device,
) -> tuple[CaptioningModel, Vocabulary]:
    """Загружает модель и словарь из чекпоинта."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = Vocabulary.load(vocab_path)

    # Восстанавливаем архитектуру модели
    encoder = EncoderCNN(
        encoded_image_size=14,
        embed_dim=512,
        fine_tune=False,
    )
    decoder = DecoderRNN(
        vocab_size=len(vocab),
        embed_size=512,
        decoder_dim=512,
        encoder_dim=512,
        dropout=0.5,
    )
    model = CaptioningModel(encoder, decoder, vocab, device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Предобработка изображения для модели."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor


def generate_caption(
    model: CaptioningModel,
    image_path: str,
    device: torch.device,
    max_len: int = 20,
    beam_size: int = 3,
    mode: str = "greedy",
) -> str:
    """Генерирует подпись для изображения."""
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        results = model.generate(
            image_tensor, max_len=max_len, beam_size=beam_size, mode=mode
        )

    # Извлекаем слова (убираем BOS/EOS)
    seq_idx, seq_words, score = results[0]
    words = [w for w in seq_words if w not in ["<bos>", "<eos>", "<pad>", "<unk>"]]
    caption = " ".join(words)
    return caption


def main():
    parser = argparse.ArgumentParser(description="Тестирование модели генерации подписей")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_checkpoint.pth.tar",
        help="Путь к чекпоинту модели",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="./vocab.pkl",
        help="Путь к словарю",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Путь к изображению для тестирования",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="Максимальная длина генерируемой подписи",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="Размер beam для beam search",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Режим генерации: greedy или beam",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство для инференса",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Загрузка модели из {args.checkpoint}...")
    model, vocab = load_model(args.checkpoint, args.vocab, device)
    print(f"Модель загружена. Размер словаря: {len(vocab)}")

    print(f"\nГенерация подписи для {args.image}...")
    caption = generate_caption(
        model,
        args.image,
        device,
        max_len=args.max_len,
        beam_size=args.beam_size,
        mode=args.mode,
    )

    print(f"\n📷 Изображение: {args.image}")
    print(f"📝 Подпись: {caption}")
    print(f"🔧 Режим: {args.mode}")


if __name__ == "__main__":
    main()

