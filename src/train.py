from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models import CaptioningModel, DecoderRNN, EncoderCNN
from utils.dataLoader import CaptioningDataPipeline, DataLoaderConfig
from utils.dataSet import Flickr8kDataset
from utils.vocabulary import Vocabulary


# -----------------------------------------------------------------------------
# Конфигурации
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    images_dir: str = ""
    caption_file: str = ""
    vocab_path: str = ""
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.2
    freq_threshold: int = 5
    pin_memory: bool = True


@dataclass
class ModelConfig:
    embed_size: int = 512
    decoder_dim: int = 512
    encoder_dim: int = 512
    dropout: float = 0.5
    encoder_fine_tune: bool = False
    encoded_image_size: int = 14


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-4
    lr_mode: str = "continue"  # continue | finetune
    finetune_lr: float = 1e-5
    grad_clip: float = 5.0
    use_amp: bool = True
    save_every_epoch: bool = False
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None


@dataclass
class SchedulerConfig:
    type: str = "ReduceLROnPlateau"
    factor: float = 0.5
    patience: int = 2
    min_lr: float = 1e-7


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5
    min_delta: float = 1e-4
    track_overfitting: bool = True
    overfitting_gap_threshold: float = 0.5  # Порог разницы train/val loss
    overfitting_patience: int = 3  # Эпохи с большим gap до остановки


@dataclass
class LoggingConfig:
    tensorboard: bool = False
    tensorboard_dir: str = "./runs"
    use_wandb: bool = False
    wandb: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    bleu: bool = True
    use_sacrebleu_if_available: bool = True


@dataclass
class ExperimentConfig:
    device: str = "cuda"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_dict(cls, raw: Dict) -> "ExperimentConfig":
        return cls(
            device=raw.get("device", "cuda"),
            seed=raw.get("seed", 42),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            early_stopping=EarlyStoppingConfig(**raw.get("early_stopping", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            metrics=MetricsConfig(**raw.get("metrics", {})),
        )


# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------


def _normalize_value(value):
    """Преобразует строковые числовые значения в числа."""
    if isinstance(value, str):
        try:
            # Пробуем преобразовать в float (для научной нотации типа 1e-4)
            if 'e' in value.lower() or 'E' in value:
                return float(value)
            # Пробуем int, затем float
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
            return float(value)
        except (ValueError, AttributeError):
            pass
    return value


def _normalize_dict(d: Dict) -> Dict:
    """Рекурсивно нормализует словарь."""
    if not isinstance(d, dict):
        return d
    return {k: _normalize_dict(v) if isinstance(v, dict) else _normalize_value(v) for k, v in d.items()}


def load_config(path: Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = _normalize_dict(data)
    return ExperimentConfig.from_dict(data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(target: str) -> torch.device:
    if target == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        model: CaptioningModel,
        vocab: Vocabulary,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.device = get_device(cfg.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx).to(
            self.device
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self.cfg.training.use_amp and self.device.type == "cuda")

        self.ckpt_dir = Path(self.cfg.training.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.start_epoch = 1
        self.epochs_without_improve = 0

        if self.cfg.training.resume_from:
            self._load_checkpoint(Path(self.cfg.training.resume_from))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        train_cfg = self.cfg.training
        params = []

        decoder_params = {
            "params": self.model.decoder.parameters(),
            "lr": train_cfg.lr,
        }
        params.append(decoder_params)

        encoder_lr = (
            train_cfg.finetune_lr if train_cfg.lr_mode == "finetune" else train_cfg.lr
        )
        encoder_params = {
            "params": self.model.encoder.parameters(),
            "lr": encoder_lr,
        }
        params.append(encoder_params)

        # Используем базовый lr, но параметры уже имеют свои индивидуальные LR
        return torch.optim.Adam(params)

    def _build_scheduler(self):
        sched_cfg = self.cfg.scheduler
        if sched_cfg.type.lower() == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_cfg.factor,
                patience=sched_cfg.patience,
                min_lr=sched_cfg.min_lr,
            )
        return None

    def train(self) -> None:
        total_epochs = self.cfg.training.epochs
        for epoch in range(self.start_epoch, total_epochs + 1):
            print(f"\n--- Epoch {epoch}/{total_epochs} ---")
            train_loss = self._run_epoch(epoch)
            val_loss, bleu = self._validate(epoch)

            if self.scheduler:
                self.scheduler.step(val_loss)

            improved = val_loss < (self.best_val_loss - self.cfg.early_stopping.min_delta)
            if improved:
                self.best_val_loss = val_loss
                self.epochs_without_improve = 0
            else:
                self.epochs_without_improve += 1

            # Отслеживание переобучения
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            overfitting_detected = self._check_overfitting(train_loss, val_loss)

            self._save_checkpoint(epoch, val_loss, is_best=improved)

            # Логирование с информацией о переобучении
            gap = train_loss - val_loss
            gap_msg = f", gap={gap:.4f}" if gap < 0 else f", gap={gap:.4f} ⚠️"
            overfit_msg = " [ПЕРЕОБУЧЕНИЕ]" if overfitting_detected else ""
            bleu_msg = f", BLEU: {bleu:.2f}" if bleu is not None else ""
            print(
                f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}{gap_msg}{bleu_msg}{overfit_msg}"
            )

            if self._should_stop():
                print("Early stopping активирован.")
                break

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")

        for images, captions, lengths in pbar:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)
            lengths_list = lengths.tolist()

            self.optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(device_type=device_type, enabled=self.cfg.training.use_amp):
                logits = self.model(images, captions, lengths_list)
                loss = self._compute_loss(logits, captions)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.cfg.training.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def _validate(self, epoch: int) -> Tuple[float, Optional[float]]:
        self.model.eval()
        total_loss = 0.0
        predictions: List[List[int]] = []
        references: List[List[int]] = []

        with torch.no_grad():
            for images, captions, lengths in tqdm(
                self.val_loader, desc=f"Val {epoch}"
            ):
                images = images.to(self.device, non_blocking=True)
                captions = captions.to(self.device, non_blocking=True)
                lengths_list = lengths.tolist()

                logits = self.model(images, captions, lengths_list)
                loss = self._compute_loss(logits, captions)
                total_loss += loss.detach().item()

                if self.cfg.metrics.bleu:
                    preds = logits.detach().argmax(dim=-1).cpu().tolist()
                    caps = captions.cpu().tolist()
                    for i, length in enumerate(lengths_list):
                        ref = caps[i][1 : max(1, length - 1)]  # без BOS/EOS
                        pred_seq = []
                        for token in preds[i]:
                            if token in (self.vocab.eos_idx, self.vocab.pad_idx):
                                if token == self.vocab.eos_idx:
                                    break
                                continue
                            pred_seq.append(token)
                        references.append(ref)
                        predictions.append(pred_seq)

        avg_loss = total_loss / len(self.val_loader)
        bleu_score = self._compute_bleu(predictions, references)
        return avg_loss, bleu_score

    def _compute_loss(self, logits: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        targets = captions[:, 1:] 
        logits_trimmed = logits[:, :-1, :]
        return self.criterion(
            logits_trimmed.reshape(-1, logits_trimmed.size(-1)),
            targets.reshape(-1),
        )

    def _compute_bleu(
        self, predictions: List[List[int]], references: List[List[int]]
    ) -> Optional[float]:
        if not self.cfg.metrics.bleu or len(predictions) == 0:
            return None

        pred_tokens = [" ".join(self.vocab.decode_indexes(seq)) for seq in predictions]
        ref_tokens = [" ".join(self.vocab.decode_indexes(seq)) for seq in references]

        if self.cfg.metrics.use_sacrebleu_if_available:
            try:
                import sacrebleu

                bleu = sacrebleu.corpus_bleu(pred_tokens, [ref_tokens])
                return float(bleu.score)
            except ImportError:
                pass

        try:
            from nltk.translate.bleu_score import corpus_bleu

            pred_split = [sent.split() for sent in pred_tokens]
            ref_split = [[sent.split()] for sent in ref_tokens]
            bleu = corpus_bleu(ref_split, pred_split)
            return float(bleu * 100.0)
        except ImportError:
            print("NLTK/SacreBLEU не найдены — BLEU пропущен.")
            return None

    def _check_overfitting(self, train_loss: float, val_loss: float) -> bool:
        """Проверяет признаки переобучения."""
        if not self.cfg.early_stopping.track_overfitting:
            return False

        gap = train_loss - val_loss
        threshold = self.cfg.early_stopping.overfitting_gap_threshold

        # Переобучение: train loss значительно меньше val loss
        if gap < -threshold:
            self.epochs_with_overfitting += 1
            return True
        else:
            self.epochs_with_overfitting = 0
            return False

    def _should_stop(self) -> bool:
        if not self.cfg.early_stopping.enabled:
            return False

        # Остановка из-за отсутствия улучшения
        if self.epochs_without_improve >= self.cfg.early_stopping.patience:
            return True

        # Остановка из-за переобучения
        if (
            self.cfg.early_stopping.track_overfitting
            and self.epochs_with_overfitting
            >= self.cfg.early_stopping.overfitting_patience
        ):
            print(
                f"⚠️ Обнаружено переобучение: gap между train/val loss превышает порог "
                f"{self.cfg.early_stopping.overfitting_gap_threshold} "
                f"в течение {self.epochs_with_overfitting} эпох."
            )
            return True

        return False

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler_state": self.scaler.state_dict()
            if self.scaler.is_enabled()
            else None,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
        }

        last_path = self.ckpt_dir / "last_checkpoint.pth.tar"
        torch.save(state, last_path)

        if self.cfg.training.save_every_epoch:
            epoch_path = self.ckpt_dir / f"epoch_{epoch:03d}.pth.tar"
            torch.save(state, epoch_path)

        if is_best:
            best_path = self.ckpt_dir / "best_checkpoint.pth.tar"
            torch.save(state, best_path)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            print(f"Checkpoint {path} не найден, стартуем с нуля.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        if self.scaler.is_enabled() and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Возобновление с эпохи {self.start_epoch}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training loop for image captioning.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к YAML-конфигу.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Путь к чекпоинту для возобновления.",
    )
    return parser.parse_args()


def build_dataloaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    data_cfg = cfg.data
    data_loader_cfg = DataLoaderConfig(
        dataset_cls=Flickr8kDataset,
        images_dir=data_cfg.images_dir,
        caption_file=data_cfg.caption_file,
        vocab_path=data_cfg.vocab_path,
        freq_threshold=data_cfg.freq_threshold,
        image_size=data_cfg.image_size,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        val_split=data_cfg.val_split,
        random_state=cfg.seed,
        pin_memory=data_cfg.pin_memory,
    )

    pipeline = CaptioningDataPipeline(data_loader_cfg)
    vocab = pipeline.initialize_vocabulary()
    train_loader, val_loader = pipeline.create_data_loaders()
    print(
        f"Data ready: train batches={len(train_loader)}, val batches={len(val_loader)}, vocab={len(vocab)}"
    )
    return train_loader, val_loader, vocab


def build_model(cfg: ExperimentConfig, vocab: Vocabulary) -> CaptioningModel:
    model_cfg = cfg.model
    encoder = EncoderCNN(
        encoded_image_size=model_cfg.encoded_image_size,
        embed_dim=model_cfg.encoder_dim,
        fine_tune=model_cfg.encoder_fine_tune,
    )
    decoder = DecoderRNN(
        vocab_size=len(vocab), 
        embed_size=model_cfg.embed_size,
        decoder_dim=model_cfg.decoder_dim,
        encoder_dim=model_cfg.encoder_dim,
        dropout=model_cfg.dropout,
    )
    device = get_device(cfg.device)
    return CaptioningModel(encoder, decoder, vocab, device=device)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    if args.resume:
        cfg.training.resume_from = args.resume

    set_seed(cfg.seed)

    train_loader, val_loader, vocab = build_dataloaders(cfg)
    model = build_model(cfg, vocab)

    trainer = Trainer(cfg, model, vocab, train_loader, val_loader)
    trainer.train()


if __name__ == "__main__":
    main()