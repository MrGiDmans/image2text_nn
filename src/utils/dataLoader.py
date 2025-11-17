from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from src.utils.dataSet import Flickr8kDataset
from src.utils.transforms import create_training_transforms
from src.utils.collate_fn import collate_fn_for_token
from src.utils.vocabulary import Vocabulary
import os

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # корневая папка проекта
root_dir = os.path.join(root_path, "./data/flickr8k/images/")
caption_file = os.path.join(root_path, "./data/flickr8k/captions.csv")

# 1) Создаём временный датасет без трансформаций → только для текста
raw_dataset = Flickr8kDataset(root_dir, caption_file, transform=None)

# 2) Строим словарь
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(raw_dataset.captions)
vocab.save_vocab(os.path.join(root_path, "./vocab.pkl"))

# 3) Датасет с трансформациями
train_transform = create_training_transforms(image_size=224, resize_size=256)
dataset = Flickr8kDataset(root_dir, caption_file, transform=train_transform)

# 4) Split
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset   = Subset(dataset, val_idx)

# 5) DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda batch: collate_fn_for_token(batch, vocab)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: collate_fn_for_token(batch, vocab)
)

debug_dataset = Subset(dataset, range(32))
debug_loader = DataLoader(
    debug_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda batch: collate_fn_for_token(batch, vocab)
)
