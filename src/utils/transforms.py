from torchvision import transforms

def create_training_transforms(image_size=224, resize_size=256):
    """Создает набор трансформаций для обучающего датасета с аугментацией."""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Normalize должен быть последним (после ToTensor)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_validation_transforms(image_size=224, resize_size=256):
    """Создает набор трансформаций для валидационного/тестового датасета без аугментации."""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])