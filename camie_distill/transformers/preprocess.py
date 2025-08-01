from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

class PadToSquare:
    ...
    # (Code exactly as in the previous message)
    ...

IMG_SIZE = 512   # EfficientNet‑V2‑L default used by Camie

preprocess = transforms.Compose([
    transforms.Resize(size=IMG_SIZE,
        max_size=IMG_SIZE,
        interpolation=InterpolationMode.LANCZOS,
        antialias=True),
    PadToSquare(fill=0),
    transforms.ToTensor(),       # 0‑1 float32, no mean/std shift
])