from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg"}


def find_image_dirs(root: str | Path) -> List[Path]:
    """
    Findet alle Verzeichnisse unter root, die mindestens ein Bild enthalten.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")

    dirs = set()
    # rglob ist ok für Cluster; wenn extrem groß, kann man später optimieren
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            dirs.add(p.parent)

    return sorted(dirs)


def collect_images_from_dirs(image_dirs: List[Path]) -> List[Path]:
    """
    Sammelt alle Bildpfade aus den gefundenen Ordnern.
    """
    paths: List[Path] = []
    for d in image_dirs:
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(p)
    # stabil sortiert
    return sorted(paths)


class RAMImageDataset(Dataset):
    """
    Lädt ALLE Bilder beim Init in RAM (als float32 Tensor, normalisiert).
    __getitem__ ist dann nur noch indexing.
    """

    def __init__(
        self,
        img_paths: List[Path],
        img_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        verbose: bool = True,
    ):
        if not img_paths:
            raise ValueError("img_paths is empty")

        self.img_paths = img_paths
        self.img_size = img_size

        tf = [T.ToImage(), T.Resize(img_size), T.ToDtype(torch.float32, scale=True)]
        if normalize:
            tf.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.tf = T.Compose(tf)

        self.data: List[torch.Tensor] = []
        bad: List[str] = []

        if verbose:
            print(f"[RAMImageDataset] Loading {len(img_paths)} images into RAM...")

        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
                x = self.tf(img)  # (3,H,W) float32
                self.data.append(x)
            except Exception as e:
                bad.append(f"{p} :: {type(e).__name__}: {e}")

        if verbose:
            print(f"[RAMImageDataset] Loaded OK: {len(self.data)}")
            if bad:
                print(f"[RAMImageDataset] Failed: {len(bad)} (showing first 10)")
                for s in bad[:10]:
                    print("  -", s)

        if len(self.data) == 0:
            raise RuntimeError("No images could be loaded into RAM (all failed).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class CUDAPrefetcher:
    """
    Wrappt einen CPU DataLoader und schiebt den nächsten Batch bereits asynchron auf die GPU.
    Usage:
        for x in CUDAPrefetcher(dl, device):
            ...
    """
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            # CPU fallback: einfach normal iterieren
            for batch in self.loader:
                yield batch.to(self.device)
            return

        first = True
        it = iter(self.loader)

        def _preload():
            nonlocal next_batch
            try:
                next_batch = next(it)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = next_batch.to(self.device, non_blocking=True)

        next_batch = None
        _preload()

        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            _preload()
            yield batch


@dataclass
class RAMDataBundle:
    image_dirs: List[Path]
    image_paths: List[Path]
    dataset: RAMImageDataset
    loader: DataLoader
    prefetcher: CUDAPrefetcher


from typing import Dict

def split_paths_deterministic(
    paths: List[Path],
    train_pct: float,
    val_pct: float,
    test_pct: float,
    seed: int,
) -> Dict[str, List[Path]]:
    """
    Deterministischer Split: gleiche seed -> gleiche Splits.
    """
    total = train_pct + val_pct + test_pct
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test must sum to 1.0, got {total}")

    # stabile Sortierung + deterministische Permutation
    paths = sorted(paths)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(paths), generator=g).tolist()
    paths = [paths[i] for i in perm]

    n = len(paths)
    n_train = int(round(n * train_pct))
    n_val = int(round(n * val_pct))
    # rest in test (damit Summe exakt passt)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


@dataclass
class RAMSplitBundle:
    image_dirs: List[Path]
    all_image_paths: List[Path]
    splits: Dict[str, List[Path]]
    datasets: Dict[str, RAMImageDataset]
    loaders: Dict[str, DataLoader]
    prefetchers: Dict[str, CUDAPrefetcher]


def build_ram_dataloaders_split(
    root: str | Path,
    batch_size: int,
    device: torch.device,
    train_pct: float = 0.8,
    val_pct: float = 0.1,
    test_pct: float = 0.1,
    seed: int = 42,
    img_size: Tuple[int, int] = (224, 224),
    shuffle_train: bool = True,
    num_workers: int = 0,  # RAM dataset -> 0 reicht
    pin_memory: bool = True,
    drop_last: bool = True,
    verbose: bool = True,
) -> RAMSplitBundle:
    image_dirs = find_image_dirs(root)
    if verbose:
        print(f"[build_ram_dataloaders_split] Found {len(image_dirs)} image directories.")
        for d in image_dirs[:20]:
            print("  -", d)
        if len(image_dirs) > 20:
            print(f"  ... (+{len(image_dirs)-20} more)")

    all_paths = collect_images_from_dirs(image_dirs)
    if verbose:
        print(f"[build_ram_dataloaders_split] Collected {len(all_paths)} total images.")

    splits = split_paths_deterministic(
        all_paths, train_pct=train_pct, val_pct=val_pct, test_pct=test_pct, seed=seed
    )

    if verbose:
        print("[split sizes]",
              f"train={len(splits['train'])}",
              f"val={len(splits['val'])}",
              f"test={len(splits['test'])}")

    datasets = {
        k: RAMImageDataset(v, img_size=img_size, normalize=True, verbose=verbose)
        for k, v in splits.items()
        if len(v) > 0
    }

    loaders: Dict[str, DataLoader] = {}
    prefetchers: Dict[str, CUDAPrefetcher] = {}

    for split_name, ds in datasets.items():
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle_train if split_name == "train" else False),
            num_workers=num_workers,
            pin_memory=pin_memory and (device.type == "cuda"),
            drop_last=(drop_last if split_name == "train" else False),
        )
        prefetchers[split_name] = CUDAPrefetcher(loaders[split_name], device=device)

    return RAMSplitBundle(
        image_dirs=image_dirs,
        all_image_paths=all_paths,
        splits=splits,
        datasets=datasets,
        loaders=loaders,
        prefetchers=prefetchers,
    )