from __future__ import annotations
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# 0) ABSOLUT WICHTIG: main_config patchen BEVOR irgendwas importiert,
#    was main_config/main.py ziehen könnte.
# ============================================================
import types, sys
import warnings
warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter.*",
    category=UserWarning,
    module="torchvision.transforms.functional"
)

import warnings

class _IgnoreAntialiasWarning:
    def __init__(self):
        self._orig = warnings.showwarning

    def __call__(self, message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if "default value of the antialias parameter" in msg and "torchvision" in filename:
            return
        return self._orig(message, category, filename, lineno, file, line)

warnings.showwarning = _IgnoreAntialiasWarning()

# ============================================================
# 1) Standard imports
# ============================================================
import os
import time
import json
import argparse
import random
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T
from PIL import Image

# ============================================================
# 2) HARD-CODED WANDB KEY (wie du wolltest)
# ============================================================
WANDB_API_KEY = "97646993dfb3ed347361401308dd9377b8c7365a"

# ============================================================
# 3) Projektpfade (script liegt im repo-root)
# ============================================================
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Visualizer"))
sys.path.insert(0, str(ROOT / "weights"))
# "model" Ordner brauchst du hier nicht zwingend, aber schadet nicht:
sys.path.insert(0, str(ROOT / "model"))

# ============================================================
# 4) CLI
# ============================================================
def env_bool(name: str, default: bool) -> bool:
    if name not in os.environ:
        return default
    return os.environ[name].lower() in ("1", "true", "yes", "y", "on")

WANDB_API_KEY = "97646993dfb3ed347361401308dd9377b8c7365a"

def wandb_init_if_enabled(args, cfg_dict: dict):
    # Wenn W&B aus: return None
    if args.no_wandb or (not args.wandb) or args.wandb_mode == "disabled":
        return None

    # Optional: wo W&B lokal schreibt (Cluster)
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir

    # online/offline aus args (oder ENV)
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_START_METHOD", "thread")

    # Login wie du es wolltest (CLI per subprocess)
    subprocess.run(["wandb", "login", WANDB_API_KEY], check=False)

    import wandb
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or None,
        tags=tags or None,
        config=cfg_dict,
        reinit=True,
    )
    return run


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--img-dir", type=str, default=os.getenv("IMG_DIR", ""), required=False,
                   help="Root folder mit Bildern (rekursiv). Beispiel: /data")
    p.add_argument("--out-dir", type=str, default=os.getenv("OUT_DIR", "weights/mim_pretrain"))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "10")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "64")))
    p.add_argument("--lr", type=float, default=float(os.getenv("LR", "1e-4")))
    p.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.05")))
    p.add_argument("--mask-ratio", type=float, default=float(os.getenv("MASK_RATIO", "0.05")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    p.add_argument("--save-every", type=int, default=int(os.getenv("SAVE_EVERY", "1")))

    p.add_argument("--train-pct", type=float, default=float(os.getenv("TRAIN_PCT", "0.8")))
    p.add_argument("--val-pct", type=float, default=float(os.getenv("VAL_PCT", "0.1")))
    p.add_argument("--test-pct", type=float, default=float(os.getenv("TEST_PCT", "0.1")))

    p.add_argument("--max-images", type=int, default=int(os.getenv("MAX_IMAGES", "0")),
                   help="Optional: limitiert Anzahl Images (0=alle)")

    # pretrained / random
    p.add_argument("--ckpt-path", type=str, default=os.getenv("CKPT_PATH", ""),
                   help="Pfad zu deren best_val.pth (oder leer = auto-find im weights/... tree)")
    p.add_argument("--use-pretrained", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE", "online"),
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "semicon"))
    p.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY", "frederic-voigt"))
    p.add_argument("--wandb-name", type=str, default=os.getenv("WANDB_NAME", ""))
    p.add_argument("--wandb-tags", type=str, default=os.getenv("WANDB_TAGS", ""))
    p.add_argument("--wandb-dir", type=str, default=os.getenv("WANDB_DIR", ""))

    # ------------------------------------------------------------
    # Multi-task pretraining
    # ------------------------------------------------------------
    p.add_argument(
        "--tasks",
        type=str,
        default=os.getenv("TASKS", "mim"),
        help="Komma-separiert: mim,simclr,rot  (z.B. 'mim,simclr,rot')",
    )

    # Manual loss weights (used when loss_balance=manual)
    p.add_argument("--w-mim", type=float, default=float(os.getenv("W_MIM", "1.0")))
    p.add_argument("--w-simclr", type=float, default=float(os.getenv("W_SIMCLR", "1.0")))
    p.add_argument("--w-rot", type=float, default=float(os.getenv("W_ROT", "1.0")))

    # Optional: automatic loss balancing (learned)
    p.add_argument(
        "--loss-balance",
        type=str,
        default=os.getenv("LOSS_BALANCE", "manual"),
        choices=["manual", "uncertainty"],
        help="manual = feste Gewichte w-*, uncertainty = lernbare Task-Balance",
    )

    # SimCLR
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMP", "0.2")))
    p.add_argument("--proj-dim", type=int, default=int(os.getenv("PROJ_DIM", "256")))

    # Rotation task
    p.add_argument(
        "--rot-angles",
        type=str,
        default=os.getenv("ROT_ANGLES", "0,90,180,270"),
        help="Komma-separiert, Vielfache von 90, z.B. '0,90,180,270'",
    )

    args = p.parse_args()

    if not args.img_dir:
        raise SystemExit("ERROR: --img-dir fehlt (oder ENV IMG_DIR setzen).")

    # pretrained default: True wenn nichts angegeben
    if not args.use_pretrained and not args.no_pretrained:
        up = env_bool("USE_PRETRAINED", True)
        args.use_pretrained = up
        args.no_pretrained = not up
    # wandb default: False wenn nichts angegeben
    if not args.wandb and not args.no_wandb:
        wb = env_bool("WANDB", False)
        args.wandb = wb
        args.no_wandb = not wb

    # sanity split
    s = args.train_pct + args.val_pct + args.test_pct
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"ERROR: train/val/test müssen zu 1.0 summieren, aktuell: {s}")

    return args

# ============================================================
# 5) Utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_checkpoint() -> Path:
    candidates = [
        ROOT / "weights" / "model" / "runs" / "FormalDatasetWindowedLinePair",
        ROOT / "weights" / "runs" / "FormalDatasetWindowedLinePair",
        ROOT / "runs" / "FormalDatasetWindowedLinePair",
    ]
    for base in candidates:
        if not base.exists():
            continue
        run_dirs = [p for p in base.iterdir() if p.is_dir()]
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for rd in run_dirs:
            for name in ["best_val.pth", "best_train.pth", "latest.pth"]:
                fp = rd / name
                if fp.exists():
                    return fp
    raise FileNotFoundError("Kein Checkpoint gefunden. Gib --ckpt-path an oder lege weights/... korrekt ab.")

def collect_images_recursively(root: Path, max_images: int = 0) -> list[Path]:
    """
    Robust & schnell auf Cluster: nutzt `find` statt pathlib.rglob/stat().
    """
    root = root.resolve()
    cmd = [
        "bash", "-lc",
        # -L NICHT verwenden (sonst folgt es symlinks evtl. in loops)
        f"find {sh_quote(str(root))} -type f \\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \\) -print"
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    paths = [Path(line) for line in out.splitlines() if line.strip()]
    paths.sort()
    if max_images and max_images > 0:
        paths = paths[:max_images]
    return paths

def sh_quote(s: str) -> str:
    # minimal shell-quoting
    return "'" + s.replace("'", "'\\''") + "'"


def split_paths(paths: list[Path], train_pct: float, val_pct: float, test_pct: float, seed: int):
    rnd = random.Random(seed)
    idx = list(range(len(paths)))
    rnd.shuffle(idx)
    n = len(paths)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    n_test = n - n_train - n_val
    train = [paths[i] for i in idx[:n_train]]
    val = [paths[i] for i in idx[n_train:n_train + n_val]]
    test = [paths[i] for i in idx[n_train + n_val:]]
    assert len(test) == n_test
    return train, val, test

def patchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    # x: (B,3,224,224) -> (B,196,768)
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    x = x.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = x.reshape(B, h * w, C * patch_size * patch_size)
    return patches

class MultiTaskPretrain(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        embed_dim: int,
        tasks: set[str],
        mask_ratio: float,
        proj_dim: int,
        temperature: float,
        rot_angles: list[int],
        loss_balance: str = "manual",
    ):
        super().__init__()
        self.base = base_model
        self.tasks = set(tasks)
        self.mask_ratio = float(mask_ratio)
        self.temperature = float(temperature)
        self.rot_angles = list(rot_angles)
        self.loss_balance = str(loss_balance)

        # shared encoder for simclr/rot (uses base under the hood)
        self.encoder = BaseEncoder(base_model)

        # --- MIM heads ---
        if "mim" in self.tasks:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            self.recon_head = nn.Linear(embed_dim, 3 * 16 * 16)

        # --- SimCLR heads ---
        if "simclr" in self.tasks:
            self.projector = MLPHead(embed_dim, proj_dim)

        # --- Rotation head ---
        if "rot" in self.tasks:
            self.rot_head = nn.Linear(embed_dim, len(self.rot_angles))

        # --- uncertainty weighting params (optional) ---
        if self.loss_balance == "uncertainty":
            # log_vars are learnable; one per active task
            self.log_vars = nn.ParameterDict()
            for t in sorted(self.tasks):
                self.log_vars[t] = nn.Parameter(torch.zeros(()))  # scalar

    def forward_mim(self, x_norm: torch.Tensor) -> torch.Tensor:
        B = x_norm.shape[0]
        x = x_norm.to(torch.float32)

        feat = self.base.cnn(x)
        tokens = self.base.cnn_postprocess(feat)     # (B,196,D)
        tokens = tokens + self.base.pos_embedding(tokens)
        N = tokens.shape[1]

        num_mask = int(self.mask_ratio * N)
        num_mask = max(1, min(N - 1, num_mask))

        mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            idx = torch.randperm(N, device=tokens.device)[:num_mask]
            mask[i, idx] = True

        tokens_masked = tokens.clone()
        tokens_masked[mask] = self.mask_token.expand(B, N, -1)[mask]

        enc = self.base.transformer(tokens_masked)   # (B,N,D)
        pred = self.recon_head(enc)                  # (B,N,768)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x_norm.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x_norm.device)[None, :, None, None]
        x_raw = (x_norm * std + mean).clamp(0, 1)

        target = patchify(x_raw, 16).to(pred.dtype)  # (B,N,768)
        return F.mse_loss(pred[mask], target[mask])

    def forward_simclr(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return simclr_nt_xent_loss(z1, z2, self.temperature)

    def forward_rot(self, x_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_rot, y = apply_random_rotations(x_base, self.rot_angles)
        h = self.encoder(x_rot)
        logits = self.rot_head(h)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def combine_losses(self, losses: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
        if self.loss_balance == "manual":
            total = 0.0
            for t, L in losses.items():
                total = total + float(weights.get(t, 1.0)) * L
            return total

        # uncertainty weighting (Kendall et al.-style):
        # total = sum( exp(-s_t)*L_t + s_t )
        total = 0.0
        for t, L in losses.items():
            s = self.log_vars[t]
            total = total + torch.exp(-s) * L + s
        return total

    def forward(self, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {}
        losses: dict[str, torch.Tensor] = {}

        if "mim" in self.tasks:
            L = self.forward_mim(batch["x_base"])
            losses["mim"] = L
            metrics["loss/mim"] = float(L.detach().cpu())

        if "simclr" in self.tasks:
            L = self.forward_simclr(batch["x1"], batch["x2"])
            losses["simclr"] = L
            metrics["loss/simclr"] = float(L.detach().cpu())

        if "rot" in self.tasks:
            L, acc = self.forward_rot(batch["x_base"])
            losses["rot"] = L
            metrics["loss/rot"] = float(L.detach().cpu())
            metrics["acc/rot"] = float(acc.detach().cpu())

        total = self.combine_losses(losses, weights)
        metrics["loss/total"] = float(total.detach().cpu())

        if self.loss_balance == "uncertainty":
            for t in sorted(self.tasks):
                metrics[f"balance/log_var_{t}"] = float(self.log_vars[t].detach().cpu())

        return total, metrics

@torch.no_grad()
def eval_multitask(model: MultiTaskPretrain, loader, device: str, weights: dict[str, float]) -> dict[str, float]:
    model.eval()
    agg = {}
    n = 0
    for batch in loader:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device, non_blocking=True)
        loss, metrics = model(batch, weights)
        # metrics already includes total + per-task
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1

    # mean
    for k in list(agg.keys()):
        agg[k] /= max(1, n)
    model.train()
    return agg

# ============================================================
# 6) RAM Dataset (lädt ALLE images einmal in RAM)
# ============================================================
class RamImages(torch.utils.data.Dataset):
    """
    Lädt Bilder einmalig in RAM (PIL RGB). __getitem__ gibt ein dict zurück:
      - immer: {"x_base": Tensor}
      - optional: {"x1": Tensor, "x2": Tensor} wenn simclr aktiv
    """
    def __init__(self, paths, tf_base, tf_simclr=None, enable_simclr=False,
                 wandb_run=None, log_every=2000, max_errors=200):
        self.paths = list(paths)
        self.tf_base = tf_base
        self.tf_simclr = tf_simclr
        self.enable_simclr = bool(enable_simclr)
        self.images = []
        self.bad = []
        self.wandb_run = wandb_run
        self.log_every = int(log_every)
        self.max_errors = int(max_errors)

        t0 = time.time()
        for i, p in enumerate(tqdm(self.paths, desc="RAM preload", unit="img"), start=1):
            try:
                with open(p, "rb") as f:
                    img = Image.open(f).convert("RGB")
                    self.images.append(img.copy())
            except Exception as e:
                self.bad.append((str(p), repr(e)))
                if len(self.bad) <= 5:
                    print(f"[RAM preload] skip broken: {p} | {e}")
                if len(self.bad) >= self.max_errors:
                    raise RuntimeError(f"Zu viele kaputte Bilder ({len(self.bad)}). Abbruch.") from e

            if (i % self.log_every) == 0:
                dt = time.time() - t0
                rate = i / max(dt, 1e-6)
                print(f"[RAM preload] loaded={i}/{len(self.paths)} | {rate:.2f} img/s | bad={len(self.bad)}")
                if self.wandb_run is not None:
                    try:
                        import wandb
                        wandb.log({
                            "load/loaded": i,
                            "load/total": len(self.paths),
                            "load/img_per_sec": rate,
                            "load/bad": len(self.bad),
                        })
                    except Exception:
                        pass

        dt = time.time() - t0
        print(f"[RAM preload] DONE: loaded={len(self.images)}/{len(self.paths)} | bad={len(self.bad)} | time={dt/60:.1f} min")

        if self.bad:
            try:
                Path("bad_images.txt").write_text("\n".join([f"{p}\t{err}" for p, err in self.bad]), encoding="utf-8")
                print("[RAM preload] wrote bad_images.txt")
            except Exception:
                pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        out = {"x_base": self.tf_base(img)}
        if self.enable_simclr:
            assert self.tf_simclr is not None
            x1, x2 = self.tf_simclr(img)
            out["x1"] = x1
            out["x2"] = x2
        return out


class BaseTransform:
    def __init__(self):
        self.tf = T.Compose([
            T.ToImage(),
            T.Resize((224, 224)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img)


class SimCLRAugment:
    """
    Two views for SimCLR. Wenn deine Daten eher monochrom/linienhaft sind,
    kannst du ColorJitter/Grayscale/Blur aggressiver oder schwächer machen.
    """
    def __init__(self):
        self.aug = T.Compose([
            T.ToImage(),
            T.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image):
        return self.aug(img), self.aug(img)


# ============================================================
# 7) MIM Model (nutzt euer CNN+Transformer encoder aus main.py)
# ============================================================
class MaskedPretrain(nn.Module):
    def __init__(self, base_model, embed_dim: int, mask_ratio: float = 0.75, patch_dim: int = 768):
        super().__init__()
        self.base = base_model
        self.mask_ratio = float(mask_ratio)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.recon_head = nn.Linear(embed_dim, patch_dim)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        B = x_norm.shape[0]

        # CNN -> tokens (B,196,dim)
        x = x_norm.to(torch.float32)
        feat = self.base.cnn(x)
        tokens = self.base.cnn_postprocess(feat)     # (B,196,dim)

        tokens = tokens + self.base.pos_embedding(tokens)
        N = tokens.shape[1]  # 196

        num_mask = int(self.mask_ratio * N)
        num_mask = max(1, min(N - 1, num_mask))

        mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            idx = torch.randperm(N, device=tokens.device)[:num_mask]
            mask[i, idx] = True

        tokens_masked = tokens.clone()
        tokens_masked[mask] = self.mask_token.expand(B, N, -1)[mask]

        enc = self.base.transformer(tokens_masked)   # (B,196,dim)
        pred = self.recon_head(enc)                  # (B,196,768)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x_norm.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x_norm.device)[None, :, None, None]
        x_raw = (x_norm * std + mean).clamp(0, 1)

        target = patchify(x_raw, 16).to(pred.dtype)
        loss = F.mse_loss(pred[mask], target[mask])
        return loss

class BaseEncoder(nn.Module):
    """
    Liefert globales Rep aus deinem base-model.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = x_norm.to(torch.float32)
        feat = self.base.cnn(x)
        tokens = self.base.cnn_postprocess(feat)      # (B,N,D)
        tokens = tokens + self.base.pos_embedding(tokens)
        enc = self.base.transformer(tokens)           # (B,N,D)
        return enc.mean(dim=1)                        # (B,D)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)

def simclr_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    B = z1.shape[0]
    assert B >= 2, "SimCLR braucht batch_size >= 2"

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B,D)

    sim = (z @ z.T) / temperature
    sim = sim.masked_fill(torch.eye(2 * B, device=z.device, dtype=torch.bool), -1e9)

    pos = torch.arange(B, device=z.device)
    targets = torch.cat([pos + B, pos], dim=0)  # positives: i<->i+B

    return F.cross_entropy(sim, targets)


def apply_random_rotations(x: torch.Tensor, angles: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    B = x.shape[0]
    K = len(angles)
    y = torch.randint(0, K, (B,), device=x.device)
    x_rot = x.clone()
    for k, deg in enumerate(angles):
        m = (y == k)
        if m.any():
            r = (deg // 90) % 4
            x_rot[m] = torch.rot90(x[m], k=r, dims=(-2, -1))
    return x_rot, y


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, encoder: BaseEncoder, embed_dim: int, proj_dim: int):
        super().__init__()
        self.encoder = encoder
        self.projector = MLPHead(embed_dim, proj_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return z1, z2


class RotationModel(nn.Module):
    def __init__(self, encoder: BaseEncoder, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.cls(h)


@torch.no_grad()
def eval_loss(mim_model, loader, device):
    mim_model.eval()
    total = 0.0
    n = 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        loss = mim_model(x)
        total += float(loss.detach().cpu())
        n += 1
    mim_model.train()
    return total / max(1, n)

# ============================================================
# 8) WandB (wie früher: subprocess login + wandb.init)
# ============================================================
def maybe_init_wandb(args, config_dict: dict, out_dir: Path):
    if args.no_wandb or args.wandb_mode == "disabled":
        return None

    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir
    else:
        # default: in out_dir schreiben (damit writable im container)
        os.environ["WANDB_DIR"] = str(out_dir / "wandb_logs")

    os.environ["WANDB_MODE"] = args.wandb_mode  # online/offline
    os.environ.setdefault("WANDB_START_METHOD", "thread")

    # login wie bei dir früher
    subprocess.run(["wandb", "login", WANDB_API_KEY], check=False)

    import wandb
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or None,
        tags=tags or None,
        config=config_dict,
        reinit=True,
    )
    return run

# ============================================================
# 9) MAIN
# ============================================================
def main():
    args = get_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tasks = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}
    valid = {"mim", "simclr", "rot"}
    if not tasks.issubset(valid) or len(tasks) == 0:
        raise SystemExit(f"ERROR: --tasks ungültig: {args.tasks} (erlaubt: mim,simclr,rot)")

    # Rotation angles
    rot_angles = [int(a.strip()) for a in args.rot_angles.split(",") if a.strip()]
    if "rot" in tasks:
        if (not rot_angles) or any(a % 90 != 0 for a in rot_angles):
            raise SystemExit(f"ERROR: --rot-angles ungültig: {args.rot_angles} (nur Vielfache von 90)")

    tf_base = BaseTransform()
    tf_simclr = SimCLRAugment() if "simclr" in tasks else None


    # --- W&B run (oder None) MUSS VOR dem RAM-Laden existieren ---
    run = None
    use_wandb = (args.wandb and not args.no_wandb and args.wandb_mode != "disabled")
    if use_wandb:
        cfg_dict_for_wandb = vars(args).copy()
        run = wandb_init_if_enabled(args, cfg_dict_for_wandb)
        # (optional) wenn du summary setzen willst:
        try:
            import wandb
            wandb.summary["host"] = os.uname().nodename if hasattr(os, "uname") else "unknown"
        except Exception:
            pass

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Bilder sammeln ----
    img_root = Path(args.img_dir).resolve()
    paths = collect_images_recursively(img_root, max_images=args.max_images)
    if not paths:
        raise SystemExit(f"Keine Bilder gefunden unter: {img_root}")
    print(f"Found {len(paths)} images under: {img_root}")

    train_paths, val_paths, test_paths = split_paths(paths, args.train_pct, args.val_pct, args.test_pct, args.seed)
    print(f"Split: train={len(train_paths)} | val={len(val_paths)} | test={len(test_paths)}")

    # ---- Transform (passt zu eurem CNN Preproc: ImageNet normalize) ----
    tf = T.Compose([
        T.ToImage(),
        T.Resize((224, 224)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- ALLES in RAM laden ----
    print("Loading images into RAM ...")
    print("Loading images into RAM ...")
    train_ds = RamImages(train_paths, tf_base, tf_simclr=tf_simclr, enable_simclr=("simclr" in tasks),
                         wandb_run=run, log_every=2000)
    val_ds = RamImages(val_paths, tf_base, tf_simclr=tf_simclr, enable_simclr=("simclr" in tasks),
                       wandb_run=run, log_every=2000)
    test_ds = RamImages(test_paths, tf_base, tf_simclr=tf_simclr, enable_simclr=("simclr" in tasks),
                        wandb_run=run, log_every=2000)


    print("RAM load done.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False) if test_ds else None

    # ---- Euer Originalmodell bauen & ggf. weights laden ----
    # WICHTIG: erst NACH main_config patch importieren:
    # --- Import original config + main ---
    import main_config as config
    import main as main_module

    # wir wollen CNN-style fürs MIM (sonst passt Tokenizer nicht)
    config.MODEL_STYLE = "cnn"

    ckpt_path_used = ""
    sd = None

    if args.use_pretrained and not args.no_pretrained:
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else find_checkpoint()
        ckpt_path_used = str(ckpt_path)
        print("Using pretrained checkpoint:", ckpt_path_used)

        ckpt = torch.load(ckpt_path_used, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # --- WICHTIG: RESULT_NUM aus checkpoint übernehmen (damit decoder_query passt) ---
        if "decoder_query.weight" in sd:
            config.RESULT_NUM = int(sd["decoder_query.weight"].shape[0])
            print("Checkpoint RESULT_NUM =", config.RESULT_NUM)

    # jetzt erst Modell bauen (mit korrektem config.RESULT_NUM)
    base = main_module.create_model()

    if sd is not None:
        base.load_state_dict(sd, strict=True)
    else:
        print("No-pretrained: random init (kein ckpt geladen)")

    base = base.to(device)

    embed_dim = int(base.decoder_query.weight.shape[1])  # meist 256

    multitask = MultiTaskPretrain(
        base_model=base,
        embed_dim=embed_dim,
        tasks=tasks,
        mask_ratio=args.mask_ratio,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        rot_angles=rot_angles,
        loss_balance=args.loss_balance,
    ).to(device)

    opt = torch.optim.AdamW(multitask.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_weights = {"mim": args.w_mim, "simclr": args.w_simclr, "rot": args.w_rot}


    # ---- logging config ----
    cfg_dict = {
        "img_dir": str(img_root),
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "train_pct": args.train_pct,
        "val_pct": args.val_pct,
        "test_pct": args.test_pct,
        "use_pretrained": bool(args.use_pretrained and not args.no_pretrained),
        "pretrained_ckpt": ckpt_path_used,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    run = maybe_init_wandb(args, cfg_dict, out_dir)
    if run is not None:
        import wandb
        wandb.summary["pretrained_ckpt"] = ckpt_path_used

    # ---- train loop ----
    multitask.train()
    global_step = 0


    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        total = 0.0
        n = 0
        last_metrics = {}

        for batch in train_loader:
            # move to device
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device, non_blocking=True)

            loss, metrics = multitask(batch, loss_weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(metrics["loss/total"])
            last_metrics = metrics
            n += 1

        train_total = total / max(1, n)
        dt = time.time() - t0

        val_metrics = eval_multitask(multitask, val_loader, device, loss_weights) if val_loader is not None else None
        test_metrics = eval_multitask(multitask, test_loader, device, loss_weights) if test_loader is not None else None

        # console log
        msg = f"Epoch {ep}/{args.epochs} | train_total={train_total:.6f} | time={dt:.1f}s"
        print(msg)
        # print per-task quick view from last batch
        print("  Train batch:", {k: round(v, 6) for k, v in last_metrics.items() if k.startswith("loss/") or k.startswith("acc/")})

        if val_metrics is not None:
            print("  Val:", {k: round(v, 6) for k, v in val_metrics.items() if k.startswith("loss/") or k.startswith("acc/")})
        if test_metrics is not None:
            print("  Test:", {k: round(v, 6) for k, v in test_metrics.items() if k.startswith("loss/") or k.startswith("acc/")})

        # wandb
        if run is not None:
            import wandb
            log = {
                "epoch": ep,
                "epoch_time_sec": dt,
                "loss/train_total": train_total,
                "tasks/active": ",".join(sorted(tasks)),
                "loss_balance/mode": args.loss_balance,
                "weights/w_mim": args.w_mim,
                "weights/w_simclr": args.w_simclr,
                "weights/w_rot": args.w_rot,
            }
            # last train batch metrics
            log.update({f"train/{k}": v for k, v in last_metrics.items()})
            if val_metrics is not None:
                log.update({f"val/{k}": v for k, v in val_metrics.items()})
            if test_metrics is not None:
                log.update({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.log(log, step=ep)

        # saving
        if args.save_every > 0 and ep % args.save_every == 0:
            save_path = out_dir / f"multitask_ep{ep}.pth"
            payload = {
                "base_model": base.state_dict(),
                "cfg": cfg_dict,
                "timestamp": int(time.time()),
                "tasks": sorted(tasks),
                "loss_balance": args.loss_balance,
                "loss_weights": loss_weights,
            }
            # task-specific heads
            if "mim" in tasks:
                payload["mask_token"] = multitask.mask_token.detach().cpu()
                payload["recon_head"] = multitask.recon_head.state_dict()
            if "simclr" in tasks:
                payload["projector"] = multitask.projector.state_dict()
                payload["temperature"] = args.temperature
                payload["proj_dim"] = args.proj_dim
            if "rot" in tasks:
                payload["rot_head"] = multitask.rot_head.state_dict()
                payload["rot_angles"] = rot_angles

            if args.loss_balance == "uncertainty":
                payload["log_vars"] = {t: float(multitask.log_vars[t].detach().cpu()) for t in multitask.log_vars}

            torch.save(payload, save_path)
            print("Saved:", save_path)

    if run is not None:
        run.finish()

    print("Done.")

if __name__ == "__main__":
    main()
