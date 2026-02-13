from __future__ import annotations
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# 0) ABSOLUT WICHTIG: main_config patchen BEVOR irgendwas importiert,
#    was main_config/main.py ziehen könnte.
# ============================================================
import types, sys


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
    p.add_argument("--mask-ratio", type=float, default=float(os.getenv("MASK_RATIO", "0.75")))
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

# ============================================================
# 6) RAM Dataset (lädt ALLE images einmal in RAM)
# ============================================================
class RamImages(torch.utils.data.Dataset):
    """
    Lädt Bilder einmalig in RAM (als uint8 RGB), gibt bei __getitem__ Tensor nach Transform aus.
    Loggt Fortschritt während dem Laden + optional wandb.
    """
    def __init__(self, paths, tf, wandb_run=None, log_every=2000, max_errors=200):
        self.paths = list(paths)
        self.tf = tf
        self.images = []
        self.bad = []
        self.wandb_run = wandb_run
        self.log_every = int(log_every)
        self.max_errors = int(max_errors)

        t0 = time.time()
        last_log_t = t0

        # tqdm Progressbar
        for i, p in enumerate(tqdm(self.paths, desc="RAM preload", unit="img"), start=1):
            try:
                # robustes Öffnen
                with open(p, "rb") as f:
                    img = Image.open(f).convert("RGB")
                    # als bytes im RAM (PIL Image) behalten; alternativ np.uint8 speichern
                    self.images.append(img.copy())
            except Exception as e:
                self.bad.append((str(p), repr(e)))
                if len(self.bad) <= 5:
                    print(f"[RAM preload] skip broken: {p} | {e}")
                if len(self.bad) >= self.max_errors:
                    raise RuntimeError(f"Zu viele kaputte Bilder ({len(self.bad)}). Abbruch.") from e
                continue

            # periodisches Logging
            if (i % self.log_every) == 0:
                now = time.time()
                dt = now - t0
                rate = i / max(dt, 1e-6)
                eta = (len(self.paths) - i) / max(rate, 1e-6)
                msg = f"[RAM preload] loaded={i}/{len(self.paths)} | {rate:.2f} img/s | ETA {eta/60:.1f} min | bad={len(self.bad)}"
                print(msg)

                if self.wandb_run is not None:
                    try:
                        import wandb
                        wandb.log({
                            "load/loaded": i,
                            "load/total": len(self.paths),
                            "load/img_per_sec": rate,
                            "load/eta_min": eta / 60.0,
                            "load/bad": len(self.bad),
                        })
                    except Exception:
                        pass

                last_log_t = now

        # final summary
        dt = time.time() - t0
        print(f"[RAM preload] DONE: loaded={len(self.images)}/{len(self.paths)} | bad={len(self.bad)} | time={dt/60:.1f} min")

        # falls du die bad-files als text speichern willst:
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
        x = self.tf(img)  # Tensor
        return x


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

        target = patchify(x_norm, patch_size=16).to(pred.dtype)
        loss = F.mse_loss(pred[mask], target[mask])
        return loss

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
    train_ds = RamImages(train_paths, tf, wandb_run=run, log_every=2000)
    val_ds = RamImages(val_paths, tf, wandb_run=run, log_every=2000)
    test_ds = RamImages(test_paths, tf, wandb_run=run, log_every=2000)

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
    mim = MaskedPretrain(base, embed_dim=embed_dim, mask_ratio=args.mask_ratio, patch_dim=3 * 16 * 16).to(device)

    opt = torch.optim.AdamW(mim.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    mim.train()
    global_step = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        total = 0.0
        n = 0

        for x in train_loader:
            global_step += 1
            x = x.to(device, non_blocking=True)
            loss = mim(x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())
            n += 1

        train_loss = total / max(1, n)
        dt = time.time() - t0

        val_loss = eval_loss(mim, val_loader, device) if val_loader is not None else None
        test_loss = eval_loss(mim, test_loader, device) if test_loader is not None else None

        print(f"Epoch {ep}/{args.epochs} | train={train_loss:.6f} | time={dt:.1f}s")
        if val_loss is not None:
            print(f"  Val loss:  {val_loss:.6f}")
        if test_loss is not None:
            print(f"  Test loss: {test_loss:.6f}")

        if run is not None:
            import wandb
            wandb.log(
                {
                    "epoch": ep,
                    "loss/train_epoch": train_loss,
                    "loss/val_epoch": val_loss,
                    "loss/test_epoch": test_loss,
                    "epoch_time_sec": dt,
                },
                step=ep,
            )

        if args.save_every > 0 and ep % args.save_every == 0:
            save_path = out_dir / f"mim_ep{ep}.pth"
            torch.save(
                {
                    "base_model": base.state_dict(),  # <-- das willst du später fürs Finetuning
                    "mask_token": mim.mask_token.detach().cpu(),
                    "recon_head": mim.recon_head.state_dict(),
                    "cfg": cfg_dict,
                    "timestamp": int(time.time()),
                },
                save_path,
            )
            print("Saved:", save_path)

    if run is not None:
        run.finish()

    print("Done.")

if __name__ == "__main__":
    main()
