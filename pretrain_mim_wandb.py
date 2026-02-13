from __future__ import annotations

import os
import sys
import time
import json
import argparse
import random
import subprocess
from pathlib import Path
from data_ram_loader import build_ram_dataloaders_split
import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T
from PIL import Image


# =========================
# 0) HARD-CODED WANDB KEY (wie du willst)
# =========================
WANDB_API_KEY = "97646993dfb3ed347361401308dd9377b8c7365a"


# =========================
# 1) PATH SETUP (Projektstruktur)
#    Script liegt im Projekt-Root (neben main.py / main_config.py)
# =========================
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Visualizer"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "weights"))


# =========================
# 2) CLI + ENV Overrides (Docker-freundlich)
# =========================
def env_bool(name: str, default: bool) -> bool:
    if name not in os.environ:
        return default
    return os.environ[name].lower() in ("1", "true", "yes", "y", "on")


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train-pct", type=float, default=float(os.getenv("TRAIN_PCT", "0.8")))
    p.add_argument("--val-pct", type=float, default=float(os.getenv("VAL_PCT", "0.1")))
    p.add_argument("--test-pct", type=float, default=float(os.getenv("TEST_PCT", "0.1")))

    # Daten & Training
    p.add_argument("--img-dir", type=str, default=os.getenv("IMG_DIR", ""), help="Folder mit Bildern")
    p.add_argument("--out-dir", type=str, default=os.getenv("OUT_DIR", "weights/mim_pretrain"))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "10")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "64")))
    p.add_argument("--lr", type=float, default=float(os.getenv("LR", "1e-4")))
    p.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.05")))
    p.add_argument("--mask-ratio", type=float, default=float(os.getenv("MASK_RATIO", "0.75")))
    p.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "4")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    p.add_argument("--save-every", type=int, default=int(os.getenv("SAVE_EVERY", "1")))

    # Pretrained vs Random
    p.add_argument("--use-pretrained", action="store_true", help="Original-Weights laden")
    p.add_argument("--no-pretrained", action="store_true", help="Random init (keine Weights laden)")
    p.add_argument("--ckpt-path", type=str, default=os.getenv("CKPT_PATH", ""), help="Optional: expliziter .pth Pfad")

    # W&B an/aus
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", action="store_true")

    # W&B parameter (alles auch über ENV)
    p.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "semicon"))
    p.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY", "frederic-voigt"))
    p.add_argument("--wandb-name", type=str, default=os.getenv("WANDB_NAME", ""))
    p.add_argument("--wandb-tags", type=str, default=os.getenv("WANDB_TAGS", ""))  # comma separated
    p.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE", "online"),
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-dir", type=str, default=os.getenv("WANDB_DIR", ""))

    args = p.parse_args()

    # img-dir Pflicht
    if not args.img_dir:
        raise SystemExit("ERROR: img-dir fehlt. Setze --img-dir oder ENV IMG_DIR=/path/to/images")

    # use_pretrained via CLI + ENV kompatibel
    # Default: True, wenn weder --use-pretrained noch --no-pretrained gesetzt
    if not args.use_pretrained and not args.no_pretrained:
        # ENV USE_PRETRAINED überschreibt default
        use_pretrained_env = env_bool("USE_PRETRAINED", True)
        args.use_pretrained = use_pretrained_env
        args.no_pretrained = not use_pretrained_env
    else:
        # explizite CLI gewinnt, aber wenn ENV gesetzt ist, lassen wir ENV gewinnen (wie Docker-override)
        if "USE_PRETRAINED" in os.environ:
            use_pretrained_env = env_bool("USE_PRETRAINED", True)
            args.use_pretrained = use_pretrained_env
            args.no_pretrained = not use_pretrained_env

    # wandb via CLI + ENV kompatibel
    if not args.wandb and not args.no_wandb:
        args.wandb = env_bool("WANDB", False)
        args.no_wandb = not args.wandb
    else:
        if "WANDB" in os.environ:
            args.wandb = env_bool("WANDB", False)
            args.no_wandb = not args.wandb

    return args


# =========================
# 3) Helpers
# =========================
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
    raise FileNotFoundError(
        "Kein Checkpoint gefunden. Erwartet z.B. weights/.../FormalDatasetWindowedLinePair/<run>/best_val.pth"
    )


def patchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    x: (B,3,224,224) -> (B,196,768)
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    x = x.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = x.reshape(B, h * w, C * patch_size * patch_size)
    return patches


# =========================
# 4) Dataset: nur Bilder
# =========================
class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: Path):
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
        if not self.img_paths:
            raise FileNotFoundError(f"Keine Bilder gefunden in: {img_dir}")

        self.tf = T.Compose([
            T.ToImage(),
            T.Resize((224, 224)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)  # (3,224,224)
        return x


# =========================
# 5) MIM Wrapper: nutzt euren CNN-Tokenizer + Transformer Encoder
# =========================
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
        x = self.base.cnn(x)
        tokens = self.base.cnn_postprocess(x)  # (B,196,dim)

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

        enc = self.base.transformer(tokens_masked)  # (B,196,dim)
        pred = self.recon_head(enc)  # (B,196,768)

        target = patchify(x_norm, patch_size=16).to(pred.dtype)
        loss = F.mse_loss(pred[mask], target[mask])
        return loss


# =========================
# 6) W&B: minimal wie früher (Flag -> login -> run oder None)
# =========================
def wandb_init_if_enabled(args, cfg_dict: dict):
    if args.no_wandb:
        return None

    if args.wandb_mode == "disabled":
        return None

    # optional: wo gespeichert wird (Cluster)
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)  # online/offline
    os.environ.setdefault("WANDB_START_METHOD", "thread")

    # Login hardcoded
    subprocess.run(["wandb", "login", WANDB_API_KEY], check=True)

    import wandb
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or None,
        tags=tags or None,
        config=cfg_dict,
        reinit=True,
    )
    return run

@torch.no_grad()
def run_eval(mim_model, iterable, device):
    mim_model.eval()
    total = 0.0
    n = 0
    for x in iterable:
        loss = mim_model(x)   # x ist schon auf GPU wenn cuda
        total += float(loss.detach().cpu())
        n += 1
    mim_model.train()
    return total / max(1, n)
# =========================
# 7) MAIN
# =========================
def main():
    args = get_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    run = None
    use_wandb = args.wandb and args.wandb_mode != "disabled"

    if use_wandb:
        # Hardcoded key (wie du wolltest)
        WANDB_KEY = "97646993dfb3ed347361401308dd9377b8c7365a"

        # erzwingt online/offline (sonst übernimmt W&B evtl. alte env)
        os.environ["WANDB_MODE"] = args.wandb_mode

        # Login wie früher (CLI)
        subprocess.run(["wandb", "login", WANDB_KEY], check=False)

        import wandb
        # optional: neue backend-api (kannst du drin lassen oder weglassen)
        # wandb.require("core")

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            # mode=args.wandb_mode,  # optional zusätzlich
        )

        # falls du eine run-id irgendwo brauchst:
        # print("W&B run id:", run.id)

    # --- Import euer Modell ---
    import main_config as config
    import main as main_module

    # MIM nutzt CNN-style (ResNet50 -> 14x14 tokens -> transformer encoder)
    config.MODEL_STYLE = "cnn"

    base = main_module.create_model()

    # --- Optional: pretrained weights laden ---
    ckpt_path_used = ""
    if args.use_pretrained and not args.no_pretrained:
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else find_checkpoint()
        ckpt_path_used = str(ckpt_path)
        print("Using pretrained checkpoint:", ckpt_path)

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # result_num aus checkpoint übernehmen (damit decoder_query passt)
        if "decoder_query.weight" in sd:
            exp_q, exp_dim = sd["decoder_query.weight"].shape
            config.RESULT_NUM = int(exp_q)

            # patch decoder_query falls create_model() anders initialisiert ist
            cur_q, cur_dim = base.decoder_query.weight.shape
            if (cur_q, cur_dim) != (exp_q, exp_dim):
                new_w = nn.Parameter(torch.empty(exp_q, exp_dim))
                nn.init.normal_(new_w, std=0.02)
                base.decoder_query.weight = new_w

        base.load_state_dict(sd, strict=True)
    else:
        print("No-pretrained: random init (kein ckpt geladen)")

    base = base.to(device)

    # --- MIM Model ---
    embed_dim = int(base.decoder_query.weight.shape[1])  # meist 256
    mim = MaskedPretrain(base, embed_dim=embed_dim, mask_ratio=args.mask_ratio, patch_dim=3 * 16 * 16).to(device)

    # --- Data (RAM + deterministic split + GPU prefetch) ---
    split_bundle = build_ram_dataloaders_split(
        root=args.img_dir,
        batch_size=args.batch_size,
        device=torch.device(device),
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,  # <- garantiert gleiche Splits
        img_size=(224, 224),
        shuffle_train=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        verbose=True,
    )

    train_iter = split_bundle.prefetchers["train"]
    val_iter = split_bundle.prefetchers.get("val", None)
    test_iter = split_bundle.prefetchers.get("test", None)

    # --- Output dir ---
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = {
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mask_ratio": args.mask_ratio,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "use_pretrained": bool(args.use_pretrained and not args.no_pretrained),
        "ckpt_path": ckpt_path_used,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "wandb_tags": args.wandb_tags,
        "wandb_mode": args.wandb_mode,
        "wandb_dir": args.wandb_dir,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    # --- W&B run (oder None) ---
    run = wandb_init_if_enabled(args, cfg_dict)
    if run is not None:
        import wandb
        wandb.summary["pretrained_ckpt"] = ckpt_path_used

    # --- Optimizer ---
    opt = torch.optim.AdamW(mim.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    mim.train()

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        ep_loss = 0.0
        n = 0

        for x in train_iter:
            global_step += 1
            x = x.to(device, non_blocking=True)

            loss = mim(x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            l = float(loss.detach().cpu())
            ep_loss += l
            n += 1

            # W&B step logging (sparsam)
            if run is not None and global_step % 50 == 0:
                import wandb
                wandb.log({"loss/step": l, "lr": opt.param_groups[0]["lr"]}, step=global_step)

        mean_loss = ep_loss / max(1, n)
        dt = time.time() - t0
        print(f"Epoch {ep}/{args.epochs} | loss={mean_loss:.6f} | time={dt:.1f}s")

        val_loss = None
        test_loss = None

        if val_iter is not None:
            val_loss = run_eval(mim, val_iter, device)

        if test_iter is not None:
            test_loss = run_eval(mim, test_iter, device)

        if val_loss is not None:
            print(f"  Val loss:  {val_loss:.6f}")
        if test_loss is not None:
            print(f"  Test loss: {test_loss:.6f}")

        if run is not None:
            import wandb
            wandb.log(
                {
                    "epoch": ep,
                    "loss/train_epoch": mean_loss,
                    "loss/val_epoch": val_loss,
                    "loss/test_epoch": test_loss,
                    "epoch_time_sec": dt,
                },
                step=ep,
            )

        if run is not None:
            import wandb
            wandb.log({"loss/epoch": mean_loss, "epoch_time_sec": dt, "epoch": ep}, step=global_step)

        # Save
        if args.save_every > 0 and ep % args.save_every == 0:
            save_path = out_dir / f"mim_ep{ep}.pth"
            torch.save(
                {
                    "base_model": base.state_dict(),          # <- das nimmst du später fürs Fine-Tuning
                    "mask_token": mim.mask_token.detach().cpu(),
                    "recon_head": mim.recon_head.state_dict(),
                    "global_step": global_step,
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
