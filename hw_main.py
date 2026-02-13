from pathlib import Path
import sys
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T
from PIL import Image


# ---------------------------
# PATHS (wie bei dir in hw_main)
# ---------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Visualizer"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "weights"))


# ---------------------------
# Checkpoint Finder
# ---------------------------
def find_checkpoint():
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
    raise FileNotFoundError("Kein Checkpoint gefunden unter weights/.../FormalDatasetWindowedLinePair/<run>/*.pth")


# ---------------------------
# Dataset (nur Bilder, keine Labels)
# ---------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: Path):
        self.img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if not self.img_paths:
            raise FileNotFoundError(f"Keine Bilder gefunden in {img_dir}")

        # gleiche Normalize wie in eurem xtransform
        self.tf = T.Compose([
            T.ToImage(),
            T.Resize((224, 224)),
            T.ToDtype(torch.float32, scale=True),  # -> [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)  # (3,224,224), normalized
        return x


# ---------------------------
# Patchify / Unpatchify Helpers
# ---------------------------
def patchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    x: (B,3,224,224) -> patches: (B,196,patch_dim) with patch_dim=3*16*16=768
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    # (B,C,H,W) -> (B,h,w,C,p,p) -> (B,h*w,C*p*p)
    x = x.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = x.reshape(B, h * w, C * patch_size * patch_size)
    return patches


# ---------------------------
# MIM Wrapper um euer Model (cnn-style)
# ---------------------------
class MaskedPretrain(nn.Module):
    def __init__(self, base_model, embed_dim: int, patch_dim: int = 768, mask_ratio: float = 0.75):
        super().__init__()
        self.base = base_model
        self.mask_ratio = mask_ratio

        # learnable mask token in token space
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # reconstruction head: token -> patch pixels (normalized space)
        self.recon_head = nn.Linear(embed_dim, patch_dim)

    def forward(self, x_norm: torch.Tensor):
        """
        x_norm: (B,3,224,224) normalized
        returns loss
        """
        B = x_norm.shape[0]

        # --- erzeugt Token-Grid wie in eurem forward(cnn) ---
        # ResNet -> (B,256,14,14) -> rearrange+linear -> (B,196,dim)
        x = x_norm.to(torch.float32)
        x = self.base.cnn(x)
        tokens = self.base.cnn_postprocess(x)  # (B,196,dim)

        # positional encoding + transformer encoder
        tokens = tokens + self.base.pos_embedding(tokens)

        # --- Masking in token space ---
        N = tokens.shape[1]  # 196
        num_mask = int(self.mask_ratio * N)
        mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)

        for i in range(B):
            idx = torch.randperm(N, device=tokens.device)[:num_mask]
            mask[i, idx] = True

        # replace masked tokens with mask_token
        tokens_masked = tokens.clone()
        tokens_masked[mask] = self.mask_token.expand(B, N, -1)[mask]

        # encoder
        enc = self.base.transformer(tokens_masked)  # (B,196,dim)

        # reconstruct patches
        pred = self.recon_head(enc)  # (B,196,768)

        # target patches = patchify on input (normalized space, damit alles konsistent ist)
        target = patchify(x_norm, patch_size=16).to(pred.dtype)

        # loss only on masked patches
        loss = F.mse_loss(pred[mask], target[mask])
        return loss


# ---------------------------
# Main Training
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # HIER deine neuen Daten setzen:
    NEW_IMG_DIR = ROOT / "hw_images"   # <-- anpassen
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    ckpt_path = find_checkpoint()
    print("Using checkpoint:", ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # importiere eure config und main/create_model
    import main_config as config
    import main as main_module

    # WICHTIG: wir wollen das gleiche Backbone-Setup wie der Checkpoint
    # (dein Checkpoint hatte RESULT_NUM=10; für MIM ist das nicht kritisch,
    # aber wir halten es konsistent, damit state_dict sauber passt)
    if "decoder_query.weight" in sd:
        config.RESULT_NUM = int(sd["decoder_query.weight"].shape[0])

    # für MIM nutzen wir den CNN-Style-Encoder (ResNet->Tokens->Transformer)
    config.MODEL_STYLE = "cnn"

    base = main_module.create_model()

    # falls RESULT_NUM mismatch war, decoder_query patchen (wie bei dir)
    if "decoder_query.weight" in sd:
        exp_q, exp_dim = sd["decoder_query.weight"].shape
        cur_q, cur_dim = base.decoder_query.weight.shape
        if (cur_q, cur_dim) != (exp_q, exp_dim):
            new_w = nn.Parameter(torch.empty(exp_q, exp_dim))
            nn.init.normal_(new_w, std=0.02)
            base.decoder_query.weight = new_w

    # weights laden
    base.load_state_dict(sd, strict=True)
    base = base.to(device)

    # embed_dim: bei euch dim = head_num*dim_head = config.NUM_HEADS*config.EMBED_DIM
    embed_dim = base.decoder_query.weight.shape[1]  # 256 bei dir
    mim = MaskedPretrain(base, embed_dim=embed_dim, patch_dim=3*16*16, mask_ratio=0.75).to(device)

    # Dataset / Loader
    ds = ImageFolderDataset(NEW_IMG_DIR)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer: klein anfangen, weil du weiter-vortrainierst
    opt = torch.optim.AdamW(mim.parameters(), lr=1e-4, weight_decay=0.05)

    mim.train()
    EPOCHS = 10

    for ep in range(EPOCHS):
        running = 0.0
        for step, x in enumerate(dl, 1):
            x = x.to(device, non_blocking=True)
            loss = mim(x)

            opt.zero_grad(set_to_none=True)
            print(loss)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu())
            if step % 50 == 0:
                print(f"ep {ep+1}/{EPOCHS} step {step} loss {running/50:.5f}")
                running = 0.0

        # save each epoch
        out = ROOT / "weights" / "mim_pretrain"
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / f"mim_ep{ep+1}.pth"
        torch.save({
            "base_model": base.state_dict(),      # das willst du später fürs Fine-Tuning nehmen
            "mask_token": mim.mask_token.detach().cpu(),
            "recon_head": mim.recon_head.state_dict(),
            "config_RESULT_NUM": getattr(config, "RESULT_NUM", None),
            "config_MODEL_STYLE": getattr(config, "MODEL_STYLE", None),
        }, save_path)
        print("saved:", save_path)

    print("Done.")


if __name__ == "__main__":
    main()
