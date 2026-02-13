"""
build_dataset.py

Generates a dataset:
- entities.json
- triples_gt.jsonl
- triples_obs.jsonl
- triples_text.txt
- meta.json
Optionally:
- image.png
- render_meta.json

Modes:
- render_mode=clean: no image degradation; labels still can be from obs if you choose
- render_mode=messy: labels from obs + optional mild jpeg/blur (still no scribbles/occlusions)

Depends on:
  core_ontology.py
  graph_gen.py
  render_drawioish.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from graph_gen import GenConfig, generate_sample
from render_drawioish import RenderConfig, render_sample


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_text(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")

def entity_to_dict(e) -> Dict[str, Any]:
    return {"id": e.id, "type": e.type.value, "subtype": e.subtype, "attrs": e.attrs}

def triple_to_dict(t) -> Dict[str, Any]:
    return {"id": t.id, "s": t.s, "p": t.p, "o": t.o}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate synthetic triple dataset (GT+OBS) with optional images.")

    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--prefix", type=str, default="sample_")
    ap.add_argument("--start_index", type=int, default=1)
    ap.add_argument("--digits", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write_index", action="store_true")

    # generator knobs
    ap.add_argument("--motifs_min", type=int, default=2)
    ap.add_argument("--motifs_max", type=int, default=5)
    ap.add_argument("--w_diffpair", type=float, default=1.0)
    ap.add_argument("--w_mirror", type=float, default=1.0)
    ap.add_argument("--w_rc", type=float, default=0.7)
    ap.add_argument("--w_invchain", type=float, default=0.5)

    # observed label noise (the main "real-world mess")
    ap.add_argument("--p_label_drop", type=float, default=0.35)
    ap.add_argument("--p_label_wrong_attach", type=float, default=0.10)
    ap.add_argument("--p_duplicate_labels", type=float, default=0.05)
    ap.add_argument("--p_typo_in_labels", type=float, default=0.10)

    # meta/reification (optional)
    ap.add_argument("--p_add_confidence_meta", type=float, default=0.25)
    ap.add_argument("--confidence_min", type=float, default=0.55)
    ap.add_argument("--confidence_max", type=float, default=0.98)

    # text
    ap.add_argument("--text_view", choices=["gt", "obs"], default="obs")
    ap.add_argument("--text_variants_per_triple", type=int, default=1)

    # rendering
    ap.add_argument("--with_images", action="store_true")
    ap.add_argument("--render_mode", choices=["clean", "messy"], default="clean")
    ap.add_argument("--img_view_for_labels", choices=["gt", "obs"], default="obs")
    ap.add_argument("--img_w", type=int, default=1200)
    ap.add_argument("--img_h", type=int, default=800)

    # optional mild image degradation (only used in messy mode unless you force it)
    ap.add_argument("--enable_mild_image_noise", action="store_true",
                    help="Enable mild jpeg/blur even in clean mode (not recommended).")

    return ap


def cfg_from_args(a: argparse.Namespace) -> GenConfig:
    return GenConfig(
        n_motifs_min=a.motifs_min,
        n_motifs_max=a.motifs_max,
        w_diffpair=a.w_diffpair,
        w_current_mirror=a.w_mirror,
        w_rc_filter=a.w_rc,
        w_inverter_chain=a.w_invchain,

        p_label_drop=a.p_label_drop,
        p_label_wrong_attach=a.p_label_wrong_attach,
        p_duplicate_labels=a.p_duplicate_labels,
        p_typo_in_labels=a.p_typo_in_labels,

        p_add_confidence_meta=a.p_add_confidence_meta,
        confidence_range=(a.confidence_min, a.confidence_max),

        textualize_view=a.text_view,
        n_text_variants_per_triple=a.text_variants_per_triple,
    )


def render_cfg_from_args(a: argparse.Namespace) -> RenderConfig:
    cfg = RenderConfig(width=a.img_w, height=a.img_h)

    # always: clean layout + fit-to-canvas ON
    cfg.fit_to_canvas = True
    cfg.p_missing_junction_dot = 0.10

    if a.render_mode == "clean":
        cfg.do_messify = False
        cfg.p_extra_bend = 0.20
        cfg.detour_strength = 0.20
        cfg.jitter_px = 8
    else:
        # "messy" without overdoing: slightly more detours, optional mild image noise
        cfg.p_extra_bend = 0.35
        cfg.detour_strength = 0.35
        cfg.jitter_px = 10
        cfg.do_messify = True

    if a.enable_mild_image_noise:
        cfg.do_messify = True

    return cfg


def main() -> None:
    ap = build_arg_parser()
    a = ap.parse_args()

    out_dir = Path(a.out)
    ensure_dir(out_dir)

    gen_cfg = cfg_from_args(a)
    rnd_cfg = render_cfg_from_args(a)

    index_rows: List[Dict[str, Any]] = []

    for i in range(a.n):
        sample_seed = a.seed + i
        sample = generate_sample(seed=sample_seed, cfg=gen_cfg)

        idx = a.start_index + i
        folder = out_dir / f"{a.prefix}{idx:0{a.digits}d}"

        if folder.exists() and not a.overwrite:
            raise FileExistsError(f"Folder exists: {folder}. Use --overwrite.")
        ensure_dir(folder)

        write_json(folder / "entities.json", [entity_to_dict(e) for e in sample.entities])
        write_jsonl(folder / "triples_gt.jsonl", [triple_to_dict(t) for t in sample.triples_gt])
        write_jsonl(folder / "triples_obs.jsonl", [triple_to_dict(t) for t in sample.triples_obs])
        write_text(folder / "triples_text.txt", sample.triples_text)

        meta = {
            "seed": sample_seed,
            "counts": {
                "entities": len(sample.entities),
                "triples_gt": len(sample.triples_gt),
                "triples_obs": len(sample.triples_obs),
                "triples_text_lines": len(sample.triples_text),
            },
            "generator_config": asdict(gen_cfg),
        }
        write_json(folder / "meta.json", meta)

        if a.with_images:
            import random
            rng = random.Random(sample_seed)
            img, rmeta = render_sample(
                sample,
                triples_view_for_labels=a.img_view_for_labels,
                rng=rng,
                cfg=rnd_cfg
            )
            img.save(folder / "image.png")
            write_json(folder / "render_meta.json", rmeta)

        index_rows.append({
            "id": f"{a.prefix}{idx:0{a.digits}d}",
            "path": str(folder.relative_to(out_dir)),
            "seed": sample_seed,
            "entities": len(sample.entities),
            "triples_gt": len(sample.triples_gt),
            "triples_obs": len(sample.triples_obs),
            "has_image": bool(a.with_images),
            "render_mode": a.render_mode if a.with_images else None,
        })

    if a.write_index:
        write_json(out_dir / "dataset_index.json", {
            "n": a.n,
            "seed": a.seed,
            "prefix": a.prefix,
            "start_index": a.start_index,
            "digits": a.digits,
            "generator_config": asdict(gen_cfg),
            "render_config": asdict(rnd_cfg) if a.with_images else None,
            "samples": index_rows,
        })

    print(f"✅ Done. Wrote {a.n} samples to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
