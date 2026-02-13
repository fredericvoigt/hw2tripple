"""
render_drawioish.py

DrawIO-ish renderer using Pillow, now with:
- Guaranteed non-overlapping grid placement
- Fit-to-canvas (crop occupied area + resize back) to avoid wasting space
- Optional mild image degradation (OFF in clean mode)

No editor UI overlays.

Requires:
  pip install pillow
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
import math
import os
import random
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from core_ontology import EntityType, Predicate


Point = Tuple[int, int]
Polyline = List[Point]


@dataclass
class RenderConfig:
    width: int = 1200
    height: int = 800
    padding: int = 40

    # node boxes
    box_w_range: Tuple[int, int] = (100, 150)
    box_h_range: Tuple[int, int] = (55, 85)
    corner_radius: int = 8
    box_outline_w_range: Tuple[int, int] = (2, 3)
    wire_w_range: Tuple[int, int] = (2, 3)

    # layout
    grid_gap: int = 16            # minimum gap between cells
    jitter_px: int = 10           # bounded jitter within cell (kept small)
    min_box_shrink: float = 0.65  # if too many nodes, boxes shrink but not below this

    # wire routing
    p_extra_bend: float = 0.25
    detour_strength: float = 0.25
    min_branch_gap: int = 10

    # junction dots
    p_missing_junction_dot: float = 0.10
    junction_r_range: Tuple[int, int] = (3, 5)

    # text
    font_size_range: Tuple[int, int] = (12, 18)
    p_hide_refdes: float = 0.03

    # fit-to-canvas
    fit_to_canvas: bool = True
    fit_margin: int = 28

    # mild post-degradation (OFF by default for clean)
    do_messify: bool = False
    jpeg_quality_range: Tuple[int, int] = (60, 92)
    blur_sigma_max: float = 0.6
    contrast_range: Tuple[float, float] = (0.92, 1.05)
    brightness_range: Tuple[float, float] = (0.95, 1.03)
    p_rescale: float = 0.10
    rescale_range: Tuple[float, float] = (0.92, 1.08)


def _find_fonts() -> List[str]:
    candidates = []
    roots = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dirpath, _, filenames in os.walk(r):
            for fn in filenames:
                if fn.lower().endswith(".ttf") or fn.lower().endswith(".otf"):
                    candidates.append(os.path.join(dirpath, fn))

    preferred = []
    for key in ["DejaVuSans", "LiberationSans", "Arial", "Calibri", "DejaVuSerif"]:
        preferred.extend([p for p in candidates if key.lower() in os.path.basename(p).lower()])

    seen, out = set(), []
    for p in preferred + candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:250]


def _load_font(rng: random.Random, cfg: RenderConfig) -> ImageFont.FreeTypeFont:
    size = rng.randint(*cfg.font_size_range)
    fonts = _find_fonts()
    if fonts:
        for _ in range(8):
            fp = rng.choice(fonts)
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _rounded_rect(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], r: int, width: int) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(xy, radius=r, outline=0, width=width, fill=255)
    else:
        draw.rectangle(xy, outline=0, width=width, fill=255)


def _route_net_polylines(
    rng: random.Random,
    cfg: RenderConfig,
    endpoints: List[Point]
) -> Tuple[List[Polyline], List[Point]]:
    """
    Orthogonal routing via a horizontal trunk (median y) + L-branches.
    Produces stable, readable wires.
    """
    if len(endpoints) <= 1:
        return [], []

    xs = [p[0] for p in endpoints]
    ys = [p[1] for p in endpoints]
    y_med = int(sorted(ys)[len(ys)//2])

    det = int(rng.uniform(-1, 1) * cfg.detour_strength * 80)
    trunk_y = _clamp(y_med + det, cfg.padding, cfg.height - cfg.padding)

    x0, x1 = min(xs), max(xs)
    polylines: List[Polyline] = []
    junctions: List[Point] = []

    if rng.random() < cfg.p_extra_bend and (x1 - x0) > 160:
        mid_x = int((x0 + x1) / 2) + int(rng.uniform(-1, 1) * cfg.detour_strength * 140)
        mid_x = _clamp(mid_x, cfg.padding, cfg.width - cfg.padding)
        trunk = [(x0, trunk_y), (mid_x, trunk_y), (x1, trunk_y)]
    else:
        trunk = [(x0, trunk_y), (x1, trunk_y)]
    polylines.append(trunk)

    for (x, y) in endpoints:
        jx, jy = x, trunk_y
        if abs(y - trunk_y) < cfg.min_branch_gap:
            jy = trunk_y + (cfg.min_branch_gap if y < trunk_y else -cfg.min_branch_gap)
            jy = _clamp(jy, cfg.padding, cfg.height - cfg.padding)
        branch: Polyline = [(x, y), (jx, y), (jx, jy)]
        polylines.append(branch)
        junctions.append((jx, jy))

    return polylines, junctions


def _apply_mild_messify(img: Image.Image, rng: random.Random, cfg: RenderConfig) -> Image.Image:
    if not cfg.do_messify:
        return img

    if rng.random() < cfg.p_rescale:
        s = rng.uniform(*cfg.rescale_range)
        w2 = max(64, int(img.size[0] * s))
        h2 = max(64, int(img.size[1] * s))
        img = img.resize((w2, h2), resample=Image.BILINEAR).resize(img.size, resample=Image.BILINEAR)

    sigma = rng.uniform(0.0, cfg.blur_sigma_max)
    if sigma > 0.01:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    img = ImageEnhance.Contrast(img).enhance(rng.uniform(*cfg.contrast_range))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(*cfg.brightness_range))

    q = rng.randint(*cfg.jpeg_quality_range)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    img = Image.open(buf).convert("L")
    return img


def _fit_to_canvas(img: Image.Image, bboxes: Dict[str, Tuple[int, int, int, int]], wires_meta: Dict[str, Any], cfg: RenderConfig) -> Image.Image:
    xs, ys = [], []
    for (x0, y0, x1, y1) in bboxes.values():
        xs += [x0, x1]
        ys += [y0, y1]
    for net in wires_meta.values():
        for pl in net.get("polylines", []):
            for (x, y) in pl:
                xs.append(x); ys.append(y)

    if not xs or not ys:
        return img

    margin = cfg.fit_margin
    x0 = _clamp(min(xs) - margin, 0, cfg.width - 1)
    y0 = _clamp(min(ys) - margin, 0, cfg.height - 1)
    x1 = _clamp(max(xs) + margin, 1, cfg.width)
    y1 = _clamp(max(ys) + margin, 1, cfg.height)

    cropped = img.crop((x0, y0, x1, y1))
    return cropped.resize((cfg.width, cfg.height), resample=Image.BILINEAR)


def _non_overlapping_grid_layout(
    rng: random.Random,
    cfg: RenderConfig,
    node_ids: List[str],
) -> Tuple[Dict[str, Tuple[int, int, int, int]], float, int, int]:
    """
    Places one node per grid cell. Ensures cell can fit max box size + gap.
    If too many nodes, boxes are uniformly shrunk (up to min_box_shrink) to fit.
    """
    n = len(node_ids)
    if n == 0:
        return {}, 1.0, 1, 1

    max_bw = cfg.box_w_range[1]
    max_bh = cfg.box_h_range[1]
    gap = cfg.grid_gap

    avail_w = cfg.width - 2 * cfg.padding
    avail_h = cfg.height - 2 * cfg.padding

    # pick best cols among candidates near sqrt(n * aspect)
    aspect = avail_w / max(1, avail_h)
    target_cols = max(2, int(math.sqrt(n * aspect)))
    candidates = list(range(max(2, target_cols - 6), min(n, target_cols + 8) + 1))
    candidates = sorted(set([c for c in candidates if c >= 2]))

    best = None
    for cols in candidates:
        rows = (n + cols - 1) // cols
        cell_w = avail_w / cols
        cell_h = avail_h / rows

        # shrink needed?
        shrink_w = (cell_w - gap) / max_bw
        shrink_h = (cell_h - gap) / max_bh
        shrink = min(shrink_w, shrink_h)

        if shrink >= cfg.min_box_shrink:
            # prefer layouts that use space well (larger cells)
            score = cell_w * cell_h
            if best is None or score > best[0]:
                best = (score, cols, rows, cell_w, cell_h, min(1.0, shrink))

    if best is None:
        # fallback: maximize cols that fit at least min shrink
        cols = max(2, int(avail_w / (max_bw * cfg.min_box_shrink + gap)))
        cols = max(cols, 2)
        rows = (n + cols - 1) // cols
        cell_w = avail_w / cols
        cell_h = avail_h / rows
        shrink_w = (cell_w - gap) / max_bw
        shrink_h = (cell_h - gap) / max_bh
        shrink = max(cfg.min_box_shrink, min(shrink_w, shrink_h, 1.0))
    else:
        _, cols, rows, cell_w, cell_h, shrink = best

    # jitter bounded so boxes stay inside cell
    jx = min(cfg.jitter_px, max(0, int((cell_w - (max_bw * shrink)) / 2)))
    jy = min(cfg.jitter_px, max(0, int((cell_h - (max_bh * shrink)) / 2)))

    bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    for idx, nid in enumerate(node_ids):
        r = idx // cols
        c = idx % cols

        bw = int(rng.randint(*cfg.box_w_range) * shrink)
        bh = int(rng.randint(*cfg.box_h_range) * shrink)

        cx = int(cfg.padding + c * cell_w + cell_w / 2)
        cy = int(cfg.padding + r * cell_h + cell_h / 2)

        if jx > 0:
            cx += rng.randint(-jx, jx)
        if jy > 0:
            cy += rng.randint(-jy, jy)

        x0 = int(cx - bw / 2)
        y0 = int(cy - bh / 2)

        x0 = _clamp(x0, cfg.padding, cfg.width - cfg.padding - bw)
        y0 = _clamp(y0, cfg.padding, cfg.height - cfg.padding - bh)

        bboxes[nid] = (x0, y0, x0 + bw, y0 + bh)

    return bboxes, shrink, cols, rows


def render_sample(sample: Any, triples_view_for_labels: str, rng: random.Random, cfg: RenderConfig) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Renders a GraphSample from graph_gen.generate_sample().

    - Uses GT connectivity (CONNECTED_VIA) for wires.
    - Uses chosen triples view for labels (gt/obs).
    """
    img = Image.new("L", (cfg.width, cfg.height), color=255)
    draw = ImageDraw.Draw(img)

    entities = {e.id: e for e in sample.entities}
    triples_gt = sample.triples_gt
    triples_obs = sample.triples_obs
    triples_for_labels = triples_obs if triples_view_for_labels == "obs" else triples_gt

    # terminal -> device
    terminal_of: Dict[str, str] = {}
    # port -> parent instance/module
    port_parent: Dict[str, str] = {}

    for t in triples_gt:
        if t.p == Predicate.TERMINAL_OF.value and isinstance(t.o, str):
            terminal_of[t.s] = t.o
        if t.p == Predicate.HAS_PORT.value and isinstance(t.o, str):
            port_parent[t.o] = t.s

    # labels (from chosen view)
    has_label: Dict[str, List[str]] = {}
    label_text: Dict[str, str] = {}
    for t in triples_for_labels:
        if t.p == Predicate.HAS_LABEL.value and isinstance(t.o, str):
            has_label.setdefault(t.s, []).append(t.o)
        if t.p == Predicate.LABEL_TEXT.value and isinstance(t.o, str):
            label_text[t.s] = t.o

    def labels_for(subject_id: str) -> List[str]:
        out = []
        for lid in has_label.get(subject_id, []):
            txt = label_text.get(lid)
            if txt:
                out.append(txt)
        return out

    # draw nodes = devices + instances
    node_ids = [e.id for e in sample.entities if e.type in (EntityType.DEVICE, EntityType.INSTANCE)]

    # layout
    bboxes, shrink, cols, rows = _non_overlapping_grid_layout(rng, cfg, node_ids)

    def anchor_for(node_id: str, role: str) -> Point:
        x0, y0, x1, y1 = bboxes[node_id]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if role == "gate":   return (x0, cy)
        if role == "drain":  return (cx, y0)
        if role == "source": return (cx, y1)
        if role == "bulk":   return (x1, cy)
        if role == "in":     return (x0, cy)
        if role == "out":    return (x1, cy)
        if role == "power":  return (cx, y0)
        if role == "ground": return (cx, y1)
        return (cx, cy)

    # gather net endpoints from GT (stable)
    net_to_endpoints: Dict[str, List[Tuple[str, Point]]] = {}
    for t in triples_gt:
        if t.p != Predicate.CONNECTED_VIA.value:
            continue
        if not (isinstance(t.s, str) and isinstance(t.o, str)):
            continue
        endpoint_id = t.s
        net_id = t.o

        if endpoint_id in terminal_of:
            dev_id = terminal_of[endpoint_id]
            dev = entities.get(dev_id)
            term = entities.get(endpoint_id)
            if dev and term and dev_id in bboxes:
                pt = anchor_for(dev_id, term.subtype or "pin")
                net_to_endpoints.setdefault(net_id, []).append((endpoint_id, pt))
        elif endpoint_id in port_parent:
            parent_id = port_parent[endpoint_id]
            port = entities.get(endpoint_id)
            if port and parent_id in bboxes:
                pt = anchor_for(parent_id, port.subtype or "pin")
                net_to_endpoints.setdefault(net_id, []).append((endpoint_id, pt))
        else:
            if endpoint_id in bboxes:
                pt = anchor_for(endpoint_id, "pin")
                net_to_endpoints.setdefault(net_id, []).append((endpoint_id, pt))

    # wires first
    wire_width = rng.randint(*cfg.wire_w_range)
    wires_meta: Dict[str, Any] = {}

    for net_id, eps in net_to_endpoints.items():
        pts = [p for _, p in eps]
        polylines, junctions = _route_net_polylines(rng, cfg, pts)

        for pl in polylines:
            if len(pl) >= 2:
                draw.line(pl, fill=0, width=wire_width)

        # junction dots (sometimes missing, but mild)
        j_drawn: List[Point] = []
        for j in junctions:
            if rng.random() < cfg.p_missing_junction_dot:
                continue
            rj = rng.randint(*cfg.junction_r_range)
            draw.ellipse([j[0]-rj, j[1]-rj, j[0]+rj, j[1]+rj], fill=0)
            j_drawn.append(j)

        # net labels if present (from labels view)
        nlabels = labels_for(net_id)
        if nlabels:
            font = _load_font(rng, cfg)
            lx = int(sum(p[0] for p in pts) / len(pts))
            ly = int(sum(p[1] for p in pts) / len(pts))
            draw.text((lx + 6, ly - 10), nlabels[0], fill=0, font=font)

        wires_meta[net_id] = {
            "polylines": polylines,
            "junctions": j_drawn,
            "endpoints": [{"endpoint": eid, "xy": pt} for eid, pt in eps],
        }

    # boxes + text
    for nid in node_ids:
        x0, y0, x1, y1 = bboxes[nid]
        outline_w = rng.randint(*cfg.box_outline_w_range)
        _rounded_rect(draw, (x0, y0, x1, y1), r=cfg.corner_radius, width=outline_w)

        e = entities[nid]
        font = _load_font(rng, cfg)

        title = ""
        if e.type == EntityType.DEVICE:
            title = e.attrs.get("refdes", e.subtype or "DEV")
        else:
            title = e.attrs.get("name", e.subtype or "INST")

        if rng.random() < cfg.p_hide_refdes:
            title = e.subtype or ""

        draw.text((x0 + 6, y0 + 4), title, fill=0, font=font)

        extra = labels_for(nid)
        if extra:
            draw.text((x0 + 6, y1 + 2), extra[0], fill=0, font=font)

    # fit-to-canvas (use full image)
    if cfg.fit_to_canvas:
        img = _fit_to_canvas(img, bboxes, wires_meta, cfg)

    # mild degradation only if enabled
    img = _apply_mild_messify(img, rng, cfg)

    meta = {
        "render_config": asdict(cfg),
        "layout": {"shrink": shrink, "cols": cols, "rows": rows},
        "bboxes": {k: list(v) for k, v in bboxes.items()},
        "wires": wires_meta,
        "labels_view": triples_view_for_labels,
    }
    return img, meta
