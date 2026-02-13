"""
graph_gen.py

Motif-based ground-truth graph generator for semiconductor schematics,
built on top of core_ontology.py.

Outputs:
- entities: List[Entity]
- triples_gt: List[Triple]          (clean ground truth)
- triples_obs: List[Triple]         (optional noisy view of what's "visible"/messy)
- triples_text: List[str]           (semi-natural language sentences for a chosen view)

Next step (later): render_svg.py will take entities + triples_obs (or a derived "diagram view")
and produce drawio-like messy images.

Usage idea:
    from core_ontology import build_core_ontology
    from graph_gen import generate_sample

    sample = generate_sample(seed=0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import random

from core_ontology import (
    build_core_ontology,
    OntologySpec,
    Entity, EntityType,
    DeviceKind, TerminalKind,
    PortDir, DomainKind,
    Predicate, Triple,
    IdFactory,
    make_device, make_net, make_label,
    textualize_triple, maybe_typo,
)

Literal = Union[str, int, float, bool, None]


# ----------------------------
# 1) Config knobs
# ----------------------------

@dataclass
class GenConfig:
    # Size / composition
    n_motifs_min: int = 2
    n_motifs_max: int = 5

    # Motif weights
    w_diffpair: float = 1.0
    w_current_mirror: float = 1.0
    w_rc_filter: float = 0.7
    w_inverter_chain: float = 0.5

    # Label / annotation noise (observed view)
    p_label_drop: float = 0.35
    p_label_wrong_attach: float = 0.10
    p_duplicate_labels: float = 0.05
    p_typo_in_labels: float = 0.10

    # Connectivity ambiguity (later primarily for renderer, but we can pre-mark)
    p_near_miss: float = 0.12
    p_missing_junction_dot: float = 0.18

    # Reification/meta info
    p_add_confidence_meta: float = 0.25
    confidence_range: Tuple[float, float] = (0.55, 0.98)

    # Text
    textualize_view: str = "obs"  # "gt" or "obs"
    n_text_variants_per_triple: int = 1


# ----------------------------
# 2) Sample container
# ----------------------------

@dataclass
class GraphSample:
    entities: List[Entity]
    triples_gt: List[Triple]
    triples_obs: List[Triple]
    triples_text: List[str]
    entity_index: Dict[str, Entity] = field(default_factory=dict)


# ----------------------------
# 3) Graph builder helpers
# ----------------------------

class GraphBuilder:
    def __init__(self, ont: OntologySpec, rng: random.Random):
        self.ont = ont
        self.rng = rng
        self.ids = IdFactory()

        self.entities: List[Entity] = []
        self.entity_by_id: Dict[str, Entity] = {}
        self.triples_gt: List[Triple] = []

        self._triple_counter = 0

    def _new_tid(self) -> str:
        self._triple_counter += 1
        return f"T{self._triple_counter}"

    def add_entity(self, e: Entity) -> Entity:
        self.entities.append(e)
        self.entity_by_id[e.id] = e
        return e

    def add_triple(self, s: str, p: Predicate, o: Union[str, Literal]) -> Triple:
        t = Triple(id=self._new_tid(), s=s, p=p.value, o=o)
        self.triples_gt.append(t)
        return t

    # ----- entity constructors -----

    def make_module(self, name: str, domain: DomainKind = DomainKind.MIXED_SIGNAL) -> Entity:
        mid = self.ids.next("E_MOD")
        m = Entity(id=mid, type=EntityType.MODULE, subtype=None, attrs={"name": name})
        self.add_entity(m)

        # Domain entity
        did = self.ids.next("E_DOM")
        d = Entity(id=did, type=EntityType.DOMAIN, subtype=domain.value, attrs={})
        self.add_entity(d)

        self.add_triple(m.id, Predicate.HAS_DOMAIN, d.id)  # optional; schema has HAS_DOMAIN
        return m

    def make_port(self, direction: PortDir, name: Optional[str] = None) -> Entity:
        pid = self.ids.next("E_PORT")
        p = Entity(id=pid, type=EntityType.PORT, subtype=direction.value, attrs={"name": name} if name else {})
        return self.add_entity(p)

    def make_parameter(self, name: str, value: Union[int, float, str]) -> Entity:
        par_id = self.ids.next("E_PAR")
        par = Entity(id=par_id, type=EntityType.PARAMETER, subtype=None, attrs={})
        self.add_entity(par)
        self.add_triple(par.id, Predicate.PARAM_NAME, name)
        self.add_triple(par.id, Predicate.PARAM_VALUE, value)
        return par

    # ----- device + terminals -----

    def add_mos(self, kind: DeviceKind, refdes: Optional[str] = None) -> Tuple[Entity, Dict[str, Entity]]:
        dev = make_device(self.ont, self.ids, kind=kind, refdes=refdes)
        dev = self.add_entity(dev)

        # MOS terminals
        terms: Dict[str, Entity] = {}
        for tk in (TerminalKind.GATE, TerminalKind.DRAIN, TerminalKind.SOURCE, TerminalKind.BULK):
            tid = self.ids.next("E_T")
            t = Entity(id=tid, type=EntityType.TERMINAL, subtype=tk.value, attrs={})
            self.add_entity(t)
            self.add_triple(dev.id, Predicate.HAS_TERMINAL, t.id)
            self.add_triple(t.id, Predicate.TERMINAL_OF, dev.id)
            terms[tk.value] = t

        # Typical parameters (synthetic but plausible)
        w = round(self.rng.uniform(0.5, 10.0), 3)   # um (ish)
        l = round(self.rng.uniform(0.05, 1.0), 3)   # um (ish)
        par_w = self.make_parameter("W_um", w)
        par_l = self.make_parameter("L_um", l)
        self.add_triple(dev.id, Predicate.HAS_PARAM, par_w.id)
        self.add_triple(dev.id, Predicate.HAS_PARAM, par_l.id)

        return dev, terms

    def add_two_pin_passive(self, kind: DeviceKind, refdes: Optional[str] = None) -> Tuple[Entity, Entity, Entity]:
        """
        For R/C/L/Diode etc. we treat as 2-pin with generic pin terminals.
        """
        dev = make_device(self.ont, self.ids, kind=kind, refdes=refdes)
        dev = self.add_entity(dev)

        t1 = Entity(id=self.ids.next("E_T"), type=EntityType.TERMINAL, subtype=TerminalKind.PIN.value, attrs={"idx": 1})
        t2 = Entity(id=self.ids.next("E_T"), type=EntityType.TERMINAL, subtype=TerminalKind.PIN.value, attrs={"idx": 2})
        self.add_entity(t1)
        self.add_entity(t2)

        self.add_triple(dev.id, Predicate.HAS_TERMINAL, t1.id)
        self.add_triple(dev.id, Predicate.HAS_TERMINAL, t2.id)
        self.add_triple(t1.id, Predicate.TERMINAL_OF, dev.id)
        self.add_triple(t2.id, Predicate.TERMINAL_OF, dev.id)

        # Parameter
        if kind == DeviceKind.RESISTOR:
            r = int(self.rng.choice([100, 220, 330, 1_000, 2_200, 10_000, 47_000]))
            par = self.make_parameter("R_ohm", r)
            self.add_triple(dev.id, Predicate.HAS_PARAM, par.id)
        elif kind == DeviceKind.CAPACITOR:
            c = float(self.rng.choice([1e-12, 10e-12, 100e-12, 1e-9, 10e-9]))
            par = self.make_parameter("C_F", c)
            self.add_triple(dev.id, Predicate.HAS_PARAM, par.id)
        elif kind == DeviceKind.INDUCTOR:
            l = float(self.rng.choice([1e-9, 10e-9, 100e-9, 1e-6]))
            par = self.make_parameter("L_H", l)
            self.add_triple(dev.id, Predicate.HAS_PARAM, par.id)

        return dev, t1, t2

    # ----- wiring -----

    def connect(self, node_id: str, net: Entity) -> None:
        self.add_triple(node_id, Predicate.CONNECTED_VIA, net.id)

    def add_net(self, name: Optional[str] = None) -> Entity:
        n = make_net(self.ids, name=name)
        return self.add_entity(n)

    # ----- annotations -----

    def attach_label(self, target_id: str, text: str) -> Entity:
        lbl = make_label(self.ids, text=text)
        lbl = self.add_entity(lbl)
        self.add_triple(target_id, Predicate.HAS_LABEL, lbl.id)
        self.add_triple(lbl.id, Predicate.LABEL_TEXT, text)
        return lbl

    # ----- hierarchy -----

    def contains(self, module: Entity, child: Entity) -> None:
        self.add_triple(module.id, Predicate.CONTAINS, child.id)

    # ----- reification/meta -----

    def add_confidence_meta(self, triple: Triple, conf: float) -> None:
        # We treat triple IDs as legal subjects for meta-triples.
        self.add_triple(triple.id, Predicate.CONFIDENCE, round(conf, 3))


# ----------------------------
# 4) Motifs (return "interface nets" to connect motifs together)
# ----------------------------

def motif_diffpair(g: GraphBuilder, parent: Entity) -> Dict[str, Entity]:
    """
    Very common analog motif:
      - two NMOS inputs sharing a tail current source
      - resistive loads to VDD (simplified)
    Returns named interface nets: {"vinp","vinn","voutp","voutn","vdd","vss","bias"}
    """
    vdd = g.add_net("VDD")
    vss = g.add_net("VSS")
    vinp = g.add_net("VINP")
    vinn = g.add_net("VINN")
    voutp = g.add_net("VOUTP")
    voutn = g.add_net("VOUTN")
    bias = g.add_net("IBIAS")

    # Devices
    m1, t1 = g.add_mos(DeviceKind.NMOS, refdes="M1")
    m2, t2 = g.add_mos(DeviceKind.NMOS, refdes="M2")
    mtail, tt = g.add_mos(DeviceKind.NMOS, refdes="MTAIL")
    r1, r1a, r1b = g.add_two_pin_passive(DeviceKind.RESISTOR, refdes="R1")
    r2, r2a, r2b = g.add_two_pin_passive(DeviceKind.RESISTOR, refdes="R2")

    # Containment
    for e in [vdd, vss, vinp, vinn, voutp, voutn, bias, m1, m2, mtail, r1, r2]:
        g.contains(parent, e)

    # Wiring (simplified)
    # Inputs to gates
    g.connect(t1["gate"].id, vinp)
    g.connect(t2["gate"].id, vinn)

    # Drains to outputs
    g.connect(t1["drain"].id, voutp)
    g.connect(t2["drain"].id, voutn)

    # Sources to tail node
    tail = g.add_net("TAIL")
    g.contains(parent, tail)
    g.connect(t1["source"].id, tail)
    g.connect(t2["source"].id, tail)
    g.connect(tt["drain"].id, tail)

    # Tail source to VSS, gate to bias
    g.connect(tt["source"].id, vss)
    g.connect(tt["gate"].id, bias)

    # Resistive loads from VDD to outputs
    g.connect(r1a.id, vdd)
    g.connect(r1b.id, voutp)
    g.connect(r2a.id, vdd)
    g.connect(r2b.id, voutn)

    # (optional) labels
    g.attach_label(vinp.id, "in+")
    g.attach_label(vinn.id, "in-")
    g.attach_label(voutp.id, "out+")
    g.attach_label(voutn.id, "out-")

    return {"vinp": vinp, "vinn": vinn, "voutp": voutp, "voutn": voutn, "vdd": vdd, "vss": vss, "bias": bias}


def motif_current_mirror(g: GraphBuilder, parent: Entity) -> Dict[str, Entity]:
    """
    Simple PMOS current mirror at the top (to source current from VDD).
    Returns {"vdd","out","ref","gate"} (gate usually tied to ref drain).
    """
    vdd = g.add_net("VDD")
    out = g.add_net("IMIR_OUT")
    ref = g.add_net("IMIR_REF")

    mp1, p1 = g.add_mos(DeviceKind.PMOS, refdes="MP1")
    mp2, p2 = g.add_mos(DeviceKind.PMOS, refdes="MP2")

    for e in [vdd, out, ref, mp1, mp2]:
        g.contains(parent, e)

    # PMOS sources to VDD
    g.connect(p1["source"].id, vdd)
    g.connect(p2["source"].id, vdd)

    # Tie gates together to ref node (diode-connected MP1)
    g.connect(p1["gate"].id, ref)
    g.connect(p2["gate"].id, ref)

    # MP1 drain to ref, MP2 drain to out
    g.connect(p1["drain"].id, ref)
    g.connect(p2["drain"].id, out)

    g.attach_label(out.id, "bias_out")
    return {"vdd": vdd, "out": out, "ref": ref}


def motif_rc_filter(g: GraphBuilder, parent: Entity) -> Dict[str, Entity]:
    """
    RC low-pass: IN -- R -- OUT, OUT -- C -- VSS
    Returns {"in","out","vss"}
    """
    vss = g.add_net("VSS")
    inn = g.add_net("RC_IN")
    out = g.add_net("RC_OUT")

    r, ra, rb = g.add_two_pin_passive(DeviceKind.RESISTOR, refdes="RLP")
    c, ca, cb = g.add_two_pin_passive(DeviceKind.CAPACITOR, refdes="CLP")

    for e in [vss, inn, out, r, c]:
        g.contains(parent, e)

    g.connect(ra.id, inn)
    g.connect(rb.id, out)

    g.connect(ca.id, out)
    g.connect(cb.id, vss)

    # label sometimes missing later
    g.attach_label(out.id, "filt")
    return {"in": inn, "out": out, "vss": vss}


def motif_inverter_chain(g: GraphBuilder, parent: Entity, n: int = 3) -> Dict[str, Entity]:
    vdd = g.add_net("DVDD")
    vss = g.add_net("DVSS")
    g.contains(parent, vdd)
    g.contains(parent, vss)

    din = g.add_net("DIN")
    g.contains(parent, din)
    net_prev = din

    inv_type = Entity(id=g.ids.next("E_MODT"), type=EntityType.MODULE, subtype="INV", attrs={"name": "INV"})
    g.add_entity(inv_type)

    for i in range(n):
        inst = Entity(id=g.ids.next("E_INST"), type=EntityType.INSTANCE, subtype="INV", attrs={"name": f"U{i+1}"})
        g.add_entity(inst)
        g.contains(parent, inst)
        g.add_triple(inst.id, Predicate.INSTANCE_OF, inv_type.id)

        pin_in = g.make_port(PortDir.IN, name="A")
        pin_out = g.make_port(PortDir.OUT, name="Y")
        g.add_triple(inst.id, Predicate.HAS_PORT, pin_in.id)
        g.add_triple(inst.id, Predicate.HAS_PORT, pin_out.id)

        net_next = g.add_net(f"D{i+1}")
        g.contains(parent, net_next)

        g.add_triple(pin_in.id, Predicate.MAPS_TO, net_prev.id)
        g.add_triple(pin_out.id, Predicate.MAPS_TO, net_next.id)

        if i == n - 1:
            g.attach_label(net_next.id, "dout")

        net_prev = net_next

    return {"in": din, "out": net_prev, "vdd": vdd, "vss": vss}



# ----------------------------
# 5) Stitching motifs into a larger design
# ----------------------------

def choose_motif(rng: random.Random, cfg: GenConfig) -> str:
    choices = [
        ("diffpair", cfg.w_diffpair),
        ("mirror", cfg.w_current_mirror),
        ("rc", cfg.w_rc_filter),
        ("invchain", cfg.w_inverter_chain),
    ]
    total = sum(w for _, w in choices)
    r = rng.random() * total
    acc = 0.0
    for name, w in choices:
        acc += w
        if r <= acc:
            return name
    return choices[-1][0]


def generate_ground_truth(ont: OntologySpec, rng: random.Random, cfg: GenConfig) -> GraphBuilder:
    g = GraphBuilder(ont, rng)

    top = g.make_module("TOP", domain=DomainKind.MIXED_SIGNAL)

    # Add common power domains/nets (not always used, but helpful)
    vdd = g.add_net("VDD")
    vss = g.add_net("VSS")
    g.contains(top, vdd)
    g.contains(top, vss)

    n_motifs = rng.randint(cfg.n_motifs_min, cfg.n_motifs_max)

    motif_ios: List[Dict[str, Entity]] = []
    for _ in range(n_motifs):
        m = choose_motif(rng, cfg)
        if m == "diffpair":
            ios = motif_diffpair(g, top)
        elif m == "mirror":
            ios = motif_current_mirror(g, top)
        elif m == "rc":
            ios = motif_rc_filter(g, top)
        else:
            # inverter chain with random length
            ios = motif_inverter_chain(g, top, n=rng.choice([2, 3, 4]))
        motif_ios.append(ios)

    # Stitch motifs together lightly: connect some outputs to some inputs (net aliasing)
    # We do this by adding CONNECTED_TO between nets (abstract) or by reusing same net id in renderer later.
    # For GT triples we can express: out_net connected_to in_net (abstract), keeping both nets existing.
    for _ in range(rng.randint(1, 3)):
        a = rng.choice(motif_ios)
        b = rng.choice(motif_ios)
        if a is b:
            continue
        # pick plausible keys
        a_out_candidates = [k for k in a.keys() if "out" in k]
        b_in_candidates = [k for k in b.keys() if k in ("in", "vinp", "vinn", "ref")]
        if not a_out_candidates or not b_in_candidates:
            continue
        na = a[rng.choice(a_out_candidates)]
        nb = b[rng.choice(b_in_candidates)]
        g.add_triple(na.id, Predicate.CONNECTED_TO, nb.id)

    # Add some meta confidence triples about random GT statements (later can be used to train robustness)
    if cfg.p_add_confidence_meta > 0:
        for t in list(g.triples_gt):
            if rng.random() < cfg.p_add_confidence_meta:
                conf = rng.uniform(*cfg.confidence_range)
                g.add_confidence_meta(t, conf)

    return g


# ----------------------------
# 6) Noisy "observed" view (labels missing, wrong attachment, duplicates, typos)
# ----------------------------

def make_observed_view(
    rng: random.Random,
    cfg: GenConfig,
    entities: List[Entity],
    triples_gt: List[Triple],
) -> List[Triple]:
    """
    Produces an "observed" triple set that simulates messy real-world diagrams.
    GT stays untouched; observed view drops or corrupts label-related triples.

    We only mutate/affect:
      - HAS_LABEL edges (drop some)
      - LABEL_TEXT literals (typos)
      - Reattach labels to wrong targets (some)
      - Duplicate labels (some)

    Later, the renderer will add *visual* ambiguity; this is the textual/semantic counterpart.
    """
    entity_by_id = {e.id: e for e in entities}

    # Separate label triples
    has_label = [t for t in triples_gt if t.p == Predicate.HAS_LABEL.value]
    label_text = [t for t in triples_gt if t.p == Predicate.LABEL_TEXT.value]
    other = [t for t in triples_gt if t.p not in (Predicate.HAS_LABEL.value, Predicate.LABEL_TEXT.value)]

    # Map label -> its text
    text_by_label: Dict[str, str] = {}
    for t in label_text:
        if isinstance(t.o, str):
            text_by_label[t.s] = t.o

    # Drop labels
    kept_has_label: List[Triple] = []
    for t in has_label:
        if rng.random() < cfg.p_label_drop:
            continue
        kept_has_label.append(t)

    # Wrong-attach: rewire some kept labels to random targets (non-label entities)
    non_label_targets = [e.id for e in entities if e.type in (EntityType.DEVICE, EntityType.NET, EntityType.PORT, EntityType.MODULE, EntityType.INSTANCE)]
    rewired: List[Triple] = []
    for t in kept_has_label:
        if rng.random() < cfg.p_label_wrong_attach and non_label_targets:
            wrong_target = rng.choice(non_label_targets)
            rewired.append(Triple(id=t.id, s=wrong_target, p=t.p, o=t.o))
        else:
            rewired.append(t)

    # Duplicate some labels (same label text used for multiple targets)
    duplicated: List[Triple] = list(rewired)
    if rng.random() < cfg.p_duplicate_labels and rewired:
        t = rng.choice(rewired)
        # Attach same label object to an additional target (very messy)
        extra_target = rng.choice(non_label_targets)
        duplicated.append(Triple(id=f"{t.id}_dup", s=extra_target, p=t.p, o=t.o))

    # Typos in label text
    new_label_text: List[Triple] = []
    for t in label_text:
        if not isinstance(t.o, str):
            new_label_text.append(t)
            continue
        corrupted = maybe_typo(t.o, rng, cfg.p_typo_in_labels)
        new_label_text.append(Triple(id=t.id, s=t.s, p=t.p, o=corrupted))

    return other + duplicated + new_label_text


# ----------------------------
# 7) Public API
# ----------------------------

def generate_sample(seed: int = 0, cfg: Optional[GenConfig] = None) -> GraphSample:
    cfg = cfg or GenConfig()
    rng = random.Random(seed)
    ont = build_core_ontology()

    # If you want config defaults tied to ontology defaults:
    # cfg.p_label_drop = cfg.p_label_drop if cfg.p_label_drop is not None else ont.noise_defaults["p_label_drop"]

    g = generate_ground_truth(ont, rng, cfg)
    triples_obs = make_observed_view(rng, cfg, g.entities, g.triples_gt)

    # Textualize chosen view
    view = g.triples_gt if cfg.textualize_view == "gt" else triples_obs
    sentences: List[str] = []
    for t in view:
        # Optionally skip meta triples in text (confidence etc.) or keep them; here we keep them.
        for _ in range(max(1, cfg.n_text_variants_per_triple)):
            sentences.append(textualize_triple(ont, t, g.entity_by_id, rng))

    return GraphSample(
        entities=g.entities,
        triples_gt=g.triples_gt,
        triples_obs=triples_obs,
        triples_text=sentences,
        entity_index=g.entity_by_id,
    )


if __name__ == "__main__":
    sample = generate_sample(seed=7)
    print(f"Entities: {len(sample.entities)}")
    print(f"GT triples: {len(sample.triples_gt)}")
    print(f"OBS triples: {len(sample.triples_obs)}")
    print("\n--- text examples ---")
    for s in sample.triples_text[:15]:
        print(s)
