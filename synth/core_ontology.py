"""
core_ontology.py

Core ontology for "semiconductor schematic + design intent" graphs,
designed for triple generation (S,P,O) and optional reification
(triples that reference other triples).

Goal: Provide a stable, extensible ontology that supports:
- Entities: devices, nets, ports, terminals, modules, parameters, constraints, labels, comments, measurement points
- Predicates: connectivity, hierarchy, parameters, constraints, annotations, intent
- Textualization: semi-natural language templates for triples
- Noise knobs: label dropout, misattachment, near-miss connectivity, etc.

This module is intentionally model-agnostic: it defines schema & vocab only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import random


# ----------------------------
# 1) Core entity taxonomy
# ----------------------------

class EntityType(str, Enum):
    MODULE = "Module"               # hierarchical container (top, block, submodule)
    INSTANCE = "Instance"           # instance of a module/subcircuit
    DEVICE = "Device"               # transistor, diode, resistor, capacitor...
    TERMINAL = "Terminal"           # device terminal (gate/drain/source/bulk/etc.)
    PORT = "Port"                   # module/instance port (in/out/inout)
    NET = "Net"                     # signal net
    BUS = "Bus"                     # optional: multi-bit net bundle
    JUNCTION = "Junction"           # dot / join point
    PARAMETER = "Parameter"         # name/value parameter
    CONSTRAINT = "Constraint"       # match/ratio/minmax/etc.
    FUNCTION = "Function"           # amplify/filter/bias/compare/regulate...
    DOMAIN = "Domain"               # analog/digital/mixed
    POWER_DOMAIN = "PowerDomain"    # VDD/VSS/AVDD/DVDD...
    LABEL = "Label"                 # text label entity
    COMMENT = "Comment"             # note box / sticky
    MEAS_POINT = "MeasurementPoint" # probe / testpoint


class DeviceKind(str, Enum):
    NMOS = "NMOS"
    PMOS = "PMOS"
    BJT_NPN = "BJT_NPN"
    BJT_PNP = "BJT_PNP"
    DIODE = "Diode"
    RESISTOR = "Resistor"
    CAPACITOR = "Capacitor"
    INDUCTOR = "Inductor"
    # Extend freely: ESD, fuse, zener, etc.


class TerminalKind(str, Enum):
    # MOS
    GATE = "gate"
    DRAIN = "drain"
    SOURCE = "source"
    BULK = "bulk"
    # BJT
    COLLECTOR = "collector"
    BASE = "base"
    EMITTER = "emitter"
    # Generic fallback
    PIN = "pin"


class PortDir(str, Enum):
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    POWER = "power"
    GROUND = "ground"


class FunctionKind(str, Enum):
    AMPLIFY = "amplify"
    FILTER = "filter"
    BIAS = "bias"
    COMPARE = "compare"
    REGULATE = "regulate"
    LEVEL_SHIFT = "level_shift"
    CONVERT = "convert"   # ADC/DAC
    PROTECT = "protect"   # ESD/protection
    SWITCH = "switch"
    LOGIC = "logic"


class DomainKind(str, Enum):
    ANALOG = "analog"
    DIGITAL = "digital"
    MIXED_SIGNAL = "mixed_signal"


# ----------------------------
# 2) Predicate vocabulary
# ----------------------------

class Predicate(str, Enum):
    # Connectivity / wiring (choose either net-based or direct, or mix)
    HAS_TERMINAL = "has_terminal"         # Device -> Terminal
    TERMINAL_OF = "terminal_of"           # Terminal -> Device (optional inverse)
    CONNECTED_VIA = "connected_via"       # Terminal/Port/Junction -> Net
    JOINS = "joins"                       # Junction -> Net (alternative)
    CONNECTED_TO = "connected_to"         # Entity -> Entity (abstract, optional)

    # Hierarchy / structure
    CONTAINS = "contains"                 # Module -> Entity
    INSTANCE_OF = "instance_of"           # Instance -> Module (type)
    HAS_PORT = "has_port"                 # Module/Instance -> Port
    MAPS_TO = "maps_to"                   # Port -> Net (port-net mapping)
    EXPOSES = "exposes"                   # Module -> Net/Port (optional)

    # Parameters / constraints
    HAS_PARAM = "has_param"               # Entity -> Parameter
    PARAM_NAME = "param_name"             # Parameter -> string
    PARAM_VALUE = "param_value"           # Parameter -> number/string
    CONSTRAINED_BY = "constrained_by"     # Entity -> Constraint
    MATCHES = "matches"                   # Entity -> Entity (e.g., device match)
    RATIO = "ratio"                       # Entity -> Entity (ratio relation; detail via param)
    MIN = "min"                           # Constraint -> value
    MAX = "max"                           # Constraint -> value

    # Intent / domains
    IMPLEMENTS = "implements"             # Module/Instance -> Function
    BELONGS_TO = "belongs_to"             # Entity -> Domain/PowerDomain
    HAS_DOMAIN = "has_domain"             # Module -> Domain
    HAS_POWER_DOMAIN = "has_power_domain" # Module -> PowerDomain

    # Annotation / text
    HAS_LABEL = "has_label"               # Entity -> Label
    LABEL_TEXT = "label_text"             # Label -> string
    COMMENT_ON = "comment_on"             # Comment -> Entity/Net
    COMMENT_TEXT = "comment_text"         # Comment -> string
    MEASURES = "measures"                 # MeasurementPoint -> Net/Entity

    # Reification / meta statements about triples (optional)
    TRIPLE_SUBJECT = "triple_subject"     # Triple -> Entity/Triple
    TRIPLE_PREDICATE = "triple_predicate" # Triple -> Predicate
    TRIPLE_OBJECT = "triple_object"       # Triple -> Entity/Triple/literal
    SUPPORTED_BY = "supported_by"         # Triple -> Label/Comment/ImageRegion (later)
    AMBIGUOUS = "ambiguous"               # Triple -> boolean/literal
    CONFIDENCE = "confidence"             # Triple -> float
    DERIVED_FROM = "derived_from"         # Triple -> literal (e.g. "layout_heuristic")


# ----------------------------
# 3) Core dataclasses
# ----------------------------

@dataclass(frozen=True)
class Entity:
    """
    Minimal entity object; attrs is an open dict for extensibility.
    """
    id: str
    type: EntityType
    subtype: Optional[str] = None  # e.g. DeviceKind, FunctionKind, etc. as string
    attrs: Dict[str, Any] = field(default_factory=dict)


Literal = Union[str, int, float, bool, None]
NodeRef = Union[str, "TripleRef"]  # entity id or triple reference


@dataclass(frozen=True)
class Triple:
    """
    Canonical triple: (s, p, o)
    - s: entity id OR triple id (if you want triple-as-subject)
    - p: predicate string
    - o: entity id OR literal OR triple id
    """
    id: str
    s: str
    p: str
    o: Union[str, Literal]


@dataclass(frozen=True)
class TripleRef:
    """
    Helper for referencing a triple as a node in other triples.
    This is optional—internally you can just use the triple id string.
    """
    triple_id: str


# ----------------------------
# 4) Ontology spec container
# ----------------------------

@dataclass
class OntologySpec:
    """
    A stable "core ontology" spec:
    - allowed entity types and subtypes
    - allowed predicates and domain/range constraints (soft, not enforced strictly)
    - textualization templates
    """
    entity_types: Tuple[EntityType, ...]
    device_kinds: Tuple[DeviceKind, ...]
    terminal_kinds: Tuple[TerminalKind, ...]
    port_dirs: Tuple[PortDir, ...]
    function_kinds: Tuple[FunctionKind, ...]
    domain_kinds: Tuple[DomainKind, ...]
    predicates: Tuple[Predicate, ...]

    # Soft schema constraints: predicate -> (allowed_subject_types, allowed_object_types_or_"LITERAL")
    # Use as guidance during generation (not required for parsing).
    predicate_schema: Dict[Predicate, Tuple[Tuple[EntityType, ...], Union[Tuple[EntityType, ...], str]]]

    # Templates for semi-natural triple text
    templates: Dict[Predicate, List[str]]

    # Noise knobs default values
    noise_defaults: Dict[str, float]


def build_core_ontology() -> OntologySpec:
    """
    Returns a compact but expressive ontology spec.
    """

    predicate_schema: Dict[Predicate, Tuple[Tuple[EntityType, ...], Union[Tuple[EntityType, ...], str]]] = {
        Predicate.HAS_TERMINAL: ((EntityType.DEVICE,), (EntityType.TERMINAL,)),
        Predicate.TERMINAL_OF: ((EntityType.TERMINAL,), (EntityType.DEVICE,)),
        Predicate.CONNECTED_VIA: (
            (EntityType.TERMINAL, EntityType.PORT, EntityType.JUNCTION, EntityType.DEVICE, EntityType.INSTANCE),
            (EntityType.NET, EntityType.BUS),
        ),
        Predicate.CONNECTED_TO: (
            (EntityType.DEVICE, EntityType.PORT, EntityType.NET, EntityType.INSTANCE, EntityType.MODULE),
            (EntityType.DEVICE, EntityType.PORT, EntityType.NET, EntityType.INSTANCE, EntityType.MODULE),
        ),
        Predicate.CONTAINS: ((EntityType.MODULE,), (EntityType.DEVICE, EntityType.NET, EntityType.PORT, EntityType.INSTANCE, EntityType.JUNCTION, EntityType.LABEL, EntityType.COMMENT, EntityType.MEAS_POINT)),
        Predicate.INSTANCE_OF: ((EntityType.INSTANCE,), (EntityType.MODULE,)),
        Predicate.HAS_PORT: ((EntityType.MODULE, EntityType.INSTANCE), (EntityType.PORT,)),
        Predicate.MAPS_TO: ((EntityType.PORT,), (EntityType.NET, EntityType.BUS)),
        Predicate.HAS_PARAM: ((EntityType.DEVICE, EntityType.INSTANCE, EntityType.MODULE, EntityType.CONSTRAINT), (EntityType.PARAMETER,)),
        Predicate.PARAM_NAME: ((EntityType.PARAMETER,), "LITERAL"),
        Predicate.PARAM_VALUE: ((EntityType.PARAMETER,), "LITERAL"),
        Predicate.CONSTRAINED_BY: ((EntityType.DEVICE, EntityType.INSTANCE, EntityType.MODULE), (EntityType.CONSTRAINT,)),
        Predicate.MATCHES: ((EntityType.DEVICE,), (EntityType.DEVICE,)),
        Predicate.RATIO: ((EntityType.DEVICE,), (EntityType.DEVICE,)),
        Predicate.IMPLEMENTS: ((EntityType.MODULE, EntityType.INSTANCE), (EntityType.FUNCTION,)),
        Predicate.BELONGS_TO: ((EntityType.DEVICE, EntityType.NET, EntityType.MODULE, EntityType.INSTANCE), (EntityType.DOMAIN, EntityType.POWER_DOMAIN)),
        Predicate.HAS_LABEL: ((EntityType.DEVICE, EntityType.NET, EntityType.PORT, EntityType.MODULE, EntityType.INSTANCE), (EntityType.LABEL,)),
        Predicate.LABEL_TEXT: ((EntityType.LABEL,), "LITERAL"),
        Predicate.COMMENT_ON: ((EntityType.COMMENT,), (EntityType.DEVICE, EntityType.NET, EntityType.MODULE, EntityType.INSTANCE, EntityType.PORT)),
        Predicate.COMMENT_TEXT: ((EntityType.COMMENT,), "LITERAL"),
        Predicate.MEASURES: ((EntityType.MEAS_POINT,), (EntityType.NET, EntityType.PORT)),
        # Reification (Triples are not EntityType; you can keep schema soft)
    }

    templates: Dict[Predicate, List[str]] = {
        Predicate.HAS_TERMINAL: [
            "{S} has terminal {O}.",
            "Terminal {O} belongs to {S}.",
        ],
        Predicate.CONNECTED_VIA: [
            "{S} is connected via {O}.",
            "{S} is wired to net {O}.",
            "{S} ties into {O}.",
        ],
        Predicate.CONTAINS: [
            "{S} contains {O}.",
            "{O} is inside {S}.",
        ],
        Predicate.INSTANCE_OF: [
            "{S} is an instance of {O}.",
            "{S} instantiates {O}.",
        ],
        Predicate.HAS_PORT: [
            "{S} has port {O}.",
            "Port {O} is defined on {S}.",
        ],
        Predicate.MAPS_TO: [
            "{S} maps to {O}.",
            "{S} is bound to {O}.",
        ],
        Predicate.HAS_PARAM: [
            "{S} has parameter {O}.",
            "{O} is a parameter of {S}.",
        ],
        Predicate.PARAM_NAME: [
            "{S} name is '{O}'.",
            "Parameter {S} is called '{O}'.",
        ],
        Predicate.PARAM_VALUE: [
            "{S} value is {O}.",
            "Parameter {S} equals {O}.",
        ],
        Predicate.CONSTRAINED_BY: [
            "{S} is constrained by {O}.",
            "{O} constrains {S}.",
        ],
        Predicate.MATCHES: [
            "{S} matches {O}.",
            "{S} and {O} are matched devices.",
        ],
        Predicate.RATIO: [
            "{S} is ratio-related to {O}.",
            "{S} is scaled relative to {O}.",
        ],
        Predicate.IMPLEMENTS: [
            "{S} implements {O}.",
            "{S} realizes function {O}.",
        ],
        Predicate.BELONGS_TO: [
            "{S} belongs to {O}.",
            "{S} is assigned to {O}.",
        ],
        Predicate.HAS_LABEL: [
            "{S} has label {O}.",
            "{O} labels {S}.",
        ],
        Predicate.LABEL_TEXT: [
            "{S} text is '{O}'.",
            "Label {S} says '{O}'.",
        ],
        Predicate.COMMENT_ON: [
            "{S} comments on {O}.",
            "{S} is a note about {O}.",
        ],
        Predicate.COMMENT_TEXT: [
            "{S} text is '{O}'.",
            "Comment {S} reads '{O}'.",
        ],
        Predicate.MEASURES: [
            "{S} measures {O}.",
            "{S} probes {O}.",
        ],
        # Reification templates can be added later
    }

    noise_defaults = {
        # Visual/text noise knobs (for later "messy render" stage)
        "p_label_drop": 0.35,
        "p_label_wrong_attach": 0.10,
        "p_duplicate_labels": 0.05,
        "p_typo_in_labels": 0.10,

        # Connectivity ambiguity (visual vs GT)
        "p_near_miss": 0.12,          # visually near connection but not actually connected (or vice versa)
        "p_missing_junction_dot": 0.18,

        # Layout messiness (renderer stage)
        "p_overlap": 0.15,
        "p_wire_crossing": 0.25,
        "wire_detour_strength": 0.35,  # 0..1
    }

    return OntologySpec(
        entity_types=tuple(EntityType),
        device_kinds=tuple(DeviceKind),
        terminal_kinds=tuple(TerminalKind),
        port_dirs=tuple(PortDir),
        function_kinds=tuple(FunctionKind),
        domain_kinds=tuple(DomainKind),
        predicates=tuple(Predicate),
        predicate_schema=predicate_schema,
        templates=templates,
        noise_defaults=noise_defaults,
    )


# ----------------------------
# 5) Helpers: ID factories, basic entity creation, textualization
# ----------------------------

@dataclass
class IdFactory:
    counters: Dict[str, int] = field(default_factory=dict)

    def next(self, prefix: str) -> str:
        n = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = n
        return f"{prefix}{n}"


def make_device(ont: OntologySpec, ids: IdFactory, kind: Optional[DeviceKind] = None, refdes: Optional[str] = None) -> Entity:
    kind = kind or random.choice(ont.device_kinds)
    did = ids.next("E_M" if kind in (DeviceKind.NMOS, DeviceKind.PMOS) else "E_D")
    attrs = {}
    if refdes:
        attrs["refdes"] = refdes
    return Entity(id=did, type=EntityType.DEVICE, subtype=kind.value, attrs=attrs)


def make_net(ids: IdFactory, name: Optional[str] = None) -> Entity:
    nid = ids.next("E_NET")
    attrs = {}
    if name:
        attrs["name"] = name
    return Entity(id=nid, type=EntityType.NET, subtype=None, attrs=attrs)


def make_label(ids: IdFactory, text: str) -> Entity:
    lid = ids.next("E_LBL")
    return Entity(id=lid, type=EntityType.LABEL, subtype=None, attrs={"text": text})


def textualize_triple(
    ont: OntologySpec,
    triple: Triple,
    entity_lookup: Dict[str, Entity],
    rng: random.Random,
) -> str:
    """
    Convert a canonical triple to a semi-natural sentence using templates.
    - Replaces {S} and {O} with readable names (prefers label/refdes/name if present).
    """
    pred = Predicate(triple.p) if triple.p in Predicate._value2member_map_ else None
    templates = ont.templates.get(pred, ["{S} {P} {O}."]) if pred else ["{S} {P} {O}."]

    tmpl = rng.choice(templates)

    def pretty(x: Union[str, Literal]) -> str:
        if isinstance(x, str) and x in entity_lookup:
            e = entity_lookup[x]
            # prefer human-ish name fields
            if e.type == EntityType.DEVICE and "refdes" in e.attrs:
                return e.attrs["refdes"]
            if e.type == EntityType.NET and "name" in e.attrs:
                return e.attrs["name"]
            if e.type == EntityType.LABEL and "text" in e.attrs:
                return f"label[{e.attrs['text']}]"
            # fallback
            return f"{e.type.value}:{e.id.split('_')[-1]}"
        return str(x)

    return tmpl.format(S=pretty(triple.s), P=triple.p, O=pretty(triple.o))


# ----------------------------
# 6) Optional: simple noise utilities for text labels
# ----------------------------

def maybe_typo(s: str, rng: random.Random, p: float) -> str:
    if rng.random() > p or not s:
        return s
    # simple typo: drop/replace one char
    i = rng.randrange(len(s))
    ops = ["drop", "swap", "replace"]
    op = rng.choice(ops)
    if op == "drop" and len(s) > 1:
        return s[:i] + s[i+1:]
    if op == "swap" and len(s) > 1 and i < len(s) - 1:
        return s[:i] + s[i+1] + s[i] + s[i+2:]
    if op == "replace":
        return s[:i] + rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789_")) + s[i+1:]
    return s


# ----------------------------
# 7) Demo (does not generate a full graph yet)
# ----------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    ont = build_core_ontology()
    ids = IdFactory()

    # tiny example entities
    m1 = make_device(ont, ids, DeviceKind.NMOS, refdes="M1")
    net1 = make_net(ids, name="NET_A")
    lbl = make_label(ids, text="bias")

    entities = {e.id: e for e in [m1, net1, lbl]}

    triples = [
        Triple(id="T1", s=m1.id, p=Predicate.CONNECTED_VIA.value, o=net1.id),
        Triple(id="T2", s=m1.id, p=Predicate.HAS_LABEL.value, o=lbl.id),
        Triple(id="T3", s=lbl.id, p=Predicate.LABEL_TEXT.value, o=lbl.attrs["text"]),
    ]

    for t in triples:
        print(textualize_triple(ont, t, entities, rng))
