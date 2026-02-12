"""Build Mermaid sources and PNG diagrams for the current RAG/Viking pipeline.

This script is intentionally self-contained and offline-safe:
- Writes canonical Mermaid `.mmd` files.
- Renders matching PNGs using Pillow (no network dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("docs/diagrams")


PIPELINE_MERMAID = r"""flowchart TD
  Q[User Query] --> UI[Streamlit UI<br/>src/ui/graph_web.py]
  UI --> LG[LangGraph App<br/>src/agent/graph.py]
  LG --> RETN[retrieve node]

  RETN --> META[Extract Query Metadata<br/>lang detection + intent/topic hints]
  META --> VR[Viking Router<br/>taxonomy-aware scope selection]
  VR --> VS[Viking Scope<br/>patterns + confidence + trace]

  %% Scoped Retrieval (Vector first)
  VS --> VEC[Vector Search (scoped)<br/>pgvector SQL + lang filter + scope patterns]
  VEC --> FB{Soft mode AND<br/>hits &lt; min_hits AND<br/>attempts &lt; max_expansions?}

  FB -- yes --> WIDEN[Widen Scope (controlled)<br/>phenomenon → category → lang-only → global]
  WIDEN --> VEC

  FB -- no --> BM25[BM25 Search (global keywords)<br/>candidate pool for fusion]

  %% BM25 pre-filter policy depends on Viking mode
  BM25 --> PRE{Viking mode?}
  PRE -- strict --> BMF[BM25 Pre-filter (strict)<br/>keep only in-scope candidates]
  PRE -- soft --> BMS[Skip BM25 pre-filter (soft)<br/>allow BM25 into fusion]

  %% Fusion decision and semantics clarification
  BMF --> FCHK{Hybrid enabled AND<br/>BM25 candidates &gt; 0?}
  BMS --> FCHK

  FCHK -- yes --> RRF[RRF Fusion<br/>merge vector + BM25 candidates]
  FCHK -- no --> VO[Vector-only path<br/>(note: BM25 may be 0 after strict filter)]

  %% Scope guard after fusion (important semantics)
  RRF --> SG[Viking Scope Guard<br/>strict: hard drop out-of-scope<br/>soft: drop out-of-scope unless in-scope &lt; MIN_IN_SCOPE]
  VO --> SG

  %% Retrieval post-processing
  SG --> REC[Recursive Retrieval / RAPTOR (optional)<br/>tree/child expansion]
  REC --> RR[Reranker (optional)]
  RR --> SE[Sibling Expansion (optional)<br/>fill title-only chunks]
  SE --> CTX[Final Context Pack<br/>doc ids + paths + snippets]

  %% Downstream LangGraph remains unchanged
  CTX --> GD[grade_documents (optional skip)]
  GD --> DEC{Relevant docs?}
  DEC -- no --> RW[rewrite + retry]
  RW --> RETN
  DEC -- yes --> GEN[generate]
  GEN --> HC[check_hallucination (optional skip)]
  HC --> OUT[Answer + References]

  %% UX note for strict mode
  VO -. strict may yield 0 docs .-> NOTE[UI should warn if strict scope returns 0 docs<br/>suggest soft mode / widen policy]
"""


HARNESS_MERMAID = r"""flowchart LR
  QS[Query Set (JSONL/YAML)<br/>Q-only in Late QA] --> HR[Harness Runner<br/>src/eval/run_retrieval_ablation.py]
  CP[Condition Profiles<br/>baseline/hybrid/viking_soft/viking_strict/raptor_on] --> HR
  CFG[Config Overlay (no UI)<br/>patch load_config or pass overrides] --> HR

  HR --> RET[RAGRetriever.retrieve_documents()<br/>retriever-core only]
  RET --> AUD[audit.jsonl<br/>per query × condition<br/>scope/trace/fallback/docs/scores/timing]
  RET --> MET[metrics.json<br/>run meta: git hash + config hash + query hash<br/>summaries per condition]
  AUD --> POST[Optional metrics post-process<br/>Hit@k/MRR/contam/coverage if labels exist]
  MET --> POST

  POST --> BK[Artifact policy<br/>Git: small manifests/metrics<br/>Backup: large logs in ~/backupRAG/&lt;run_id&gt;]
"""


@dataclass
class Node:
    node_id: str
    label: str
    x: int
    y: int
    w: int = 260
    h: int = 84
    fill: str = "#EEF3FB"


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    points: Tuple[Tuple[int, int], ...] = ()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _multiline_center(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    line_heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = y1 + (y2 - y1 - total_h) // 2
    for i, line in enumerate(lines):
        w = widths[i]
        h = line_heights[i]
        x = x1 + (x2 - x1 - w) // 2
        draw.text((x, y), line, fill="#0F172A", font=font)
        y += h + 4


def _anchor_for_target(node: Node, target: Tuple[int, int]) -> Tuple[int, int]:
    tx, ty = target
    left = node.x
    right = node.x + node.w
    top = node.y
    bottom = node.y + node.h
    cx = node.x + node.w // 2
    cy = node.y + node.h // 2

    if tx > right:
        return (right, cy)
    if tx < left:
        return (left, cy)
    if ty > bottom:
        return (cx, bottom)
    return (cx, top)


def _draw_arrow(draw: ImageDraw.ImageDraw, p1: Tuple[int, int], p2: Tuple[int, int], color: str = "#334155") -> None:
    draw.line([p1, p2], fill=color, width=3)
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return
    import math

    ang = math.atan2(dy, dx)
    head_len = 12
    head_w = 7
    back_x = x2 - head_len * math.cos(ang)
    back_y = y2 - head_len * math.sin(ang)
    left = (
        back_x + head_w * math.cos(ang + math.pi / 2),
        back_y + head_w * math.sin(ang + math.pi / 2),
    )
    right = (
        back_x + head_w * math.cos(ang - math.pi / 2),
        back_y + head_w * math.sin(ang - math.pi / 2),
    )
    draw.polygon([p2, left, right], fill=color)


def _draw_edges(
    draw: ImageDraw.ImageDraw,
    nodes: Dict[str, Node],
    edges: List[Edge],
    font: ImageFont.ImageFont,
) -> None:
    for edge in edges:
        src = nodes[edge.src]
        dst = nodes[edge.dst]

        waypoints = list(edge.points)
        first_target = waypoints[0] if waypoints else (dst.x + dst.w // 2, dst.y + dst.h // 2)
        last_source = waypoints[-1] if waypoints else (src.x + src.w // 2, src.y + src.h // 2)

        start = _anchor_for_target(src, first_target)
        end = _anchor_for_target(dst, last_source)

        path = [start] + waypoints + [end]

        for i in range(len(path) - 1):
            seg_start = path[i]
            seg_end = path[i + 1]
            if i == len(path) - 2:
                _draw_arrow(draw, seg_start, seg_end)
            else:
                draw.line([seg_start, seg_end], fill="#334155", width=3)

        if edge.label:
            mx = (path[0][0] + path[-1][0]) // 2
            my = (path[0][1] + path[-1][1]) // 2
            tw = draw.textbbox((0, 0), edge.label, font=font)[2]
            th = draw.textbbox((0, 0), edge.label, font=font)[3]
            draw.rectangle((mx - tw // 2 - 4, my - th // 2 - 2, mx + tw // 2 + 4, my + th // 2 + 2), fill="#FFFFFF")
            draw.text((mx - tw // 2, my - th // 2), edge.label, fill="#1E293B", font=font)


def _draw_diagram(
    title: str,
    width: int,
    height: int,
    nodes: List[Node],
    edges: List[Edge],
    out_path: Path,
) -> None:
    img = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(img)

    title_font = _font(28)
    node_font = _font(16)
    edge_font = _font(14)

    draw.text((32, 20), title, fill="#0F172A", font=title_font)

    node_map: Dict[str, Node] = {n.node_id: n for n in nodes}

    _draw_edges(draw, node_map, edges, edge_font)

    for n in nodes:
        x1, y1, x2, y2 = n.x, n.y, n.x + n.w, n.y + n.h
        draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=n.fill, outline="#475569", width=2)
        _multiline_center(draw, (x1 + 8, y1 + 8, x2 - 8, y2 - 8), n.label, node_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _pipeline_layout() -> Tuple[List[Node], List[Edge], int, int]:
    nodes = [
        Node("Q", "User Query", 40, 90, 210, 64, "#DBEAFE"),
        Node("UI", "Streamlit UI\nsrc/ui/graph_web.py", 290, 80, 280, 92, "#DBEAFE"),
        Node("LG", "LangGraph App\nsrc/agent/graph.py", 620, 80, 270, 92, "#DBEAFE"),
        Node("RETN", "retrieve node", 940, 90, 220, 72, "#DBEAFE"),

        Node("META", "Extract Query Metadata\nlang + topic/intent hints", 80, 240, 260, 92),
        Node("VR", "Viking Router\ntaxonomy-aware scope", 390, 240, 240, 92),
        Node("VS", "Viking Scope\npatterns + confidence + trace", 680, 240, 280, 92),
        Node("VEC", "Vector Search (scoped)\npgvector SQL + lang/scope filters", 1010, 240, 290, 92),
        Node("FB", "Fallback Gate\nsoft & hits<min_hits\n& attempts<max_expansions", 1350, 220, 290, 130, "#FEF3C7"),
        Node("WIDEN", "Widen Scope\nphenomena->category\n->lang-only->all", 1540, 220, 260, 120, "#FEF3C7"),

        Node("BM25", "BM25 Search (global)\nkeyword candidate pool", 1010, 430, 290, 92),
        Node("PRE", "Mode Policy", 1220, 410, 190, 92, "#FEF3C7"),
        Node("BMF", "strict: BM25 pre-filter", 1460, 370, 240, 72, "#FEE2E2"),
        Node("BMS", "soft: skip pre-filter", 1460, 470, 240, 72, "#DCFCE7"),
        Node("FCHK", "Fusion Check\nhybrid enabled\nand BM25 > 0?", 1760, 410, 240, 92),
        Node("RRF", "RRF Fusion\nmerge vector+BM25\n(candidate merge stage)", 2060, 370, 260, 92),
        Node("VO", "Vector-only", 2060, 470, 220, 72),

        Node("SG", "Viking Scope Guard\nstrict: hard drop OOS\nsoft: drop OOS unless in-scope<3", 960, 610, 340, 120, "#E2E8F0"),
        Node("REC", "Recursive Expand\nL1->L0 (optional)", 1360, 620, 250, 86),
        Node("RR", "Reranker\n(optional)", 1660, 620, 210, 86),
        Node("SE", "Sibling Expansion\n(optional)", 1920, 620, 220, 86),
        Node("CTX", "Final Context Pack\ndoc ids + paths + snippets", 2180, 620, 250, 92, "#DBEAFE"),
        Node("NOTE", "Strict mode may return 0 docs.\nUI should warn and suggest\nsoft/widen tuning.", 2070, 520, 320, 96, "#FEF3C7"),

        Node("GD", "grade_documents\n(optional skip)", 960, 820, 250, 92),
        Node("DEC", "Relevant docs?", 1260, 830, 220, 72, "#FEF3C7"),
        Node("RW", "rewrite node", 1560, 940, 220, 72, "#FEE2E2"),
        Node("GEN", "generate node", 1560, 790, 220, 72, "#DCFCE7"),
        Node("HC", "check_hallucination\n(optional skip)", 1840, 790, 250, 92, "#DCFCE7"),
        Node("OUT", "Answer + References", 2140, 810, 240, 72, "#DBEAFE"),
    ]

    edges = [
        Edge("Q", "UI"),
        Edge("UI", "LG"),
        Edge("LG", "RETN"),
        Edge("RETN", "META"),
        Edge("META", "VR"),
        Edge("VR", "VS"),
        Edge("VS", "VEC"),
        Edge("VEC", "FB"),
        Edge("FB", "WIDEN", "yes"),
        Edge("WIDEN", "VEC", points=((1500, 620), (980, 620), (980, 286))),
        Edge("FB", "BM25", "no", points=((1360, 370), (1360, 456), (1170, 456))),
        Edge("BM25", "PRE"),
        Edge("PRE", "BMF", "strict"),
        Edge("PRE", "BMS", "soft"),
        Edge("BMF", "FCHK"),
        Edge("BMS", "FCHK"),
        Edge("FCHK", "RRF", "yes"),
        Edge("FCHK", "VO", "no"),
        Edge("RRF", "SG", points=((2200, 550), (1120, 550))),
        Edge("VO", "SG", points=((2170, 560), (1120, 560))),
        Edge("SG", "REC"),
        Edge("REC", "RR"),
        Edge("RR", "SE"),
        Edge("SE", "CTX"),
        Edge("CTX", "GD", points=((2290, 760), (1085, 760))),
        Edge("GD", "DEC"),
        Edge("DEC", "RW", "no", points=((1480, 866), (1560, 976))),
        Edge("DEC", "GEN", "yes"),
        Edge("RW", "RETN", points=((1670, 940), (1670, 140), (1160, 140))),
        Edge("GEN", "HC"),
        Edge("HC", "OUT"),
        Edge("VO", "NOTE", points=((2280, 520), (2070, 570))),
    ]

    return nodes, edges, 2520, 1120


def _harness_layout() -> Tuple[List[Node], List[Edge], int, int]:
    nodes = [
        Node("QS", "Query Set (JSONL/YAML)\nQ-only in late QA", 40, 140, 300, 96, "#DBEAFE"),
        Node("CP", "Condition Profiles\nbaseline/hybrid/viking_soft/...", 40, 300, 300, 96, "#FEF3C7"),
        Node("CFG", "Config Overlay (no UI)\npatch load_config/overrides", 40, 460, 300, 96, "#FEF3C7"),

        Node("HR", "Harness Runner\nsrc/eval/run_retrieval_ablation.py", 420, 270, 360, 110, "#DBEAFE"),
        Node("RET", "RAGRetriever.retrieve_documents()\n(retriever-core only)", 860, 270, 360, 110, "#E2E8F0"),

        Node("AUD", "audit.jsonl\nper query x condition\nscope/trace/fallback/docs/scores/timing", 1310, 160, 380, 130, "#DCFCE7"),
        Node("MET", "metrics.json\nrun meta: git/config/query hash\ncondition summaries", 1310, 360, 380, 130, "#DCFCE7"),
        Node("POST", "Optional post-process\nHit@k/MRR/contam/coverage", 1780, 270, 300, 100, "#DBEAFE"),
        Node("BK", "Artifact policy\nGit: small manifests/metrics\nBackup: large logs in ~/backupRAG/<run_id>", 2140, 250, 420, 140, "#DBEAFE"),
    ]

    edges = [
        Edge("QS", "HR"),
        Edge("CP", "HR"),
        Edge("CFG", "HR"),
        Edge("HR", "RET"),
        Edge("RET", "AUD"),
        Edge("RET", "MET"),
        Edge("AUD", "POST"),
        Edge("MET", "POST"),
        Edge("POST", "BK"),
    ]

    return nodes, edges, 2620, 760


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "rag_viking_pipeline.mmd").write_text(PIPELINE_MERMAID + "\n", encoding="utf-8")
    (OUT_DIR / "retrieval_harness.mmd").write_text(HARNESS_MERMAID + "\n", encoding="utf-8")

    p_nodes, p_edges, p_w, p_h = _pipeline_layout()
    _draw_diagram(
        "LTDB RAG/Viking Runtime Pipeline (Current)",
        p_w,
        p_h,
        p_nodes,
        p_edges,
        OUT_DIR / "rag_viking_pipeline.png",
    )

    h_nodes, h_edges, h_w, h_h = _harness_layout()
    _draw_diagram(
        "Retriever-Core Ablation Harness (Current)",
        h_w,
        h_h,
        h_nodes,
        h_edges,
        OUT_DIR / "retrieval_harness.png",
    )

    print("Wrote diagrams:")
    print(f"- {OUT_DIR / 'rag_viking_pipeline.mmd'}")
    print(f"- {OUT_DIR / 'rag_viking_pipeline.png'}")
    print(f"- {OUT_DIR / 'retrieval_harness.mmd'}")
    print(f"- {OUT_DIR / 'retrieval_harness.png'}")


if __name__ == "__main__":
    main()
