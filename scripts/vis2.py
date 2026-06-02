import os
import json
import html
import base64
import re

from pdf2image import convert_from_path

pdf_path   = "../data/ISLP_website.pdf"
json_path  = "../data/outputs/100_stran_ISLP.json"
output_dir = "../data/islp_debug_docling"

os.makedirs(output_dir, exist_ok=True)

print("Loading JSON and images from cache...")
with open(json_path, "r", encoding="utf-8") as f:
    document_data = json.load(f)

images = convert_from_path(pdf_path)

CATEGORY_COLORS = {
    "Caption":        "#a78bfa",
    "Picture":        "#fb923c",
    "Table":          "#38bdf8",
    "Formula":        "#34d399",
    "Section-header": "#f472b6",
    "Page-footer":    "#94a3b8",
    "Page-header":    "#fbbf24",
    "Text":           "#6ee7b7",
    "Other":          "#6b7280",
    "Unknown":        "#374151",
}


def conf_pct(val):
    if val is None: return "—"
    return f"{val * 100:.0f}%"


def get_node_label(node):
    """Extract GAT predicted label and original Docling label."""
    gat     = node.get("predicted_label") or node.get("gat_reclassified_label") or node.get("label", "Unknown")
    docling = node.get("label", "Unknown")
    return gat, docling


def get_gat_conf(node):
    c = node.get("predicted_confidence") or node.get("gat_confidence")
    return float(c) if c is not None else None


def build_node_id_map(nodes_list):
    return {n["node_id"]: i for i, n in enumerate(nodes_list)}


def build_legend_classes():
    parts = []
    for label, color in CATEGORY_COLORS.items():
        if label == "Unknown": continue
        parts.append(
            f'<span class="lchip">'
            f'<span class="lchip-dot" style="background:{color}"></span>{label}'
            f'</span>'
        )
    return " ".join(parts)


LEGEND_CLASSES_HTML = build_legend_classes()

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Page {page_number} — Doc Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:      #090c10;
  --panel:   #0d1117;
  --border:  #1c2128;
  --border2: #2d333b;
  --text:    #cdd9e5;
  --muted:   #545d68;
  --accent:  #539bf5;
  --mono:    'JetBrains Mono', monospace;
  --sans:    'Syne', sans-serif;
  --changed: #f0883e;
  --agreed:  #3fb950;
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 13px; line-height: 1.5; min-height: 100vh; }}

.topbar {{
  position: sticky; top: 0; z-index: 300;
  background: rgba(9,12,16,0.96); backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  height: 46px; display: flex; align-items: center; gap: 12px; padding: 0 18px;
}}
.brand {{ font-family: var(--mono); font-size: 11px; font-weight: 700; color: var(--accent); letter-spacing: .12em; text-transform: uppercase; }}
.nav-btns {{ display: flex; gap: 5px; }}
.nav-btns a {{
  font-family: var(--mono); font-size: 11px; padding: 4px 11px;
  background: var(--panel); border: 1px solid var(--border2);
  color: var(--text); text-decoration: none; border-radius: 3px; transition: all .15s;
}}
.nav-btns a:hover {{ border-color: var(--accent); color: var(--accent); background: rgba(83,155,245,0.08); }}
.topbar-page {{ margin-left: auto; font-family: var(--mono); font-size: 10px; color: var(--muted); }}

.legend {{ background: var(--panel); border-bottom: 1px solid var(--border); padding: 7px 18px; display: flex; flex-wrap: wrap; align-items: center; gap: 5px 12px; }}
.legend-label {{ font-family: var(--mono); font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
.lchip {{ display: inline-flex; align-items: center; gap: 5px; font-size: 11px; font-family: var(--mono); padding: 2px 7px; border-radius: 3px; background: rgba(255,255,255,0.03); border: 1px solid var(--border2); }}
.lchip-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}

.diff-legend {{ background: var(--bg); border-bottom: 1px solid var(--border); padding: 6px 18px; display: flex; align-items: center; gap: 16px; }}
.diff-item {{ display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 10px; color: var(--muted); }}
.diff-box-agreed {{ width: 14px; height: 14px; border: 2px solid var(--agreed); border-radius: 2px; }}
.diff-box-changed {{ width: 14px; height: 14px; border: 2px dashed var(--changed); border-radius: 2px; }}
.diff-box-table {{ width: 14px; height: 14px; border: 2px solid #38bdf8; border-radius: 2px; display: flex; align-items: center; justify-content: center; font-size: 8px; color: #38bdf8; }}

.statsbar {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); background: var(--panel); overflow-x: auto; }}
.stat {{ padding: 7px 16px; border-right: 1px solid var(--border); font-family: var(--mono); font-size: 11px; color: var(--muted); white-space: nowrap; }}
.stat b {{ color: var(--text); font-weight: 700; }}
.hi {{ color: #3fb950; }} .mid {{ color: #d29922; }} .lo {{ color: #f85149; }}

.canvas-wrap {{ padding: 20px; display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 20px; }}
.page-container {{ position: relative; display: inline-block; flex-shrink: 0; border: 1px solid var(--border2); border-radius: 4px; overflow: visible; box-shadow: 0 8px 48px rgba(0,0,0,0.7); }}
.page-container img {{ display: block; max-width: 980px; width: 100%; height: auto; border-radius: 3px; }}
.edges-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 5; overflow: visible; }}

.table-side-panel {{ display: none; flex-direction: column; width: max-content; min-width: 500px; max-width: 900px; background: #ffffff; border-radius: 8px; box-shadow: 0 8px 48px rgba(0,0,0,0.5); position: sticky; top: 66px; }}
.table-side-panel.visible {{ display: flex; }}
.tsp-header {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; background: #f0f4ff; border-bottom: 1px solid #dde3f0; flex-shrink: 0; border-radius: 8px 8px 0 0; }}
.tsp-title {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #1e2a3a; letter-spacing: .04em; display: flex; align-items: center; gap: 6px; }}
.tsp-title-icon {{ color: #38bdf8; font-size: 14px; }}
.tsp-actions {{ display: flex; gap: 6px; align-items: center; }}
.tsp-btn {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; padding: 5px 12px; border-radius: 5px; border: none; cursor: pointer; transition: all .15s; letter-spacing: .03em; }}
.tsp-btn-copy {{ background: #2563eb; color: #fff; }}
.tsp-btn-copy:hover {{ background: #1d4ed8; }}
.tsp-btn-copy.copied {{ background: #16a34a; }}
.tsp-btn-close {{ background: #e5e7eb; color: #374151; padding: 5px 9px; }}
.tsp-btn-close:hover {{ background: #d1d5db; }}
.tsp-body {{ padding: 12px; background: #fff; border-radius: 0 0 8px 8px; }}
.tsp-body table {{ border-collapse: collapse; font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #111; width: 100%; }}
.tsp-body th, .tsp-body td {{ border: 1px solid #ccc; padding: 5px 9px; text-align: left; vertical-align: top; white-space: nowrap; }}
.tsp-body thead tr {{ background: #f0f4ff; }}
.tsp-body tbody tr:nth-child(even) {{ background: #fafafa; }}
.tsp-empty {{ padding: 24px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #9ca3af; }}

.bbox {{ position: absolute; border-radius: 2px; cursor: pointer; z-index: 10; transition: background .1s, box-shadow .1s; }}
.bbox:hover {{ z-index: 50; }}
.bbox.agreed  {{ border: 2px solid; }}
.bbox.changed {{ border: 2px dashed; }}
.bbox.is-table {{ cursor: pointer; border: 2px solid #38bdf8 !important; z-index: 30 !important; }}
.bbox.is-table:hover {{ z-index: 60 !important; box-shadow: 0 0 0 3px rgba(56,189,248,0.45), 0 0 16px rgba(56,189,248,0.2); background: rgba(56,189,248,0.06) !important; }}
.bbox.is-table.active {{ z-index: 60 !important; box-shadow: 0 0 0 3px rgba(56,189,248,0.7), 0 0 24px rgba(56,189,248,0.3); background: rgba(56,189,248,0.10) !important; }}

.bbox-chip {{ position: absolute; top: -26px; left: -1px; display: flex; align-items: center; gap: 3px; pointer-events: none; z-index: 51; white-space: nowrap; }}
.chip {{ font-family: var(--mono); font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 2px; color: #fff; letter-spacing: .02em; }}
.chip-old {{ opacity: 0.55; text-decoration: line-through; font-size: 8.5px; }}
.chip-arrow {{ font-size: 9px; color: var(--changed); opacity: 0.8; }}
.table-icon {{ font-family: var(--mono); font-size: 10px; padding: 2px 5px; background: rgba(56,189,248,0.18); border: 1px solid #38bdf8; border-radius: 2px; color: #38bdf8; font-weight: 700; }}

.tooltip {{ visibility: hidden; opacity: 0; position: absolute; top: calc(100% + 8px); left: 0; z-index: 200; width: 270px; background: var(--panel); border: 1px solid var(--border2); border-radius: 6px; padding: 12px; box-shadow: 0 10px 32px rgba(0,0,0,0.7); pointer-events: none; transition: opacity .1s; }}
.bbox:hover .tooltip {{ visibility: visible; opacity: 1; }}
.tt-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }}
.tt-badge {{ font-family: var(--mono); font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 3px; color: #fff; }}
.tt-tag {{ font-family: var(--mono); font-size: 8px; padding: 1px 5px; border-radius: 2px; margin-left: auto; }}
.tt-changed {{ background: rgba(240,136,62,0.2); border: 1px solid var(--changed); color: var(--changed); }}
.tt-agreed  {{ background: rgba(63,185,80,0.15); border: 1px solid var(--agreed);  color: var(--agreed);  }}

.diff-block {{ background: rgba(255,255,255,0.03); border: 1px solid var(--border2); border-radius: 4px; padding: 8px 10px; margin-bottom: 8px; }}
.diff-block-title {{ font-family: var(--mono); font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }}
.diff-line {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
.diff-model {{ font-family: var(--mono); font-size: 9px; color: var(--muted); width: 46px; flex-shrink: 0; }}
.diff-label-pill {{ font-family: var(--mono); font-size: 9px; font-weight: 700; padding: 1px 6px; border-radius: 2px; color: #fff; }}
.diff-conf {{ font-family: var(--mono); font-size: 9px; color: var(--muted); margin-left: auto; }}
.diff-bar-wrap {{ flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }}
.diff-bar {{ height: 100%; border-radius: 2px; }}

.tt-divider {{ border: none; border-top: 1px solid var(--border); margin: 8px 0; }}
.tt-text {{ font-size: 10px; color: var(--muted); font-style: italic; max-height: 55px; overflow: hidden; line-height: 1.55; }}
</style>
</head>
<body>

<div class="topbar">
  <span class="brand">▸ DocGraph</span>
  <div class="nav-btns">{nav_links}</div>
  <span class="topbar-page">Page {page_number} / {total_pages}</span>
</div>

<div class="legend">
  <span class="legend-label">Classes:</span>
  {legend_classes}
</div>

<div class="diff-legend">
  <span class="diff-item"><span class="diff-box-agreed"></span>GAT = Docling (Agreed)</span>
  <span class="diff-item"><span class="diff-box-changed"></span>GAT ≠ Docling (GAT Changed)</span>
  <span class="diff-item"><span class="diff-box-table">⊞</span>Click = Show Table</span>
</div>

<div class="statsbar">
  {stats_html}
</div>

<div class="canvas-wrap">
  <div class="page-container">
    <img src="image_{page_number}.png" alt="Page {page_number}">
    <svg class="edges-svg">
      <defs>
        <marker id="arrow-fig" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <polygon points="0 0, 8 4, 0 8" fill="#fb923c"/>
        </marker>
        <marker id="arrow-tab" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <polygon points="0 0, 8 4, 0 8" fill="#38bdf8"/>
        </marker>
        <marker id="arrow-default" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <polygon points="0 0, 8 4, 0 8" fill="#ef4444"/>
        </marker>
      </defs>
      {svg_edges_html}
    </svg>
    {boxes_html}
  </div>

  <div class="table-side-panel" id="tableSidePanel">
    <div class="tsp-header">
      <div class="tsp-title">
        <span class="tsp-title-icon">⊞</span>
        <span id="tspTitle">Table</span>
      </div>
      <div class="tsp-actions">
        <button class="tsp-btn tsp-btn-copy" id="tspCopyBtn" onclick="copyTable()">📋 Copy Table</button>
        <button class="tsp-btn tsp-btn-close" onclick="closeSidePanel()">✕</button>
      </div>
    </div>
    <div class="tsp-body" id="tspBody"></div>
  </div>
</div>

<script>
const sidePanel  = document.getElementById("tableSidePanel");
const tspBody    = document.getElementById("tspBody");
const tspTitle   = document.getElementById("tspTitle");
const tspCopyBtn = document.getElementById("tspCopyBtn");

let currentTableHtml = "";
let activeBox = null;

function showSidePanel(boxEl) {{
  const b64Html   = boxEl.getAttribute("data-table-b64") || "";
  const titleText = boxEl.getAttribute("data-table-title") || "Table";
  let tableHtml = "";
  if (b64Html) {{
    try {{ tableHtml = decodeURIComponent(escape(window.atob(b64Html))); }}
    catch (e) {{ console.error("Error decoding Base64 table data:", e); }}
  }}
  if (activeBox && activeBox !== boxEl) activeBox.classList.remove("active");
  if (activeBox === boxEl && sidePanel.classList.contains("visible")) {{ closeSidePanel(); return; }}
  activeBox = boxEl;
  boxEl.classList.add("active");
  currentTableHtml = tableHtml;
  tspTitle.textContent = titleText;
  tspBody.innerHTML = tableHtml || '<div class="tsp-empty">Table does not contain extracted data.</div>';
  sidePanel.classList.add("visible");
  resetCopyBtn();
}}

function closeSidePanel() {{
  sidePanel.classList.remove("visible");
  if (activeBox) {{ activeBox.classList.remove("active"); activeBox = null; }}
  currentTableHtml = "";
}}

function resetCopyBtn() {{
  tspCopyBtn.textContent = "📋 Copy Table";
  tspCopyBtn.classList.remove("copied");
}}

async function copyTable() {{
  if (!currentTableHtml) return;
  const fullHtml = `<html><body>${{currentTableHtml}}</body></html>`;
  try {{
    const blob = new Blob([fullHtml], {{ type: "text/html" }});
    await navigator.clipboard.write([new ClipboardItem({{ "text/html": blob }})]);
  }} catch (e) {{
    const ta = document.createElement("textarea");
    ta.value = fullHtml; ta.style.cssText = "position:fixed;opacity:0;pointer-events:none;";
    document.body.appendChild(ta); ta.select(); document.execCommand("copy"); document.body.removeChild(ta);
  }}
  tspCopyBtn.textContent = "✓ Copied!";
  tspCopyBtn.classList.add("copied");
  setTimeout(resetCopyBtn, 2200);
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
total_pages = document_data["metadata"]["total_pages"]

for page_data in document_data["pages"]:
    page_number = page_data["page_number"]
    page_index  = page_number - 1

    if page_index >= len(images):
        continue

    print(f"Generating page {page_number}...")

    images[page_index].save(os.path.join(output_dir, f"image_{page_number}.png"))

    nodes_list   = page_data.get("nodes", [])
    edges_list   = page_data.get("edges", [])
    node_id_map  = build_node_id_map(nodes_list)

    # ── METRICS ──
    changed_nodes = [n for n in nodes_list if get_node_label(n)[0] != get_node_label(n)[1]]
    agreed_nodes  = [n for n in nodes_list if get_node_label(n)[0] == get_node_label(n)[1]]
    table_nodes   = [n for n in nodes_list if get_node_label(n)[0] == "Table"]

    high_e = sum(1 for e in edges_list if e.get("confidence", 0) >= 0.7)
    mid_e  = sum(1 for e in edges_list if 0.5 <= e.get("confidence", 0) < 0.7)
    low_e  = sum(1 for e in edges_list if e.get("confidence", 0) < 0.5)

    gat_confs = [c for n in nodes_list if (c := get_gat_conf(n)) is not None]
    avg_gat   = f"{sum(gat_confs) / len(gat_confs) * 100:.0f}%" if gat_confs else "—"

    stats_html = (
        f'<div class="stat">Nodes: <b>{len(nodes_list)}</b></div>'
        f'<div class="stat hi">✓ Agreed: <b>{len(agreed_nodes)}</b></div>'
        f'<div class="stat" style="color:var(--changed)">✦ GAT Changed: <b>{len(changed_nodes)}</b></div>'
        f'<div class="stat" style="color:#38bdf8">⊞ Tables: <b>{len(table_nodes)}</b></div>'
        f'<div class="stat">Edges: <b>{len(edges_list)}</b></div>'
        f'<div class="stat hi">▲ ≥70%: <b>{high_e}</b></div>'
        f'<div class="stat mid">● 50–70%: <b>{mid_e}</b></div>'
        f'<div class="stat lo">▼ &lt;50%: <b>{low_e}</b></div>'
        f'<div class="stat">Ø GAT Conf: <b>{avg_gat}</b></div>'
    )

    # ── BOUNDING BOXES ──
    boxes_html    = ""
    table_counter = 0

    for node in nodes_list:
        nx1, ny1, nx2, ny2 = node["geometry"]["normalized_coords"]
        lp = nx1 * 100
        tp = ny1 * 100
        wp = (nx2 - nx1) * 100
        hp = (ny2 - ny1) * 100

        gat_label, docling_label = get_node_label(node)          # zmenené
        changed        = (gat_label != docling_label)
        is_table       = (gat_label == "Table")
        box_color      = CATEGORY_COLORS.get(gat_label,     CATEGORY_COLORS["Unknown"])
        docling_color  = CATEGORY_COLORS.get(docling_label, CATEGORY_COLORS["Unknown"])  # zmenené
        gat_conf_val     = get_gat_conf(node)
        docling_conf_val = node.get("confidence")                 # zmenené

        def build_bar(val, color):
            pct = int((val or 0) * 100)
            return (
                f'<div class="diff-bar-wrap">'
                f'<div class="diff-bar" style="width:{pct}%;background:{color};"></div>'
                f'</div>'
            )

        # ── chip ──
        chip_html = '<div class="bbox-chip">'
        if changed:
            chip_html += (
                f'<span class="chip chip-old" style="background:{docling_color}40;color:{docling_color};border:1px solid {docling_color}60;">'
                f'{docling_label} {conf_pct(docling_conf_val)}</span>'
                f'<span class="chip-arrow">→</span>'
            )
        chip_html += f'<span class="chip" style="background:{box_color};">{gat_label} {conf_pct(gat_conf_val)}</span>'
        chip_html += '</div>'

        # ── status tag ──
        status_tag = (
            f'<span class="tt-tag tt-changed">GAT CHANGED</span>' if changed
            else f'<span class="tt-tag tt-agreed">AGREED</span>'
        )

        # ── diff block ──
        diff_block = (
            f'<div class="diff-block">'
            f'<div class="diff-block-title">{"Classification — DIFF" if changed else "Classification"}</div>'
            f'<div class="diff-line">'
            f'<span class="diff-model">GAT</span>'
            f'<span class="diff-label-pill" style="background:{box_color};">{gat_label}</span>'
            f'{build_bar(gat_conf_val, box_color)}'
            f'<span class="diff-conf">{conf_pct(gat_conf_val)}</span>'
            f'</div>'
            f'<div class="diff-line" style="{"opacity:0.5;" if not changed else ""}">'
            f'<span class="diff-model">Docling</span>'                          # zmenené
            f'<span class="diff-label-pill" style="background:{docling_color};">{docling_label}</span>'
            f'{build_bar(docling_conf_val, docling_color)}'
            f'<span class="diff-conf">{conf_pct(docling_conf_val)}</span>'
            f'</div>'
            f'</div>'
        )

        extracted_text = html.escape(node.get("text", "") or "")
        text_preview   = f'<div class="tt-text">{extracted_text or "<i>(no text)</i>"}</div>'

        tooltip = (
            f'<div class="tooltip">'
            f'<div class="tt-row">'
            f'<span class="tt-badge" style="background:{box_color};">{gat_label}</span>'
            f'<span style="font-family:var(--mono);font-size:9px;color:var(--muted)">node #{node.get("node_id","?")}</span>'
            f'{status_tag}'
            f'</div>'
            f'{diff_block}'
            f'<hr class="tt-divider">'
            f'{text_preview}'
            f'</div>'
        )

        if is_table:
            table_counter += 1

            table_data = node.get("table_data")
            raw_html   = ""
            if isinstance(table_data, dict) and "html" in table_data:
                raw_html = table_data["html"]
            elif isinstance(table_data, str):
                raw_html = table_data

            table_html = ""
            if raw_html:
                match = re.search(r'<table[\s\S]*?</table>', raw_html, re.IGNORECASE)
                table_html = match.group(0).replace('100%%', '100%') if match else raw_html

            if not table_html and node.get("text"):
                table_html = f"<pre style='font-family:var(--mono);font-size:11px;white-space:pre-wrap;'>{html.escape(node.get('text'))}</pre>"

            b64_table = base64.b64encode(table_html.encode('utf-8')).decode('utf-8') if table_html else ""

            boxes_html += (
                f'<div class="bbox is-table"'
                f' style="left:{lp:.3f}%;top:{tp:.3f}%;width:{wp:.3f}%;height:{hp:.3f}%;border-color:#38bdf8;"'
                f' data-table-b64="{b64_table}"'
                f' data-table-title="Table #{table_counter}"'
                f' onclick="showSidePanel(this)">'
                f'{chip_html}{tooltip}'
                f'</div>\n'
            )
        else:
            boxes_html += (
                f'<div class="bbox {"changed" if changed else "agreed"}"'
                f' style="left:{lp:.3f}%;top:{tp:.3f}%;width:{wp:.3f}%;height:{hp:.3f}%;border-color:{box_color};">'
                f'{chip_html}{tooltip}'
                f'</div>\n'
            )

    # ── SVG EDGES ──
    svg_edges_html = ""
    best_edges     = {}

    for edge in edges_list:
        prob     = edge.get("confidence", 1.0)
        rel_type = edge.get("relation_type")

        if prob < 0.4:
            continue

        src_raw = edge.get("source")
        dst_raw = edge.get("target")
        if src_raw is None or dst_raw is None:
            continue

        cap_idx = node_id_map.get(src_raw)
        obj_idx = node_id_map.get(dst_raw)
        if cap_idx is None or obj_idx is None:
            continue

        if rel_type == 2:
            edge_color, marker_id, text_color = "#38bdf8", "arrow-tab", "#bae6fd"
        elif rel_type == 1:
            edge_color, marker_id, text_color = "#fb923c", "arrow-fig", "#fed7aa"
        else:
            edge_color, marker_id, text_color = "#ef4444", "arrow-default", "#fca5a5"

        if obj_idx not in best_edges or prob > best_edges[obj_idx]["prob"]:
            best_edges[obj_idx] = {
                "src": cap_idx, "dst": obj_idx, "prob": prob,
                "color": edge_color, "marker": marker_id, "tcolor": text_color
            }

    for obj_idx, ed in best_edges.items():
        sn   = nodes_list[ed["src"]]
        dn   = nodes_list[ed["dst"]]
        prob = ed["prob"]
        sc   = sn["geometry"]["normalized_center"]
        dc   = dn["geometry"]["normalized_center"]
        x1, y1 = sc[0] * 100, sc[1] * 100
        x2, y2 = dc[0] * 100, dc[1] * 100
        mx, my  = (x1 + x2) / 2, (y1 + y2) / 2
        opacity = round(max(0.40, prob), 2)

        svg_edges_html += (
            f'<line x1="{x1:.2f}%" y1="{y1:.2f}%" x2="{x2:.2f}%" y2="{y2:.2f}%"'
            f' stroke="{ed["color"]}" stroke-width="2.5" stroke-dasharray="5,4"'
            f' opacity="{opacity}" stroke-linecap="round" marker-end="url(#{ed["marker"]})"/>\n'
            f'<rect x="calc({mx:.2f}% - 16px)" y="calc({my:.2f}% - 9px)" width="32" height="18" rx="4" fill="#0d1117" opacity="0.95" stroke="{ed["color"]}" stroke-width="1"/>\n'
            f'<text x="{mx:.2f}%" y="{my:.2f}%" fill="{ed["tcolor"]}" font-size="9" font-family="JetBrains Mono,monospace"'
            f' font-weight="700" text-anchor="middle" dominant-baseline="central">{prob * 100:.0f}%</text>\n'
        )

    # ── NAVIGATION ──
    prev_link = f'<a href="page_{page_number - 1}.html">&larr; Prev</a>' if page_number > 1 else ""
    next_link = f'<a href="page_{page_number + 1}.html">Next &rarr;</a>' if page_number < total_pages else ""

    final_html = HTML_TEMPLATE.format(
        page_number=page_number,
        total_pages=total_pages,
        nav_links=f"{prev_link} {next_link}",
        legend_classes=LEGEND_CLASSES_HTML,
        stats_html=stats_html,
        svg_edges_html=svg_edges_html,
        boxes_html=boxes_html,
    )

    with open(os.path.join(output_dir, f"page_{page_number}.html"), "w", encoding="utf-8") as f:
        f.write(final_html)

print(f"\nDone! Saved to: '{output_dir}'")