import os
import json
import joblib
import html

# Configuration
json_path = "data/document_graph_with_edges.json"
cache_path = "data/cached_images.joblib"
output_dir = "data/interactive_pages"

os.makedirs(output_dir, exist_ok=True)

print("Loading JSON a images from cache...")
with open(json_path, 'r', encoding='utf-8') as f:
    document_data = json.load(f)

images = joblib.load(cache_path)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="sk">
<head>
    <meta charset="UTF-8">
    <title>Strana {page_number}</title>
    <style>
        body {{ background-color: #f0f0f0; padding: 20px; font-family: sans-serif; }}
        .page-container {{ position: relative; display: inline-block; box-shadow: 0 4px 8px rgba(0,0,0,0.2); background: white; }}
        .page-container img {{ display: block; max-width: 1000px; height: auto; }}
        .edges-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 5; }}
        .bbox {{ position: absolute; border: 2px solid rgba(255, 0, 0, 0.4); cursor: pointer; box-sizing: border-box; transition: background-color 0.2s, border-color 0.2s; z-index: 6; }}
        .bbox:hover {{ background-color: rgba(255, 0, 0, 0.2); border-color: rgba(255, 0, 0, 1); z-index: 10; }}
        .tooltip {{ visibility: hidden; position: absolute; bottom: 105%; left: 0; background-color: #333; color: #fff; padding: 10px; border-radius: 5px; width: max-content; max-width: 400px; white-space: pre-wrap; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-size: 14px; opacity: 0; transition: opacity 0.2s; pointer-events: none; }}
        .bbox:hover .tooltip {{ visibility: visible; opacity: 1; }}
        .tag-label {{ font-weight: bold; color: #ff9999; margin-bottom: 5px; display: block; }}
        .nav {{ margin-bottom: 20px; }}
        .nav a {{ padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px; }}
        .nav a:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <div class="nav">{nav_links}</div>
    <h2>Strana {page_number} / {total_pages}</h2>
    <div class="page-container">
        <img src="image_{page_number}.png" alt="Strana {page_number}">
        <svg class="edges-overlay">{svg_edges_html}</svg>
        {boxes_html}
    </div>
</body>
</html>
"""

total_pages = document_data["metadata"]["total_pages"]

for page_data in document_data["pages"]:
    page_number = page_data["page_number"]
    page_index = page_number - 1

    if page_index >= len(images):
        continue

    print(f"Generating html page {page_number}...")

    img_filename = f"image_{page_number}.png"
    img_path = os.path.join(output_dir, img_filename)
    images[page_index].save(img_path)

    boxes_html = ""
    nodes_list = page_data.get("nodes", [])

    # 1. Genereting bboxes (nodes)
    for node in nodes_list:
        norm_x1, norm_y1, norm_x2, norm_y2 = node["geometry"]["normalized_coords"]
        left_pct = norm_x1 * 100
        top_pct = norm_y1 * 100
        width_pct = (norm_x2 - norm_x1) * 100
        height_pct = (norm_y2 - norm_y1) * 100

        label = html.escape(node["label"])
        confidence = node["yolo_confidence"]
        extracted_text = html.escape(node["text"]) if node.get("text") else "<i>(No text)</i>"

        box_div = f"""
        <div class="bbox" style="left: {left_pct}%; top: {top_pct}%; width: {width_pct}%; height: {height_pct}%;">
            <div class="tooltip">
                <span class="tag-label">[{label}] - Confidence: {confidence:.2f}</span>
                {extracted_text}
            </div>
        </div>
        """
        boxes_html += box_div

    # 2. Edge drawing
    svg_edges_html = ""
    edges_list = page_data.get("edges", [])

    best_edges_for_objects = {}

    for edge in edges_list:
        src_idx = edge.get("source", edge.get("from"))
        dst_idx = edge.get("target", edge.get("to"))
        prob = edge.get("probability", 1.0)

        if src_idx is None or dst_idx is None or src_idx >= len(nodes_list) or dst_idx >= len(nodes_list):
            continue

        label_src = nodes_list[src_idx]["label"]
        label_dst = nodes_list[dst_idx]["label"]

        obj_idx = None
        cap_idx = None

        if label_src in ["Picture", "Table"] and label_dst == "Caption":
            obj_idx, cap_idx = src_idx, dst_idx
        elif label_dst in ["Picture", "Table"] and label_src == "Caption":
            obj_idx, cap_idx = dst_idx, src_idx


        if obj_idx is not None and prob > 0.3:
            if obj_idx not in best_edges_for_objects or prob > best_edges_for_objects[obj_idx]["probability"]:
                best_edges_for_objects[obj_idx] = {
                    "source": obj_idx,
                    "target": cap_idx,
                    "probability": prob
                }

    for obj_idx, best_edge in best_edges_for_objects.items():
        src_node = nodes_list[best_edge["source"]]
        dst_node = nodes_list[best_edge["target"]]

        src_cx, src_cy = src_node["geometry"]["normalized_center"]
        dst_cx, dst_cy = dst_node["geometry"]["normalized_center"]

        x1_pct = src_cx * 100
        y1_pct = src_cy * 100
        x2_pct = dst_cx * 100
        y2_pct = dst_cy * 100

        line_html = f'<line x1="{x1_pct}%" y1="{y1_pct}%" x2="{x2_pct}%" y2="{y2_pct}%" stroke="#28a745" stroke-width="3" />\n'

        prob_text = f"{(best_edge['probability'] * 100):.1f}%"
        mid_x = (x1_pct + x2_pct) / 2
        mid_y = (y1_pct + y2_pct) / 2
        line_html += f'<text x="{mid_x}%" y="{mid_y}%" fill="blue" font-size="12px" font-weight="bold">{prob_text}</text>\n'

        svg_edges_html += line_html

    prev_link = f'<a href="page_{page_number - 1}.html">&laquo; Predchádzajúca</a>' if page_number > 1 else ''
    next_link = f'<a href="page_{page_number + 1}.html">Ďalšia &raquo;</a>' if page_number < total_pages else ''
    nav_links = f"{prev_link} {next_link}"

    final_html = HTML_TEMPLATE.format(
        page_number=page_number,
        total_pages=total_pages,
        nav_links=nav_links,
        boxes_html=boxes_html,
        svg_edges_html=svg_edges_html
    )

    html_filepath = os.path.join(output_dir, f"page_{page_number}.html")
    with open(html_filepath, 'w', encoding='utf-8') as f:
        f.write(final_html)

print(f"\nDone! Saved in '{output_dir}'")