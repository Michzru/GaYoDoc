import sys
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch import Tensor
from typing import Sequence, Optional
import numpy as np
from bs4 import BeautifulSoup

from ..models.ocr import get_ocr_model
from ..utils.iou import crop_with_margin
from ..config import UNITABLE_DIR
from ..utils.iou import safe_crop
sys.path.insert(0, UNITABLE_DIR)

from ..models.unitable.src.utils import (
    subsequent_mask,
    pred_token_within_range,
    greedy_sampling,
    bbox_str_to_token_list,
    html_str_to_token_list,
    build_table_from_html_and_cell,
    html_table_template,
)
VALID_HTML_TOKEN = '<valid_html>'
VALID_BBOX_TOKEN = '<valid_bbox>'
INVALID_CELL_TOKEN = '<invalid_cell>'


def image_to_tensor(image: Image.Image, size, device) -> Tensor:
    T = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.86597056, 0.88463002, 0.87491087],
            std=[0.20686628, 0.18201602, 0.18485524]
        )
    ])
    return T(image).unsqueeze(0).to(device)


def rescale_bbox(bbox, src, tgt):
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    return [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]


def autoregressive_decode(
    model,
    image: Tensor,
    prefix: Sequence[int],
    max_decode_len: int,
    eos_id: int,
    device,
    token_whitelist: Optional[Sequence[int]] = None,
    token_blacklist: Optional[Sequence[int]] = None,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(image)
        context = torch.tensor(prefix, dtype=torch.int32).repeat(image.shape[0], 1).to(device)

    for _ in range(max_decode_len):
        eos_flag = (context == eos_id).any(dim=1)
        if eos_flag.all():
            break

        with torch.no_grad():
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(memory, context, tgt_mask=causal_mask, tgt_padding_mask=None)
            logits = model.generator(logits)[:, -1, :]

        logits = pred_token_within_range(
            logits.detach(),
            white_list=token_whitelist,
            black_list=token_blacklist,
        )

        next_probs, next_tokens = greedy_sampling(logits)
        context = torch.cat([context, next_tokens], dim=1)

    return context


def extract_cell_text_with_ocr(cell_image):
    model = get_ocr_model(True)
    cell_image_np = np.array(cell_image.convert("RGB"))
    results = model.readtext(cell_image_np, detail=0, paragraph=True)
    return " ".join(results).strip()

def run_table_extraction(data, pngs, verbose=True, gpu=True):
    from models.table import get_unitable_model

    vocab_struct, model_struct, device = get_unitable_model("structure", verbose=verbose, gpu=gpu)
    vocab_bbox,   model_bbox,   device = get_unitable_model("bbox",      verbose=verbose, gpu=gpu)

    # We use DOCTR OCR
    #vocab_content,model_content,device = get_unitable_model("content",   verbose=verbose, gpu=gpu)

    table_tasks = []
    for page_idx, page in enumerate(data["pages"]):
        for node in page["nodes"]:
            if node.get("predicted_label_id") == 2:
                table_tasks.append((page_idx, node))

    iterator = tqdm(
        table_tasks,
        total=len(table_tasks),
        desc="Table extraction",
        disable=not verbose
    )

    for page_idx, node in iterator:
        pil_image = pngs[page_idx]

        x1, y1, x2, y2 = node["geometry"]["absolute_pixel_coords"]

        # Crop with margin
        table_crop, new_coords = crop_with_margin(
            pil_image,
            (x1, y1, x2, y2),
            margin_ratio=0.07
        )

        try:
            # 1. TSR with Unitable
            image_tensor = image_to_tensor(table_crop, (448, 448), device)

            pred_html = autoregressive_decode(
                model=model_struct,
                image=image_tensor,
                prefix=[vocab_struct.token_to_id("[html]")],
                max_decode_len=512,
                eos_id=vocab_struct.token_to_id("<eos>"),
                device=device,
                token_whitelist=[vocab_struct.token_to_id(i) for i in VALID_HTML_TOKEN],
            )

            pred_html = pred_html.detach().cpu().numpy()[0]
            pred_html = vocab_struct.decode(pred_html, skip_special_tokens=False)
            pred_html = html_str_to_token_list(pred_html)

            # 2. BBOX with Unitable
            pred_bbox = autoregressive_decode(
                model=model_bbox,
                image=image_tensor,
                prefix=[vocab_bbox.token_to_id("[bbox]")],
                max_decode_len=1024,
                eos_id=vocab_bbox.token_to_id("<eos>"),
                device=device,
                token_whitelist=[vocab_bbox.token_to_id(i) for i in VALID_BBOX_TOKEN[:449]],
            )

            pred_bbox = pred_bbox.detach().cpu().numpy()[0]
            pred_bbox = vocab_bbox.decode(pred_bbox, skip_special_tokens=False)
            pred_bbox = bbox_str_to_token_list(pred_bbox)

            pred_bbox_rescaled = rescale_bbox(
                pred_bbox,
                src=(448, 448),
                tgt=table_crop.size
            )

            # 3. CELLS with DOCTR replaced for Unitable OCR
            pred_cell = []
            for bbox in pred_bbox_rescaled:
                cropped = safe_crop(table_crop, bbox)
                if cropped is not None:
                    cell_text = extract_cell_text_with_ocr(cropped)
                    pred_cell.append(cell_text)

            # 4. HTML
            pred_code = build_table_from_html_and_cell(pred_html, pred_cell)
            pred_code = "".join(pred_code)
            pred_code = html_table_template(pred_code)

            print(f"\n=== TABLE Page {page_idx + 1}, node {node['node_id']} ===")
            soup = BeautifulSoup(pred_code, "html.parser")
            print(soup.prettify())
            print("=====================================\n")

            node["table_data"] = {
                "html": pred_code,
                "bbox": pred_bbox_rescaled,
                "crop_coords": list(new_coords),
            }

        except Exception as e:
            if verbose:
                print(f"[Table] Page {page_idx + 1}, node {node['node_id']}: {e}")
            node["table_data"] = None

    return data