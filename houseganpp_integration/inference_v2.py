"""
House-GAN++ Inference Module V2 (Research Grade)
=================================================

Implements the House-GAN++ inference pipeline with:
    1. Iterative refinement matching the original paper (test.py)
    2. Post-processing for mask quality (connected components, hole filling,
       overlap resolution)
    3. Best-of-N generation with architectural quality scoring
    4. Professional rendering with room labels and legend

Key changes from baseline:
    - ``post_process_masks`` removes artefacts and resolves overlap
    - ``generate_best_of_n`` scores N candidates and returns the best
    - ``score_floor_plan`` evaluates area balance, coverage, shape,
      overlap, and adjacency fidelity on a 0–100 scale
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import datetime

try:
    import webcolors
except ImportError:
    webcolors = None

from .models import Generator, load_generator
from .requirements_converter import RequirementsConverter
from .template_selector import TemplateSelector, create_edges_from_ed_rm
from .architectural_constraints import (
    REAL_ROOM_TYPES, DOOR_TYPES, ROOM_DISPLAY_NAMES,
    post_process_masks, score_floor_plan, validate_masks,
)


# ============================================================================
# Constants
# ============================================================================

ROOM_CLASS = {
    "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4,
    "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
    "storage": 10, "front door": 15, "unknown": 16, "interior_door": 17,
}
CLASS_ROOM = {v: k for k, v in ROOM_CLASS.items()}

# Colour palette (matching original House-GAN++)
ID_COLOR = {
    1: '#EE4D4D',   # living_room  — red
    2: '#C67C7B',   # kitchen      — salmon
    3: '#FFD274',   # bedroom      — yellow
    4: '#BEBEBE',   # bathroom     — gray
    5: '#BFE3E8',   # balcony      — light blue
    6: '#7BA779',   # entrance     — green
    7: '#E87A90',   # dining room  — pink
    8: '#FF8C69',   # study room   — orange
    10: '#1F849B',  # storage      — teal
    15: '#727171',  # front door   — dark gray
    16: '#785A67',  # unknown      — mauve
    17: '#D3A2C7',  # int. door    — light purple
}

ROOM_NAMES = {
    1: 'Living Room', 2: 'Kitchen', 3: 'Bedroom', 4: 'Bathroom',
    5: 'Balcony', 6: 'Entrance', 7: 'Dining Room', 8: 'Study',
    10: 'Storage', 15: 'Front Door', 17: 'Door',
}


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex colour string to (R, G, B) tuple."""
    if webcolors is not None:
        return webcolors.hex_to_rgb(hex_color)
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# ============================================================================
# Core functions (matching original House-GAN++)
# ============================================================================

def one_hot_embedding(labels: np.ndarray, num_classes: int = 19) -> np.ndarray:
    """One-hot encode room type labels. Returns [N, num_classes-1]."""
    y = np.eye(num_classes)
    result = y[labels]
    return result[:, 1:]  # drop column 0


def fix_nodes(prev_mks: torch.Tensor, ind_fixed_nodes: List[int]) -> torch.Tensor:
    """
    Prepare mask input with fixed / non-fixed node indicators.
    Matches original House-GAN++ implementation exactly.
    """
    given_masks = prev_mks.clone()

    all_nodes = set(range(len(given_masks)))
    ind_not_fixed = list(all_nodes - set(ind_fixed_nodes))

    if ind_not_fixed:
        given_masks[ind_not_fixed] = -1.0

    given_masks = given_masks.unsqueeze(1)

    inds_masks = torch.zeros_like(given_masks)
    if ind_fixed_nodes is not None and len(ind_fixed_nodes) > 0:
        inds_masks[list(ind_fixed_nodes)] = 1.0

    given_masks = torch.cat([given_masks, inds_masks], 1)
    return given_masks


# ============================================================================
# Rendering
# ============================================================================

def draw_masks(masks: np.ndarray, real_nodes: np.ndarray,
               im_size: int = 256) -> Image.Image:
    """
    Render room masks to a PIL RGBA image.

    Args:
        masks:       [N, 64, 64]  values in [-1, 1]
        real_nodes:  0-based room type indices (room_type - 1)
        im_size:     output image dimension
    """
    bg_img = Image.new("RGBA", (im_size, im_size), (255, 255, 255, 255))

    for m, nd in zip(masks, real_nodes):
        m = m.copy()
        m[m > 0] = 255
        m[m < 0] = 0

        m_lg = cv2.resize(m.astype(np.float32), (im_size, im_size),
                          interpolation=cv2.INTER_AREA)

        color = ID_COLOR.get(nd + 1, '#808080')
        r, g, b = _hex_to_rgb(color)

        dr = ImageDraw.Draw(bg_img)
        m_pil = Image.fromarray(m_lg.astype(np.uint8))
        dr.bitmap((0, 0), m_pil.convert('L'), fill=(r, g, b, 255))

        # Wall contours
        m_cv = m_lg[:, :, np.newaxis].astype(np.uint8)
        _, thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = np.zeros((im_size, im_size, 3), dtype=np.uint8)
            cv2.drawContours(cnt, contours, -1, (255, 255, 255), 1)
            cnt_pil = Image.fromarray(cnt)
            dr.bitmap((0, 0), cnt_pil.convert('L'), fill=(0, 0, 0, 255))

    return bg_img


def draw_masks_with_legend(masks: np.ndarray, real_nodes: np.ndarray,
                           room_names: List[str],
                           im_size: int = 512) -> Image.Image:
    """Render floor plan with a colour-coded legend panel."""
    DOOR_0BASED = {14, 16}  # 0-based indices for front_door / int_door

    fp_img = draw_masks(masks, real_nodes, im_size)

    legend_width = 220
    total_width = im_size + legend_width
    final_img = Image.new("RGBA", (total_width, im_size), (255, 255, 255, 255))
    final_img.paste(fp_img, (0, 0))

    dr = ImageDraw.Draw(final_img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        item_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        title_font = ImageFont.load_default()
        item_font = ImageFont.load_default()

    x0 = im_size + 15
    y = 20
    dr.text((x0, y), "Room Legend", fill=(0, 0, 0), font=title_font)
    y += 30
    dr.line([(x0, y - 5), (x0 + legend_width - 30, y - 5)],
            fill=(100, 100, 100), width=2)

    shown = set()
    for nd, name in zip(real_nodes, room_names):
        if nd in DOOR_0BASED:
            continue
        if name in shown:
            continue
        shown.add(name)

        color = ID_COLOR.get(nd + 1, '#808080')
        r, g, b = _hex_to_rgb(color)

        dr.rectangle([x0, y, x0 + 20, y + 20],
                     fill=(r, g, b), outline=(50, 50, 50))
        disp = name if len(name) <= 20 else name[:18] + '..'
        dr.text((x0 + 28, y + 3), disp, fill=(0, 0, 0), font=item_font)
        y += 28
        if y > im_size - 40:
            break

    return final_img


# ============================================================================
# Professional Architectural Rendering (with doors, labels, thick walls)
# ============================================================================

# Muted pastel palette — professional floor plan convention
ARCH_FILL = {
    1: (245, 218, 218),   # Living Room — soft blush
    2: (240, 228, 200),   # Kitchen — warm vanilla
    3: (248, 240, 205),   # Bedroom — cream
    4: (205, 225, 242),   # Bathroom — cool sky
    5: (208, 238, 225),   # Balcony — mint
    6: (228, 228, 228),   # Entrance — light gray
    7: (240, 218, 225),   # Dining Room — soft rose
    8: (225, 222, 240),   # Study — lavender
    10: (222, 222, 212),  # Storage — warm stone
}

_WALL = (35, 35, 35)
_WALL_OUTER = 4
_WALL_INNER = 2
_DOOR_W = 16
_DOOR_CLR = (50, 50, 50)
_LABEL_CLR = (25, 25, 25)


def _up(mask64: np.ndarray, sz: int) -> np.ndarray:
    """Upscale 64×64 mask to sz×sz, return binary uint8."""
    m = cv2.resize(mask64.astype(np.float32), (sz, sz), interpolation=cv2.INTER_AREA)
    return (m > 0).astype(np.uint8)


def _shared_boundary(b1: np.ndarray, b2: np.ndarray):
    """Find the shared wall pixels between two rectangular room masks.

    After rectangularization, rooms share clean axis-aligned edges.
    We look for the 1-pixel-wide strip where the two rooms actually
    touch (or nearly touch within 1 px gap due to overlap resolution).
    """
    k = np.ones((3, 3), np.uint8)
    # Dilate each by 1 px and intersect with the other
    d1 = cv2.dilate(b1, k, iterations=1)
    d2 = cv2.dilate(b2, k, iterations=1)
    bnd = (d1 & b2) | (d2 & b1)
    if bnd.sum() >= 2:
        return bnd
    # Rooms might have a 1-2px gap after overlap resolution → expand more
    d1 = cv2.dilate(b1, k, iterations=2)
    d2 = cv2.dilate(b2, k, iterations=2)
    bnd = d1 & d2
    return bnd


def _door_symbol(img, cx, cy, horiz, dw=_DOOR_W, clr=_DOOR_CLR):
    """Draw an architectural door symbol (opening + quarter-circle arc)."""
    hw = dw // 2
    if horiz:
        # Horizontal wall — clear gap, hinge on left, swing downward
        cv2.line(img, (cx - hw, cy), (cx + hw, cy), (255, 255, 255), _WALL_INNER + 3)
        cv2.line(img, (cx - hw, cy), (cx - hw, cy + dw), clr, 1, cv2.LINE_AA)
        cv2.ellipse(img, (cx - hw, cy), (dw, dw), 0, 0, 90, clr, 1, cv2.LINE_AA)
    else:
        # Vertical wall — clear gap, hinge on top, swing rightward
        cv2.line(img, (cx, cy - hw), (cx, cy + hw), (255, 255, 255), _WALL_INNER + 3)
        cv2.line(img, (cx, cy - hw), (cx + dw, cy - hw), clr, 1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy - hw), (dw, dw), 0, 0, 90, clr, 1, cv2.LINE_AA)


def draw_professional_floorplan(masks: np.ndarray,
                                nodes_arr: np.ndarray,
                                room_names: List[str],
                                edges: np.ndarray,
                                im_size: int = 512) -> Image.Image:
    """
    Render an **architectural-quality** floor plan with thick walls,
    door symbols, room labels, and a colour-coded legend.

    Args:
        masks:       [N, 64, 64]  cleaned masks (values ∈ {-1, 1})
        nodes_arr:   [N]          1-based room-type IDs
        room_names:  [N]          display names
        edges:       [E, 3]       edge triples
        im_size:     Output pixel size (default 512)

    Returns:
        PIL.Image (RGB) of floor plan + legend panel.
    """
    N = len(masks)
    edges = np.asarray(edges)

    # ---- Upscale all masks ----
    bins = [_up(masks[i], im_size) for i in range(N)]

    # ---- Snap room masks to axis-aligned rectangles (rendering only) ----
    for i in range(N):
        rt = int(nodes_arr[i])
        if rt in DOOR_TYPES:
            continue
        ys, xs = np.where(bins[i] > 0)
        if len(ys) < 4:
            continue
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        rect_bin = np.zeros_like(bins[i])
        rect_bin[r0:r1 + 1, c0:c1 + 1] = 1
        bins[i] = rect_bin

    # ---- Resolve rendering overlaps (so rectangles tile cleanly) ----
    room_areas = np.array([int(bins[i].sum()) for i in range(N)], dtype=float)
    overlap_map = sum(bins[i] for i in range(N) if int(nodes_arr[i]) not in DOOR_TYPES)
    if np.any(overlap_map > 1):
        contested = np.where(overlap_map > 1)
        for py, px in zip(contested[0], contested[1]):
            cands = [i for i in range(N) if int(nodes_arr[i]) not in DOOR_TYPES and bins[i][py, px] > 0]
            if len(cands) <= 1:
                continue
            winner = max(cands, key=lambda i: room_areas[i])
            for c in cands:
                if c != winner:
                    bins[c][py, px] = 0

    # ---- 1. Room fills (muted pastels) ----
    canvas = np.ones((im_size, im_size, 3), dtype=np.uint8) * 255
    for i in range(N):
        rt = int(nodes_arr[i])
        if rt in DOOR_TYPES:
            continue
        color = ARCH_FILL.get(rt, (230, 230, 230))
        mask_3d = bins[i][:, :, None]
        room_color = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
        canvas = np.where(mask_3d > 0, room_color, canvas)

    # ---- 2. Walls ----
    # Outer wall (entire floor plan boundary)
    all_rooms = np.zeros((im_size, im_size), dtype=np.uint8)
    for i in range(N):
        if int(nodes_arr[i]) not in DOOR_TYPES:
            all_rooms |= bins[i]
    smooth_all = cv2.GaussianBlur(all_rooms.astype(np.float32), (3, 3), 0)
    smooth_all = (smooth_all > 0.3).astype(np.uint8) * 255
    outer_c, _ = cv2.findContours(smooth_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, outer_c, -1, _WALL, _WALL_OUTER)

    # Inner walls (per-room contours)
    for i in range(N):
        if int(nodes_arr[i]) in DOOR_TYPES:
            continue
        smooth = cv2.GaussianBlur(bins[i].astype(np.float32), (3, 3), 0)
        smooth = (smooth > 0.3).astype(np.uint8) * 255
        cnt, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnt, -1, _WALL, _WALL_INNER)

    # ---- 3. Door symbols ----
    door_idxs = [i for i in range(N) if int(nodes_arr[i]) in DOOR_TYPES]
    for di in door_idxs:
        # Find connected rooms via adjacency edges
        connected = []
        for e in edges:
            s, r, d = int(e[0]), int(e[1]), int(e[2])
            if r != 1:
                continue
            if s == di and int(nodes_arr[d]) not in DOOR_TYPES:
                connected.append(d)
            elif d == di and int(nodes_arr[s]) not in DOOR_TYPES:
                connected.append(s)

        is_front = int(nodes_arr[di]) == 15

        if is_front and connected:
            # Front door — place on a proper exterior WALL of the connected room.
            # Strategy:
            # 1. Compute exterior boundary (room boundary minus adjacent rooms).
            # 2. Classify each ext-bnd pixel by which room bbox edge it belongs to.
            # 3. For horizontal edges (top/bottom) find the longest contiguous
            #    horizontal run; for vertical edges (left/right) the longest
            #    contiguous vertical run.
            # 4. Pick the best edge (bottom > left > right > top) and centre
            #    the door symbol on that run.
            ri = connected[0]
            k3 = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(bins[ri], k3, iterations=1)
            room_bnd = dilated - bins[ri]
            others = np.zeros_like(all_rooms)
            for j in range(N):
                if j != ri and int(nodes_arr[j]) not in DOOR_TYPES:
                    others |= bins[j]
            ext_bnd = room_bnd & (1 - cv2.dilate(others, k3, iterations=1))
            if ext_bnd.any():
                ys_all, xs_all = np.where(bins[ri] > 0)
                r_ymin, r_ymax = int(ys_all.min()), int(ys_all.max())
                r_xmin, r_xmax = int(xs_all.min()), int(xs_all.max())
                tol = 3  # pixels tolerance for "on edge"

                # Classify exterior boundary pixels by edge
                eys, exs = np.where(ext_bnd > 0)
                edge_groups = {}  # edge_name -> (xs_arr, ys_arr, is_horiz)
                for ename, edge_val, axis_arr, is_horiz in [
                    ('bottom', r_ymax, eys, True),
                    ('top',    r_ymin, eys, True),
                    ('left',   r_xmin, exs, False),
                    ('right',  r_xmax, exs, False),
                ]:
                    sel = np.abs(axis_arr - edge_val) <= tol
                    if sel.any():
                        edge_groups[ename] = (exs[sel], eys[sel], is_horiz)

                # Pick best edge: prefer bottom > left > right > top
                preference = ['bottom', 'left', 'right', 'top']
                chosen = None
                for pref in preference:
                    if pref in edge_groups:
                        chosen = (pref, *edge_groups[pref])
                        break
                if chosen is None and edge_groups:
                    k0 = next(iter(edge_groups))
                    chosen = (k0, *edge_groups[k0])

                if chosen:
                    ename, exsel, eysel, is_horiz = chosen
                    # Find the longest contiguous run along the wall so the
                    # door is centred on actual wall, not in a gap.
                    if is_horiz:
                        coords = np.sort(np.unique(exsel))
                    else:
                        coords = np.sort(np.unique(eysel))
                    # Split into contiguous runs (gap > 2px)
                    best_run = coords
                    if len(coords) > 1:
                        gaps = np.where(np.diff(coords) > 2)[0]
                        runs = np.split(coords, gaps + 1)
                        best_run = max(runs, key=len)
                    mid = int(best_run[len(best_run) // 2])
                    if is_horiz:
                        cx = mid
                        cy = int(np.median(eysel))
                    else:
                        cx = int(np.median(exsel))
                        cy = mid
                    _door_symbol(canvas, cx, cy, is_horiz,
                                 dw=_DOOR_W + 4, clr=(30, 30, 30))
        elif len(connected) >= 2:
            # Interior door between two rooms
            ra, rb = connected[0], connected[1]
            bnd = _shared_boundary(bins[ra], bins[rb])
            if bnd.any():
                ys, xs = np.where(bnd > 0)
                cx, cy = int(np.median(xs)), int(np.median(ys))
                sx = int(xs.max() - xs.min()) if len(xs) > 1 else 0
                sy = int(ys.max() - ys.min()) if len(ys) > 1 else 0
                _door_symbol(canvas, cx, cy, sx >= sy)

    # ---- 4. Room labels (name centred inside each room) ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(N):
        rt = int(nodes_arr[i])
        if rt in DOOR_TYPES:
            continue
        area = int(bins[i].sum())
        if area < 60:
            continue
        ys, xs = np.where(bins[i] > 0)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        label = room_names[i] if i < len(room_names) else f'Room {i}'
        fs = 0.40 if area > 2000 else (0.33 if area > 800 else 0.26)
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        tx, ty = cx - tw // 2, cy + th // 2
        cv2.rectangle(canvas, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 5),
                      (255, 255, 255), -1)
        cv2.putText(canvas, label, (tx, ty), font, fs, _LABEL_CLR, 1, cv2.LINE_AA)

    # ---- 5. Legend panel ----
    legend_w = 220
    total_w = im_size + legend_w
    final = np.ones((im_size, total_w, 3), dtype=np.uint8) * 255
    final[:, :im_size, :] = canvas
    cv2.line(final, (im_size + 2, 10), (im_size + 2, im_size - 10), (180, 180, 180), 1)

    x0 = im_size + 15
    y = 25
    cv2.putText(final, "Room Legend", (x0, y), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 12
    cv2.line(final, (x0, y), (x0 + legend_w - 30, y), (150, 150, 150), 1)
    y += 15

    shown = set()
    for i in range(N):
        rt = int(nodes_arr[i])
        if rt in DOOR_TYPES:
            continue
        nm = room_names[i] if i < len(room_names) else f'Room {i}'
        if nm in shown:
            continue
        shown.add(nm)
        color = ARCH_FILL.get(rt, (230, 230, 230))
        cv2.rectangle(final, (x0, y), (x0 + 18, y + 18), color, -1)
        cv2.rectangle(final, (x0, y), (x0 + 18, y + 18), (80, 80, 80), 1)
        disp = nm[:20] if len(nm) > 20 else nm
        cv2.putText(final, disp, (x0 + 26, y + 14), font, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
        y += 26
        if y > im_size - 80:
            break

    # Door symbol legend
    y += 15
    cv2.putText(final, "Door Symbols", (x0, y), font, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    y += 22
    _door_symbol(final, x0 + 12, y, False, dw=12, clr=_DOOR_CLR)
    cv2.putText(final, "Interior Door", (x0 + 32, y + 5), font, 0.35, (60, 60, 60), 1, cv2.LINE_AA)
    y += 28
    _door_symbol(final, x0 + 12, y, False, dw=14, clr=(30, 30, 30))
    cv2.putText(final, "Front Door", (x0 + 32, y + 5), font, 0.35, (60, 60, 60), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB) if False else final)


# ============================================================================
# Inference Engine
# ============================================================================

class HouseGANPPInferenceV2:
    """
    Research-grade House-GAN++ inference with:
    - Proper iterative refinement (matching original paper)
    - Mask post-processing (connected components, overlap removal)
    - Best-of-N generation with quality scoring
    """

    def __init__(self, checkpoint_path: str = None, device: str = 'auto'):
        self.device = (device if device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = load_generator(checkpoint_path, self.device)
        self.converter = RequirementsConverter()
        self.mask_size = 64

    # ------------------------------------------------------------------
    # Low-level inference
    # ------------------------------------------------------------------

    def _init_input(self, graph: Tuple, prev_state: Dict = None):
        given_nds, given_eds = graph
        given_nds = (torch.FloatTensor(given_nds)
                     if not isinstance(given_nds, torch.Tensor)
                     else given_nds.float())
        given_eds = (torch.LongTensor(given_eds)
                     if not isinstance(given_eds, torch.Tensor)
                     else given_eds.long())

        n_nodes = len(given_nds)
        z = torch.randn(n_nodes, 128).float()

        fixed_nodes = prev_state.get('fixed_nodes', []) if prev_state else []
        prev_mks = prev_state.get('masks', None) if prev_state else None

        if prev_mks is None:
            prev_mks = torch.zeros((n_nodes, self.mask_size, self.mask_size)) - 1.0
        else:
            prev_mks = torch.tensor(prev_mks).float()

        given_masks_in = fix_nodes(prev_mks, fixed_nodes)
        return z, given_masks_in, given_nds, given_eds

    def _infer(self, graph: Tuple, prev_state: Dict = None) -> np.ndarray:
        z, given_masks_in, given_nds, given_eds = self._init_input(graph, prev_state)
        with torch.no_grad():
            masks = self.model(
                z.to(self.device),
                given_masks_in.to(self.device),
                given_nds.to(self.device),
                given_eds.to(self.device),
            )
            masks = masks.detach().cpu().numpy()
        return masks

    # ------------------------------------------------------------------
    # High-level generation
    # ------------------------------------------------------------------

    def generate_from_graph(self, nodes, edges,
                            room_names: List[str] = None,
                            num_iterations: int = 5,
                            seed: int = None,
                            apply_post_processing: bool = True) -> Dict:
        """
        Generate a floor plan from a graph specification.

        Follows the exact iterative-refinement process of the original
        House-GAN++ test.py:

            1. Initial generation (no fixed nodes)
            2. Iterative refinement — fix rooms type-by-type
            3. Final refinement passes — fix all real rooms

        Args:
            nodes:          Room type IDs (1-based)
            edges:          Edge triples [E, 3]
            room_names:     Display names (auto-generated if None)
            num_iterations: Additional refinement passes
            seed:           Random seed for reproducibility
            apply_post_processing: Run mask cleanup (connected components,
                                   overlap removal)

        Returns:
            Dict with image, masks, nodes, edges, room_names, real_nodes, score.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        nodes_arr = np.array(nodes)
        nodes_onehot = one_hot_embedding(nodes_arr)
        real_nodes = nodes_arr - 1  # 0-based for colouring

        graph = (nodes_onehot, edges)

        # Unique real room types (excluding doors)
        room_types_only = [int(t) for t in nodes_arr if int(t) not in DOOR_TYPES]
        unique_types = sorted(set(room_types_only))

        # Build progressive-fix schedule
        selected_types = [unique_types[:k + 1]
                          for k in range(min(10, len(unique_types)))]

        # Step 1: initial generation (blank canvas)
        state = {'masks': None, 'fixed_nodes': []}
        masks = self._infer(graph, state)

        # Step 2: iterative refinement by room type
        for _types in selected_types:
            _fixed = []
            for _t in _types:
                _fixed.extend(np.where(nodes_arr == _t)[0].tolist())
            state = {'masks': masks, 'fixed_nodes': _fixed}
            masks = self._infer(graph, state)

        # Step 3: final refinement passes (all real rooms fixed)
        all_real = [i for i, t in enumerate(nodes_arr)
                    if int(t) not in DOOR_TYPES]
        for _ in range(num_iterations):
            state = {'masks': masks, 'fixed_nodes': all_real}
            masks = self._infer(graph, state)

        # Post-process
        if apply_post_processing:
            masks = post_process_masks(masks, nodes_arr)

        # Auto-generate room names
        if room_names is None:
            room_names = self._auto_room_names(nodes_arr)

        # Render (professional architectural style with doors)
        edges_arr = np.array(edges) if not isinstance(edges, np.ndarray) else edges
        image = draw_professional_floorplan(masks, nodes_arr, room_names, edges_arr)

        # Quality score
        q_score = score_floor_plan(masks, nodes_arr, edges_arr)

        return {
            'image': image,
            'masks': masks,
            'nodes': nodes_arr,
            'edges': np.array(edges),
            'room_names': room_names,
            'real_nodes': real_nodes,
            'score': q_score,
        }

    def generate_best_of_n(self, nodes, edges,
                           room_names: List[str] = None,
                           num_iterations: int = 5,
                           n_candidates: int = 8,
                           base_seed: int = None,
                           apply_post_processing: bool = True) -> Dict:
        """
        Generate N floor plans and return the highest-scoring one.

        This is the recommended entry-point for research-quality output.

        Args:
            n_candidates: Number of candidates to generate and score.
            base_seed:    If set, seeds are base_seed, base_seed+1, ...
                          If None, random seeds are used.

        Returns:
            Best result dict (same schema as generate_from_graph),
            plus ``all_scores`` and ``all_seeds``.
        """
        import random

        best_result = None
        best_score = -1
        all_scores = []
        all_seeds = []

        for k in range(n_candidates):
            seed_k = (base_seed + k) if base_seed is not None else random.randint(1, 99999)
            all_seeds.append(seed_k)

            result = self.generate_from_graph(
                nodes, edges,
                room_names=room_names,
                num_iterations=num_iterations,
                seed=seed_k,
                apply_post_processing=apply_post_processing,
            )
            all_scores.append(result['score'])

            if result['score'] > best_score:
                best_score = result['score']
                best_result = result

        best_result['all_scores'] = all_scores
        best_result['all_seeds'] = all_seeds
        return best_result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def generate_from_requirements(self, requirements: Dict,
                                   num_iterations: int = 5,
                                   seed: int = None,
                                   use_template: bool = True,
                                   n_candidates: int = 8) -> Dict:
        """
        Generate floor plan from a requirements dict.

        Workflow:
            1. If use_template: find the best matching template
            2. Otherwise: build graph from requirements
            3. Generate best-of-N candidates

        Args:
            requirements:  num_bedrooms, num_bathrooms, has_balcony, …
            use_template:  Try template matching first
            n_candidates:  Number of candidates for best-of-N
        """
        if use_template:
            try:
                selector = TemplateSelector()
                best_match = selector.find_best_match(requirements)
                if best_match is not None:
                    # Reject template if it has balcony but requirements say no
                    if not requirements.get('has_balcony', False) and best_match.has_balcony:
                        print(f"  Template {Path(best_match.path).name} has balcony — skipping, using direct graph.")
                    # Reject template if score is too high (poor match → direct graph is better)
                    elif best_match.score > 5:
                        print(f"  Template {Path(best_match.path).name} score={best_match.score:.0f} (poor match) — using direct graph.")
                    else:
                        print(f"  Using template: {Path(best_match.path).name} "
                              f"({best_match.num_bedrooms}BR/"
                              f"{best_match.num_bathrooms}BA, "
                              f"score={best_match.score:.0f})")
                        return self.generate_from_json_file(
                            best_match.path,
                            num_iterations=num_iterations,
                            seed=seed,
                            n_candidates=n_candidates,
                        )
            except Exception as e:
                print(f"  Template selection failed ({e}), using direct graph.")

        # Direct graph construction
        nodes, edges, room_names = self.converter.convert(requirements)
        return self.generate_best_of_n(
            nodes=nodes.tolist(), edges=edges,
            room_names=room_names,
            num_iterations=num_iterations,
            n_candidates=n_candidates,
            base_seed=seed,
        )

    def generate_from_json_file(self, json_path: str,
                                num_iterations: int = 5,
                                seed: int = None,
                                n_candidates: int = 8) -> Dict:
        """
        Generate floor plan from a House-GAN++ JSON template.
        """
        with open(json_path) as f:
            data = json.load(f)

        nodes = np.array(data['room_type'])
        n_rooms = len(nodes)

        # Build edges from ed_rm if available
        if 'ed_rm' in data:
            edges = create_edges_from_ed_rm(data['ed_rm'], n_rooms)
        else:
            edges = []
            for i in range(n_rooms):
                for j in range(i + 1, n_rooms):
                    edges.append([i, 1, j])
            edges = np.array(edges, dtype=np.int64)

        return self.generate_best_of_n(
            nodes=nodes.tolist(), edges=edges,
            num_iterations=num_iterations,
            n_candidates=n_candidates,
            base_seed=seed,
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_result(self, result: Dict, output_dir: str,
                    filename: str = 'floorplan') -> Dict[str, str]:
        """Save generation result (image, masks, metadata)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths = {}

        img_path = output_path / f'{filename}.png'
        result['image'].save(str(img_path))
        paths['image'] = str(img_path)

        masks_path = output_path / f'{filename}_masks.npy'
        np.save(str(masks_path), result['masks'])
        paths['masks'] = str(masks_path)

        meta = {
            'room_names': result['room_names'],
            'nodes': result['nodes'].tolist(),
            'edges': result['edges'].tolist() if len(result['edges']) > 0 else [],
            'score': float(result.get('score', 0)),
            'all_scores': result.get('all_scores', []),
            'all_seeds': result.get('all_seeds', []),
        }
        meta_path = output_path / f'{filename}_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        paths['metadata'] = str(meta_path)

        return paths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_room_names(nodes: np.ndarray) -> List[str]:
        """Generate display names with numbering."""
        names = []
        counters: Dict[int, int] = {}
        for nd in nodes:
            nd = int(nd)
            counters[nd] = counters.get(nd, 0) + 1
            c = counters[nd]
            if nd == 3:
                names.append('Master Bedroom' if c == 1 else f'Bedroom {c}')
            elif nd == 4:
                names.append('Attached Bath' if c == 1 else f'Bathroom {c}')
            elif nd in DOOR_TYPES:
                names.append('Door' if nd == 17 else 'Front Door')
            else:
                names.append(ROOM_NAMES.get(nd, f'Room {nd}'))
        return names


# ============================================================================
# Convenience standalone function
# ============================================================================

def generate_floorplan(requirements: Dict = None,
                       json_path: str = None,
                       output_dir: str = 'generated_floorplans',
                       seed: int = None,
                       n_candidates: int = 8) -> Tuple[Image.Image, str]:
    """
    One-call floor plan generation.

    Returns:
        (PIL.Image, path_to_saved_png)
    """
    inference = HouseGANPPInferenceV2()

    if json_path:
        result = inference.generate_from_json_file(json_path, seed=seed,
                                                    n_candidates=n_candidates)
    elif requirements:
        result = inference.generate_from_requirements(requirements, seed=seed,
                                                       n_candidates=n_candidates)
    else:
        raise ValueError("Provide either requirements or json_path")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = inference.save_result(result, output_dir, f'floorplan_{timestamp}')
    return result['image'], paths['image']


if __name__ == "__main__":
    print("Testing House-GAN++ Inference V2 (Research Grade)...")

    inference = HouseGANPPInferenceV2()

    # Template test
    result = inference.generate_from_json_file(
        'houseganpp/data/json/18477.json', seed=42, n_candidates=4,
    )
    result['image'].save('test_v2_template.png')
    print(f"Template test: score={result['score']:.1f}/100")

    # Requirements test
    req = {
        "num_bedrooms": 2, "num_bathrooms": 2,
        "has_balcony": False, "combined_living_dining": True,
    }
    result = inference.generate_from_requirements(req, seed=42, n_candidates=4)
    result['image'].save('test_v2_requirements.png')
    print(f"Requirements test: score={result['score']:.1f}/100")
