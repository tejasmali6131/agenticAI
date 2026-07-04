"""
Architectural Constraints for Floor Plan Generation
====================================================

Research-grade architectural rules and constraints for House-GAN++ 
floor plan generation. Enforces proper room placement, adjacency rules,
area proportions, and connectivity patterns based on real-world 
architectural standards and RPLAN training data analysis.

Graph Topology (derived from RPLAN training data analysis):
    - Hub-and-spoke: Living room is the central hub
    - Door:Room ratio = 1:1 (N-1 interior doors + 1 front door for N real rooms)
    - Each non-hub room connects to hub (living) via exactly 1 interior door
    - Exception: attached bathrooms connect to parent bedroom

References:
    - House-GAN++: Generative Adversarial Layout Refinement Network (CVPR 2021)
    - RPLAN: A Large-Scale Residential Floor Plan Dataset
    - National Building Code of India (NBC 2016)
    - Neufert Architects' Data (4th Ed.)
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field


# ============================================================================
# Room Type Constants (House-GAN++ / RPLAN)
# ============================================================================

ROOM_TYPES = {
    'living_room': 1,
    'kitchen': 2,
    'bedroom': 3,
    'bathroom': 4,
    'balcony': 5,
    'entrance': 6,
    'dining_room': 7,
    'study': 8,
    'storage': 10,
    'front_door': 15,
    'interior_door': 17,
}

ROOM_TYPE_NAMES = {v: k for k, v in ROOM_TYPES.items()}

ROOM_DISPLAY_NAMES = {
    1: 'Living Room', 2: 'Kitchen', 3: 'Bedroom', 4: 'Bathroom',
    5: 'Balcony', 6: 'Entrance', 7: 'Dining Room', 8: 'Study',
    10: 'Storage', 15: 'Front Door', 17: 'Interior Door'
}

# Room types that are actual habitable rooms (not doors/connectors)
REAL_ROOM_TYPES = {1, 2, 3, 4, 5, 6, 7, 8, 10}

# Door/connector types
DOOR_TYPES = {15, 17}


# ============================================================================
# User-Friendly Room Name Aliases
# ============================================================================

ROOM_ALIASES = {
    # Living areas
    'living': 1, 'living_room': 1, 'living room': 1,
    'hall': 1, 'drawing': 1, 'drawing room': 1, 'lobby': 1,
    # Kitchen
    'kitchen': 2, 'kitchenette': 2, 'pantry': 2,
    # Bedrooms
    'bedroom': 3, 'master_bedroom': 3, 'master bedroom': 3,
    'guest_room': 3, 'guest room': 3,
    'kids_room': 3, 'kids room': 3, "children's room": 3,
    # Bathrooms
    'bathroom': 4, 'toilet': 4, 'washroom': 4, 'restroom': 4,
    'wc': 4, 'attached_bath': 4, 'attached bath': 4,
    'common_bath': 4, 'common bath': 4,
    # Balcony
    'balcony': 5, 'terrace': 5, 'patio': 5, 'verandah': 5, 'deck': 5,
    # Entrance
    'entrance': 6, 'foyer': 6, 'entry': 6, 'vestibule': 6,
    # Dining
    'dining': 7, 'dining room': 7, 'dining_room': 7, 'dining area': 7,
    # Study
    'study': 8, 'study room': 8, 'study_room': 8,
    'office': 8, 'home office': 8, 'workspace': 8, 'library': 8,
    # Storage
    'storage': 10, 'store': 10, 'store room': 10, 'store_room': 10,
    'storeroom': 10, 'closet': 10, 'wardrobe': 10,
    'utility': 10, 'utility room': 10,
    # Special (Indian residential)
    'pooja': 10, 'pooja room': 10, 'prayer': 10,
    'prayer room': 10, 'puja': 10, 'puja room': 10,
    # Misc
    'unknown': 16, 'other': 16,
}


def map_room_type(room_name: str) -> int:
    """
    Map a room name to House-GAN++ room type ID.
    
    Args:
        room_name: Human-readable room name
        
    Returns:
        Room type ID (1-17)
    """
    name_lower = room_name.lower().strip()

    # Exact match
    if name_lower in ROOM_ALIASES:
        return ROOM_ALIASES[name_lower]

    # Partial / substring match
    for alias, room_id in ROOM_ALIASES.items():
        if alias in name_lower or name_lower in alias:
            return room_id

    return 16  # unknown


# ============================================================================
# Area Proportion Constraints
# ============================================================================
# (min_fraction, max_fraction) of total habitable floor area.
# Based on NBC India 2016 and standard residential design guidelines.

AREA_PROPORTIONS = {
    1: (0.15, 0.28),   # Living Room: largest room, central hub
    2: (0.07, 0.14),   # Kitchen
    3: (0.10, 0.20),   # Bedroom (master can be larger)
    4: (0.03, 0.08),   # Bathroom
    5: (0.02, 0.06),   # Balcony
    6: (0.02, 0.05),   # Entrance / Foyer
    7: (0.06, 0.12),   # Dining Room
    8: (0.05, 0.10),   # Study
    10: (0.02, 0.05),  # Storage
}

# Absolute minimum area in pixels on the 64×64 mask grid.
# Rooms below this threshold are considered degenerate.
MIN_ROOM_AREA_PIXELS = {
    1: 200, 2: 100, 3: 150, 4: 50, 5: 40,
    6: 30,  7: 80,  8: 80,  10: 30,
}

# Maximum bounding-box aspect ratio before a room is flagged as
# unnaturally shaped (long corridor-like slivers).
MAX_ASPECT_RATIO = 3.5


# ============================================================================
# Adjacency Constraint Sets
# ============================================================================
# Based on:
#   - National Building Code of India (NBC 2016)
#   - Neufert Architects' Data (4th Ed.)
#   - RPLAN training data analysis
#   - Standard Indian residential design practice

# Pairs that MUST be adjacent (relation = 1) when both exist.
MANDATORY_ADJACENCIES: Set[Tuple[int, int]] = {
    (1, 2),   # Living Room ↔ Kitchen
    (2, 7),   # Kitchen ↔ Dining Room (when separate)
    (1, 7),   # Living Room ↔ Dining Room
    (1, 6),   # Living Room ↔ Entrance / Foyer (front door opens to living)
}

# Pairs that SHOULD be adjacent when both exist (soft constraint).
# NOTE: Balcony ↔ Bedroom is NOT here — balcony adjacency is handled
# explicitly in _build_edges (only to its door-target room + living).
PREFERRED_ADJACENCIES: Set[Tuple[int, int]] = {
    (1, 5),   # Living Room ↔ Balcony (balcony off living or master bed)
    (1, 8),   # Living Room ↔ Study
    (2, 10),  # Kitchen ↔ Storage (utility/service proximity)
    (1, 10),  # Living Room ↔ Storage/Pooja (pooja can attach to living)
    (7, 10),  # Dining ↔ Storage/Pooja (pooja can attach to dining)
}

# Pairs that must NEVER be adjacent (relation = -1).
FORBIDDEN_ADJACENCIES: Set[Tuple[int, int]] = {
    (2, 4),   # Kitchen ↔ Bathroom  (hygiene — NBC India 2016 cl. 9.3)
    (4, 5),   # Bathroom ↔ Balcony  (water exposure / convention)
    (3, 2),   # Bedroom ↔ Kitchen   (noise & odour separation)
    (4, 7),   # Bathroom ↔ Dining   (hygiene / aesthetics)
    (4, 2),   # Bathroom ↔ Kitchen  (duplicate direction for safety)
    (4, 6),   # Bathroom ↔ Entrance (toilet should not face entry)
    (2, 6),   # Kitchen ↔ Entrance  (kitchen should not face main entrance)
    (3, 6),   # Bedroom ↔ Entrance  (bedrooms should not open to main entrance)
}

# Room types that should have only ONE door (single entry point).
SINGLE_DOOR_ROOMS = {3, 4}  # Bedrooms and Bathrooms

# Room types that MUST be on an exterior wall.
MUST_BE_EXTERIOR = {5}  # Balcony


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RoomNode:
    """A single room in the floor plan graph."""
    index: int              # Position in the node list
    room_type: int          # House-GAN++ room type ID
    display_name: str       # Human-readable label
    role: str               # Semantic role: 'hub', 'master_bedroom', 'attached_bath', 'common_bath', etc.
    door_target_idx: int    # Index of the room this room's door connects to (-1 = none / hub)


@dataclass
class DoorConnection:
    """An interior-door link between two rooms."""
    door_index: int         # Index of the door node
    room_a_index: int       # First room connected
    room_b_index: int       # Second room connected


@dataclass
class ValidationResult:
    """Result of architectural validation."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Input Validation
# ============================================================================

def validate_requirements(requirements: Dict) -> Tuple[Dict, List[str]]:
    """
    Validate and normalise user requirements to safe ranges.
    
    Enforces:
        - 1 ≤ bedrooms ≤ 4
        - 1 ≤ bathrooms ≤ min(bedrooms + 1, 3)
        - Total real rooms ≤ 10 (model sweet spot: 5-8)
    
    Args:
        requirements: Raw requirements dict from chatbot / user.
        
    Returns:
        (normalised_requirements, list_of_warnings)
    """
    warnings: List[str] = []
    req = dict(requirements)

    # --- Bedrooms ---
    bedrooms = req.get('num_bedrooms', 2)
    if not isinstance(bedrooms, (int, float)) or bedrooms < 1:
        warnings.append(f"Invalid num_bedrooms={bedrooms}, defaulting to 2")
        bedrooms = 2
    bedrooms = int(bedrooms)
    if bedrooms > 4:
        warnings.append(f"num_bedrooms={bedrooms} exceeds max 4, capping")
        bedrooms = 4
    req['num_bedrooms'] = bedrooms

    # --- Bathrooms ---
    bathrooms = req.get('num_bathrooms', 1)
    if not isinstance(bathrooms, (int, float)) or bathrooms < 1:
        warnings.append(f"Invalid num_bathrooms={bathrooms}, defaulting to 1")
        bathrooms = 1
    bathrooms = int(bathrooms)
    max_bath = min(bedrooms + 1, 3)
    if bathrooms > max_bath:
        warnings.append(f"num_bathrooms={bathrooms} too high for {bedrooms} bedrooms, capping to {max_bath}")
        bathrooms = max_bath
    req['num_bathrooms'] = bathrooms

    # --- Booleans with safe defaults ---
    req['has_balcony'] = bool(req.get('has_balcony', False))
    req['has_study'] = bool(req.get('has_study', False))
    req['has_storage'] = bool(req.get('has_storage', False))
    req['combined_living_dining'] = bool(req.get('combined_living_dining', True))

    # --- Total room count warning ---
    total = 1 + 1 + bedrooms + bathrooms  # living + kitchen + beds + baths
    if not req['combined_living_dining']:
        total += 1  # dining
    if req['has_balcony']:
        total += 1
    if req['has_study']:
        total += 1
    if req['has_storage']:
        total += 1

    if total > 10:
        warnings.append(
            f"Total room count ({total}) is high. "
            f"House-GAN++ produces best results with 5-8 real rooms."
        )

    req['_total_real_rooms'] = total
    return req, warnings


# ============================================================================
# Graph Validation
# ============================================================================

def validate_graph(nodes: np.ndarray, edges: np.ndarray,
                   room_names: List[str]) -> ValidationResult:
    """
    Validate a floor plan graph against architectural constraints.
    
    Checks:
        1. Door-to-room ratio (should be 1:1)
        2. Living room exists and acts as hub
        3. Mandatory adjacencies satisfied
        4. Forbidden adjacencies not present
        5. Each interior door connects exactly 2 rooms
    """
    result = ValidationResult()
    n = len(nodes)

    real_rooms = [(i, int(nodes[i])) for i in range(n) if int(nodes[i]) in REAL_ROOM_TYPES]
    door_rooms = [(i, int(nodes[i])) for i in range(n) if int(nodes[i]) in DOOR_TYPES]

    # Build adjacency from edge triples
    adj: Dict[int, Set[int]] = {}
    for edge in edges:
        src, rel, dst = int(edge[0]), int(edge[1]), int(edge[2])
        if rel == 1:
            adj.setdefault(src, set()).add(dst)
            adj.setdefault(dst, set()).add(src)

    n_interior = sum(1 for _, t in door_rooms if t == 17)
    n_front = sum(1 for _, t in door_rooms if t == 15)
    n_real = len(real_rooms)

    # Check 1: door ratio
    expected_interior = n_real - 1  # every room except living hub
    if n_interior != expected_interior:
        result.warnings.append(
            f"Interior door count ({n_interior}) != expected ({expected_interior}). "
            f"Training data uses exactly N_real-1 interior doors."
        )

    if n_front != 1:
        result.errors.append(f"Expected 1 front door, found {n_front}")
        result.is_valid = False

    # Check 2: living room hub
    living_indices = [i for i, t in real_rooms if t == 1]
    if not living_indices:
        result.errors.append("No living room — required as central hub")
        result.is_valid = False
    else:
        li = living_indices[0]
        neighbours = adj.get(li, set())
        connected_reals = sum(1 for i in neighbours if int(nodes[i]) in REAL_ROOM_TYPES)
        result.scores['hub_connectivity'] = connected_reals / max(1, n_real - 1)

    # Check 3: mandatory adjacencies (only when BOTH room types exist)
    present_types = {t for _, t in real_rooms}
    for ta, tb in MANDATORY_ADJACENCIES:
        if ta not in present_types or tb not in present_types:
            continue  # skip if either room type is absent from the graph
        for ia, _ in [(i, t) for i, t in real_rooms if t == ta]:
            has_adj = False
            for ib, _ in [(i, t) for i, t in real_rooms if t == tb]:
                if ib in adj.get(ia, set()):
                    has_adj = True
                    break
            if not has_adj:
                result.warnings.append(
                    f"Mandatory adjacency missing: "
                    f"{ROOM_DISPLAY_NAMES.get(ta)} ↔ {ROOM_DISPLAY_NAMES.get(tb)}"
                )

    # Check 4: forbidden adjacencies
    for ta, tb in FORBIDDEN_ADJACENCIES:
        for ia, _ in [(i, t) for i, t in real_rooms if t == ta]:
            for ib, _ in [(i, t) for i, t in real_rooms if t == tb]:
                if ib in adj.get(ia, set()):
                    result.errors.append(
                        f"Forbidden adjacency: "
                        f"{ROOM_DISPLAY_NAMES.get(ta)} (idx {ia}) ↔ "
                        f"{ROOM_DISPLAY_NAMES.get(tb)} (idx {ib})"
                    )

    # Check 5: each interior door connects exactly 2 rooms
    for di, dt in door_rooms:
        if dt == 17:
            n_neighbours = len(adj.get(di, set()))
            if n_neighbours != 2:
                result.warnings.append(
                    f"Interior door idx {di} has {n_neighbours} neighbours (expected 2)"
                )

    return result


# ============================================================================
# Mask / Output Validation & Scoring
# ============================================================================

def validate_masks(masks: np.ndarray, nodes: np.ndarray) -> ValidationResult:
    """
    Validate generated masks against area and shape constraints.
    """
    result = ValidationResult()
    total_area = masks.shape[1] * masks.shape[2]

    for i in range(len(masks)):
        rt = int(nodes[i])
        if rt in DOOR_TYPES:
            continue

        mask = masks[i]
        area = int(np.sum(mask > 0))
        name = ROOM_DISPLAY_NAMES.get(rt, f'Room-{rt}')

        # Minimum area
        min_px = MIN_ROOM_AREA_PIXELS.get(rt, 30)
        if area < min_px:
            result.warnings.append(f"{name} (idx {i}): area={area}px < min {min_px}px")

        # Area proportion
        prop = area / total_area
        lo, hi = AREA_PROPORTIONS.get(rt, (0.02, 0.30))
        if prop < lo * 0.5:
            result.warnings.append(
                f"{name} (idx {i}): proportion {prop:.1%} far below expected {lo:.0%}–{hi:.0%}"
            )

        # Aspect ratio
        if area > 10:
            binary = (mask > 0).astype(np.uint8)
            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                ar = max(w, h) / max(min(w, h), 1)
                if ar > MAX_ASPECT_RATIO:
                    result.warnings.append(
                        f"{name} (idx {i}): aspect ratio {ar:.1f} > max {MAX_ASPECT_RATIO}"
                    )

    return result


def score_floor_plan(masks: np.ndarray, nodes: np.ndarray,
                     edges: np.ndarray) -> float:
    """
    Score a generated floor plan on a 0–100 scale.
    
    Components (weights):
        1. Area balance          — 15 pts  (room sizes proportional)
        2. Coverage / gaps       — 20 pts  (rooms fill space, no holes)
        3. Shape quality         — 15 pts  (rectangularity)
        4. Overlap penalty       — 15 pts  (rooms should not overlap)
        5. Adjacency fidelity    — 15 pts  (adjacent rooms touch in output)
        6. Proportion fidelity   — 10 pts  (bath/balcony smaller than bedroom)
        7. Placement rules       — 10 pts  (forbidden adjacency, exterior balcony, etc.)
    
    Args:
        masks:  [N, 64, 64]  generated room masks
        nodes:  [N]           room type IDs
        edges:  [E, 3]        edge triples
        
    Returns:
        Float score in [0, 100].
    """
    mask_size = masks.shape[1]
    total_px = mask_size * mask_size
    real_indices = [i for i in range(len(nodes)) if int(nodes[i]) in REAL_ROOM_TYPES]

    # ---------- 1. Area balance (15 pts) ----------
    areas = np.array([np.sum(masks[i] > 0) for i in real_indices], dtype=float)
    if len(areas) == 0 or areas.sum() == 0:
        return 0.0

    cv = np.std(areas) / max(np.mean(areas), 1)
    area_score = max(0, 15 * (1 - cv / 2))
    # Degenerate penalty: −5 per room with < 20px
    area_score -= 5 * int(np.sum(areas < 20))
    area_score = max(0, area_score)

    # ---------- 2. Coverage / gap penalty (20 pts) ----------
    combined = np.zeros((mask_size, mask_size), dtype=bool)
    for i in real_indices:
        combined |= (masks[i] > 0)
    coverage = np.sum(combined) / total_px
    coverage_score = min(15, 15 * coverage / 0.55)

    # Penalise interior gaps: holes inside the convex hull of the floor plate
    floor_uint = combined.astype(np.uint8)
    inv = (1 - floor_uint).astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(inv, connectivity=4)
    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)
    gap_px = 0
    for lbl in range(1, n_lab):
        if lbl not in border_labels:
            gap_px += int(np.sum(labels == lbl))
    gap_ratio = gap_px / max(float(np.sum(combined)), 1.0)
    gap_bonus = max(0, 5 * (1 - gap_ratio * 10))  # 5 pts if no gaps
    coverage_score += gap_bonus

    # ---------- 3. Shape quality (15 pts) ----------
    rect_scores = []
    for i in real_indices:
        m = masks[i]
        a = np.sum(m > 0)
        if a < 10:
            rect_scores.append(0.0)
            continue
        binary = (m > 0).astype(np.uint8)
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
            rect_scores.append(a / max(bbox_area, 1))
        else:
            rect_scores.append(0.0)
    shape_score = 15 * (np.mean(rect_scores) if rect_scores else 0)

    # ---------- 4. Overlap penalty (15 pts) ----------
    overlap_total = 0
    for idx_a in range(len(real_indices)):
        for idx_b in range(idx_a + 1, len(real_indices)):
            i, j = real_indices[idx_a], real_indices[idx_b]
            overlap_total += int(np.sum((masks[i] > 0) & (masks[j] > 0)))
    overlap_ratio = overlap_total / max(areas.sum(), 1)
    overlap_score = max(0, 15 * (1 - overlap_ratio * 5))

    # ---------- 5. Adjacency fidelity (15 pts) ----------
    kernel = np.ones((3, 3), np.uint8)
    adj_ok = 0
    adj_total = 0
    for edge in edges:
        src, rel, dst = int(edge[0]), int(edge[1]), int(edge[2])
        if rel != 1:
            continue
        if int(nodes[src]) not in REAL_ROOM_TYPES or int(nodes[dst]) not in REAL_ROOM_TYPES:
            continue
        adj_total += 1
        m_src = (masks[src] > 0).astype(np.uint8)
        m_dst = (masks[dst] > 0).astype(np.uint8)
        dilated = cv2.dilate(m_src, kernel, iterations=1)
        if np.sum(dilated & m_dst) > 0:
            adj_ok += 1
    adj_score = 15 * (adj_ok / max(adj_total, 1))

    # ---------- 6. Proportion fidelity (10 pts) ----------
    # Bathrooms & balconies should be meaningfully smaller than bedrooms
    bedroom_areas = [np.sum(masks[i] > 0) for i in range(len(nodes))
                      if int(nodes[i]) == 3 and np.sum(masks[i] > 0) > 10]
    min_bed = min(bedroom_areas) if bedroom_areas else total_px * 0.10
    prop_penalties = 0
    n_checked = 0
    for i in range(len(nodes)):
        rt = int(nodes[i])
        if rt not in (4, 5):
            continue
        a = np.sum(masks[i] > 0)
        if a < 5:
            continue
        n_checked += 1
        if a > min_bed * 0.7:
            prop_penalties += 1  # bathroom / balcony almost as big as bedroom
    prop_score = 10 * (1 - prop_penalties / max(n_checked, 1)) if n_checked > 0 else 10

    # ---------- 7. Placement rules (10 pts) ----------
    # Checks architectural rules in the *physical* output masks.
    placement_score = 10.0

    # 7a. Forbidden adjacency violations in physical masks (-2 pts each, max −8)
    #     Two rooms that should NEVER touch but do in the generated layout.
    forbid_violations = 0
    for idx_a in range(len(real_indices)):
        for idx_b in range(idx_a + 1, len(real_indices)):
            i, j = real_indices[idx_a], real_indices[idx_b]
            ti, tj = int(nodes[i]), int(nodes[j])
            if (ti, tj) in FORBIDDEN_ADJACENCIES or (tj, ti) in FORBIDDEN_ADJACENCIES:
                m_i = (masks[i] > 0).astype(np.uint8)
                m_j = (masks[j] > 0).astype(np.uint8)
                dilated_i = cv2.dilate(m_i, kernel, iterations=1)
                if np.sum(dilated_i & m_j) > 0:
                    forbid_violations += 1
    placement_score -= min(8.0, forbid_violations * 2.0)

    # 7b. Balcony on exterior wall (+2 bonus if all balconies touch border)
    balcony_bonus = 0.0
    n_balconies = 0
    edge_mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
    edge_mask[0:2, :] = 1; edge_mask[-2:, :] = 1
    edge_mask[:, 0:2] = 1; edge_mask[:, -2:] = 1
    floor_union = np.zeros((mask_size, mask_size), dtype=np.uint8)
    for i in real_indices:
        floor_union |= (masks[i] > 0).astype(np.uint8)
    dilated_fp = cv2.dilate(floor_union, kernel, iterations=2)
    exterior = dilated_fp & (~floor_union).astype(np.uint8)
    exterior |= (floor_union & edge_mask)
    for i in range(len(nodes)):
        if int(nodes[i]) != 5:
            continue
        bal = (masks[i] > 0).astype(np.uint8)
        if bal.sum() < 5:
            continue
        n_balconies += 1
        if np.sum(cv2.dilate(bal, kernel, iterations=1) & exterior) > 0:
            balcony_bonus += 1.0
    if n_balconies > 0:
        placement_score += 2.0 * (balcony_bonus / n_balconies)
    else:
        placement_score += 2.0  # no balcony → no penalty

    placement_score = max(0.0, min(12.0, placement_score))

    total = (area_score + coverage_score + shape_score + overlap_score
             + adj_score + prop_score + placement_score)
    return min(100.0, max(0.0, total))


# ============================================================================
# Layout Refinement Helpers
# ============================================================================

def _enforce_room_sizes(masks: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Enforce realistic room size proportions.

    Bathrooms and balconies should be noticeably smaller than bedrooms.
    Any room exceeding its ``AREA_PROPORTIONS`` band (with tolerance) is
    morphologically eroded until it fits.

    Operates **in-place** on *masks* and returns the same array.
    """
    N, H, W = masks.shape
    total_area = H * W
    kernel = np.ones((3, 3), np.uint8)

    # --- reference: smallest bedroom area --------------------------------
    bedroom_areas = []
    for i in range(N):
        if int(nodes[i]) == 3:
            a = int(np.sum(masks[i] > 0))
            if a > 10:
                bedroom_areas.append(a)
    min_bedroom_area = min(bedroom_areas) if bedroom_areas else int(total_area * 0.10)

    for i in range(N):
        rt = int(nodes[i])
        if rt in DOOR_TYPES:
            continue

        binary = (masks[i] > 0).astype(np.uint8)
        area = int(binary.sum())
        if area < 10:
            continue

        _, hi = AREA_PROPORTIONS.get(rt, (0.02, 0.30))
        max_area = int(hi * total_area * 1.3)          # 30 % tolerance

        # Bathrooms & balconies: never bigger than 55 % of smallest bedroom
        if rt in (4, 5):
            max_area = min(max_area, int(min_bedroom_area * 0.55))

        # Erode oversized room (max 8 iterations to avoid vanishing)
        iters = 0
        while area > max_area and iters < 8:
            candidate = cv2.erode(binary, kernel, iterations=1)
            if candidate.sum() < 10:
                break                             # don't erase completely
            binary = candidate
            area = int(binary.sum())
            iters += 1

        masks[i] = np.where(binary > 0, 1.0, -1.0)

    return masks


def _move_balcony_to_exterior(masks: np.ndarray,
                              nodes: np.ndarray) -> np.ndarray:
    """
    Relocate interior balconies to the exterior boundary.

    Algorithm:
        1. Build the union floor-plate and its exterior boundary.
        2. For every balcony that does **not** touch the exterior,
           find the closest bedroom / living room with exterior exposure.
        3. Create a rectangular balcony strip on that room's exterior
           wall and clear overlaps.

    Operates in-place and returns *masks*.
    """
    N, H, W = masks.shape
    kernel = np.ones((3, 3), np.uint8)

    # --- union floor plate -----------------------------------------------
    floor_union = np.zeros((H, W), dtype=np.uint8)
    for i in range(N):
        if int(nodes[i]) not in DOOR_TYPES:
            floor_union |= (masks[i] > 0).astype(np.uint8)

    # Exterior = just outside the floor plate + grid edges
    dilated_fp = cv2.dilate(floor_union, kernel, iterations=2)
    exterior = dilated_fp & (~floor_union).astype(np.uint8)
    edge_mask = np.zeros((H, W), dtype=np.uint8)
    edge_mask[0:2, :] = 1
    edge_mask[-2:, :] = 1
    edge_mask[:, 0:2] = 1
    edge_mask[:, -2:] = 1
    exterior |= (floor_union & edge_mask)

    for i in range(N):
        if int(nodes[i]) != 5:                    # balcony only
            continue

        bal = (masks[i] > 0).astype(np.uint8)
        bal_area = int(bal.sum())
        if bal_area < 5:
            continue

        # Already on exterior?  Require a meaningful portion of the
        # balcony perimeter to touch the exterior — a single pixel of
        # overlap from dilation is not enough (it can happen when the
        # balcony is barely at the floor-plate boundary but visually
        # still looks interior).
        bal_dilated = cv2.dilate(bal, kernel, iterations=1)
        ext_overlap = int(np.sum(bal_dilated & exterior))
        # Also require the balcony itself (not just dilation) to be
        # within 2px of the grid edge.
        near_edge = (
            np.any(bal[:3, :] > 0) or np.any(bal[-3:, :] > 0) or
            np.any(bal[:, :3] > 0) or np.any(bal[:, -3:] > 0)
        )
        bal_perimeter = int(np.sum(bal_dilated) - bal_area)
        # Accept as exterior only if ≥25% of its expansion ring is exterior
        # AND it is near a grid edge.
        if ext_overlap > bal_perimeter * 0.25 and near_edge:
            continue

        # --- find best parent room (bedroom / living with ext exposure) ---
        candidates = []
        for j in range(N):
            rt_j = int(nodes[j])
            if rt_j not in (1, 3):                # living room or bedroom
                continue
            rm = (masks[j] > 0).astype(np.uint8)
            if rm.sum() < 20:
                continue
            ext_touch = int(np.sum(cv2.dilate(rm, kernel, iterations=1) & exterior))
            if ext_touch == 0:
                continue
            # proximity to current balcony
            prox = int(np.sum(cv2.dilate(bal, kernel, iterations=5) & rm))
            candidates.append((j, ext_touch, prox, rm))

        if not candidates:
            continue

        # prefer room close to balcony, then most exterior exposure
        candidates.sort(key=lambda c: (c[2] > 0, c[1]), reverse=True)
        tgt_idx, _, _, tgt_rm = candidates[0]

        # --- exterior edge of target room --------------------------------
        dil_tgt = cv2.dilate(tgt_rm, kernel, iterations=1)
        room_ext_edge = dil_tgt & (~floor_union).astype(np.uint8)
        room_ext_edge |= (dil_tgt - tgt_rm) & edge_mask
        if room_ext_edge.sum() < 3:
            continue

        ys, xs = np.where(room_ext_edge > 0)
        y_span = int(ys.max() - ys.min())
        x_span = int(xs.max() - xs.min())

        # room centroid (to decide inward / outward direction)
        rys, rxs = np.where(tgt_rm > 0)
        room_cy, room_cx = int(np.mean(rys)), int(np.mean(rxs))

        # --- create rectangular balcony strip ----------------------------
        new_bal = np.zeros((H, W), dtype=np.uint8)
        DEPTH = max(3, min(6, int(np.sqrt(bal_area / 3))))

        if x_span >= y_span:                       # horizontal wall
            mid_x, mid_y = int(np.median(xs)), int(np.median(ys))
            bw = min(int(np.sqrt(bal_area * 2.5)), x_span, W - 2)
            bw = max(bw, 4)
            x0 = max(0, mid_x - bw // 2)
            x1 = min(W, x0 + bw)
            if mid_y < room_cy:                    # wall above room
                y0 = max(0, mid_y - DEPTH)
                y1 = mid_y + 1
            else:                                  # wall below room
                y0 = mid_y
                y1 = min(H, mid_y + DEPTH)
            new_bal[y0:y1, x0:x1] = 1
        else:                                      # vertical wall
            mid_x, mid_y = int(np.median(xs)), int(np.median(ys))
            bh = min(int(np.sqrt(bal_area * 2.5)), y_span, H - 2)
            bh = max(bh, 4)
            y0 = max(0, mid_y - bh // 2)
            y1 = min(H, y0 + bh)
            if mid_x < room_cx:                    # wall left of room
                x0 = max(0, mid_x - DEPTH)
                x1 = mid_x + 1
            else:                                  # wall right of room
                x0 = mid_x
                x1 = min(W, mid_x + DEPTH)
            new_bal[y0:y1, x0:x1] = 1

        if new_bal.sum() < 5:
            continue

        # Clear overlap from other rooms
        for j in range(N):
            if j == i:
                continue
            masks[j] = np.where(new_bal > 0, -1.0, masks[j])

        masks[i] = np.where(new_bal > 0, 1.0, -1.0)

        # Recompute floor_union after relocation
        floor_union = np.zeros((H, W), dtype=np.uint8)
        for k in range(N):
            if int(nodes[k]) not in DOOR_TYPES:
                floor_union |= (masks[k] > 0).astype(np.uint8)
        dilated_fp = cv2.dilate(floor_union, kernel, iterations=2)
        exterior = dilated_fp & (~floor_union).astype(np.uint8)
        exterior |= (floor_union & edge_mask)

    return masks


def _fill_gaps(masks: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Eliminate gaps between rooms by iteratively expanding adjacent rooms.

    Steps:
        1. Build the floor plate and fill its internal holes.
        2. Identify *gap pixels* — inside the plate but unassigned.
        3. Simultaneously dilate every room by 1 px per iteration,
           claiming only gap pixels; ties broken by larger room area.

    Operates in-place and returns *masks*.
    """
    N, H, W = masks.shape
    kernel = np.ones((3, 3), np.uint8)

    # --- 1. floor plate with holes filled --------------------------------
    floor_union = np.zeros((H, W), dtype=np.uint8)
    for i in range(N):
        if int(nodes[i]) not in DOOR_TYPES:
            floor_union |= (masks[i] > 0).astype(np.uint8)

    # Label background components; those NOT touching the border are holes
    inv = (1 - floor_union).astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(inv, connectivity=4)
    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)

    filled = floor_union.copy()
    for lbl in range(1, n_lab):
        if lbl not in border_labels:
            filled[labels == lbl] = 1

    # Also include a slight dilation to close thin 1 px seams
    filled = cv2.dilate(filled, kernel, iterations=1)
    # Re-intersect with a bounding rect so we don't go outside original footprint
    fy, fx = np.where(floor_union > 0)
    if len(fy) == 0:
        return masks
    pad = 2
    r0, r1 = max(0, fy.min() - pad), min(H, fy.max() + pad + 1)
    c0, c1 = max(0, fx.min() - pad), min(W, fx.max() + pad + 1)
    bbox_mask = np.zeros((H, W), dtype=np.uint8)
    bbox_mask[r0:r1, c0:c1] = 1
    filled &= bbox_mask

    # --- 2. gap pixels ----------------------------------------------------
    assigned = np.zeros((H, W), dtype=bool)
    for i in range(N):
        if int(nodes[i]) not in DOOR_TYPES:
            assigned |= (masks[i] > 0)

    gap = (filled > 0) & ~assigned
    if not gap.any():
        return masks

    # --- room areas (for priority) ----------------------------------------
    room_areas = np.array([
        int(np.sum(masks[i] > 0)) if int(nodes[i]) not in DOOR_TYPES else 0
        for i in range(N)
    ], dtype=float)

    # --- 3. iterative simultaneous dilation -------------------------------
    for _ in range(30):
        if not gap.any():
            break

        assignment = np.full((H, W), -1, dtype=int)
        priority   = np.full((H, W), -1.0)

        for i in range(N):
            if int(nodes[i]) in DOOR_TYPES or room_areas[i] < 5:
                continue
            binary = (masks[i] > 0).astype(np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            new_px = (dilated > 0) & gap
            if not new_px.any():
                continue
            better = new_px & (room_areas[i] > priority)
            assignment[better] = i
            priority[better] = room_areas[i]

        if (assignment < 0).all() | (~gap.any()):
            break

        changed = False
        for i in range(N):
            mine = (assignment == i)
            if mine.any():
                masks[i][mine] = 1.0
                gap[mine] = False
                room_areas[i] = float(np.sum(masks[i] > 0))
                changed = True

        if not changed:
            break

    return masks


# ============================================================================
# Mask Post-Processing
# ============================================================================


def _rectangular_gap_fill(masks: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Fill interior gaps by expanding rooms' rectangles, not organically.

    Unlike ``_fill_gaps`` which dilates masks (creating irregular shapes),
    this approach keeps rooms rectangular by expanding their bounding boxes
    outward one row/column at a time into empty space.

    For each iteration:
        1. Compute the floor-plate boundary (convex hull of all rooms).
        2. Find gap pixels (inside floor plate, claimed by no room).
        3. For each room (non-door), try expanding its bounding box by
           1 pixel in each of the 4 directions.
        4. Keep an expansion only if the new row/column is entirely
           empty (no other room claims any pixel in that strip).
        5. Repeat until no more expansions are possible.
    Falls back to organic ``_fill_gaps`` for any remaining stubborn gaps.
    """
    N, H, W = masks.shape

    # Determine the floor-plate region (convex hull of all occupied pixels)
    occupied = np.any(masks > 0, axis=0).astype(np.uint8)
    if occupied.sum() == 0:
        return masks
    filled = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8), iterations=3)
    # Convex hull of floor plate
    pts = np.column_stack(np.where(filled > 0))
    if len(pts) < 3:
        return masks
    from cv2 import convexHull
    hull = convexHull(pts[:, ::-1].astype(np.int32))  # (x, y) format
    hull_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    assigned = np.any(masks > 0, axis=0)
    gap = (hull_mask > 0) & ~assigned

    if not gap.any():
        return masks

    # Iterative rectangular expansion
    for _iter in range(40):
        if not gap.any():
            break

        progress = False
        for i in range(N):
            rt = int(nodes[i])
            if rt in DOOR_TYPES:
                continue

            ys, xs = np.where(masks[i] > 0)
            if len(ys) == 0:
                continue

            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())

            # Try expanding in each direction
            for dr0, dr1, dc0, dc1 in [(-1, 0, 0, 0),   # up
                                         (0, +1, 0, 0),   # down
                                         (0, 0, -1, 0),   # left
                                         (0, 0, 0, +1)]:  # right
                nr0 = r0 + dr0
                nr1 = r1 + dr1
                nc0 = c0 + dc0
                nc1 = c1 + dc1

                if nr0 < 0 or nr1 >= H or nc0 < 0 or nc1 >= W:
                    continue

                # The newly added strip
                if dr0 == -1:
                    strip = gap[nr0, nc0:nc1 + 1]
                    fill_slice = (nr0, slice(nc0, nc1 + 1))
                elif dr1 == 1:
                    strip = gap[nr1, nc0:nc1 + 1]
                    fill_slice = (nr1, slice(nc0, nc1 + 1))
                elif dc0 == -1:
                    strip = gap[nr0:nr1 + 1, nc0]
                    fill_slice = (slice(nr0, nr1 + 1), nc0)
                else:
                    strip = gap[nr0:nr1 + 1, nc1]
                    fill_slice = (slice(nr0, nr1 + 1), nc1)

                # Only expand if ALL pixels in the strip are gap pixels
                # and the strip is inside the floor plate
                if not strip.all():
                    continue

                # Apply expansion
                masks[i][fill_slice] = 1.0
                gap[fill_slice] = False
                progress = True
                # Update bounds for next direction check
                r0, r1, c0, c1 = nr0, nr1, nc0, nc1

        if not progress:
            break

    # Fallback: any remaining gaps use organic fill
    if gap.any():
        masks = _fill_gaps(masks, nodes)

    return masks


def _rectangularize_rooms(masks: np.ndarray, nodes: np.ndarray,
                          respect_others: bool = False) -> np.ndarray:
    """
    Snap each room mask to its axis-aligned bounding rectangle.

    The GAN often produces irregular, organic room shapes.  Real floor
    plans have rectangular rooms.  For every room we:
        1. Find the axis-aligned bounding box of its mask.
        2. Replace the mask with the filled bounding box.

    When *respect_others* is True (used after overlap resolution),
    fill bounding-box pixels only where no other room already claims
    the space — this keeps rooms rectangular without creating new
    overlaps.

    Operates in-place and returns *masks*.
    """
    N, H, W = masks.shape

    # Pre-compute occupancy map when we need to avoid creating overlaps
    if respect_others:
        occupied = np.any(masks > 0, axis=0)  # [H, W]

    for i in range(N):
        rt = int(nodes[i])
        if rt in DOOR_TYPES:
            continue

        binary = (masks[i] > 0).astype(np.uint8)
        area = int(binary.sum())
        if area < 10:
            continue

        ys, xs = np.where(binary > 0)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())

        if respect_others:
            # Fill bbox pixels that are unoccupied or already ours
            own = masks[i] > 0
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if masks[i, r, c] <= 0 and not occupied[r, c]:
                        masks[i, r, c] = 1.0
                        occupied[r, c] = True
        else:
            # Aggressive fill — overlaps resolved later
            new_mask = np.full((H, W), -1.0)
            new_mask[r0:r1 + 1, c0:c1 + 1] = 1.0
            masks[i] = new_mask

    return masks


def post_process_masks(masks: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Clean up generated masks for research-quality output.
    
    Steps:
        1. Hard threshold at 0 (values > 0 → room, ≤ 0 → background)
        2. Keep only the largest connected component per room
        3. Morphological closing to fill small holes (2px)
        4. Rectangularize rooms (snap to axis-aligned bounding box)
        5. Resolve inter-room overlaps via area-weighted majority vote
    
    Args:
        masks:  [N, 64, 64]  raw model output in [-1, 1]
        nodes:  [N]           room type IDs
        
    Returns:
        Cleaned masks [N, 64, 64] with values in {-1, 1}.
    """
    N, H, W = masks.shape
    clean = np.full_like(masks, -1.0)
    kernel_close = np.ones((3, 3), np.uint8)

    for i in range(N):
        rt = int(nodes[i])
        binary = (masks[i] > 0).astype(np.uint8)
        area = binary.sum()
        if area < 5:
            continue

        # Keep largest connected component
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_labels <= 1:
            continue
        # Component 0 is background; find largest foreground
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = (labels == largest_label).astype(np.uint8)

        # Morphological closing (fill small holes)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        clean[i] = np.where(binary > 0, 1.0, -1.0)

    # ---- Rectangularize rooms (snap to bounding box) ----
    clean = _rectangularize_rooms(clean, nodes)

    # ---- Resolve overlaps: per-pixel majority vote weighted by room area ----
    room_areas = np.array([np.sum(clean[i] > 0) for i in range(N)], dtype=float)
    overlap_map = np.sum(clean > 0, axis=0)  # [H, W] count of overlapping rooms

    if np.any(overlap_map > 1):
        contested = np.where(overlap_map > 1)
        for py, px in zip(contested[0], contested[1]):
            candidates = [i for i in range(N) if clean[i, py, px] > 0]
            if len(candidates) <= 1:
                continue
            # Keep the room with the largest total area at this pixel
            winner = max(candidates, key=lambda i: room_areas[i])
            for c in candidates:
                if c != winner:
                    clean[c, py, px] = -1.0

    # ---- Layout refinement (post-overlap) ----
    # 1. Enforce realistic room sizes (erode oversized baths / balconies)
    clean = _enforce_room_sizes(clean, nodes)
    # 2. Move interior balconies to the exterior boundary
    clean = _move_balcony_to_exterior(clean, nodes)
    # 3. Re-rectangularize: fill bbox gaps created by overlap resolution
    clean = _rectangularize_rooms(clean, nodes, respect_others=True)
    # 4. Fill gaps via rectangular expansion (maintains room shape)
    clean = _rectangular_gap_fill(clean, nodes)

    return clean
