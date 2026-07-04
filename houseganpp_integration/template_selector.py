"""
Template-Based Floor Plan Generation (Research Grade)
=====================================================

Selects the best-matching House-GAN++ template for given user requirements.
Templates are pre-existing JSON files from the RPLAN dataset with
architecturally valid graph structures the model was trained on.

Why templates produce better results:
    The House-GAN++ generator was trained on exactly these graph structures.
    Using a template graph guarantees the nodes, edges, and door placements
    match patterns the model has seen during training, yielding higher
    quality output than a synthetically constructed graph.

Scoring rubric:
    - Bedroom count match:   weight 10 (per bedroom difference)
    - Bathroom count match:  weight 5
    - Balcony match:         weight 3
    - Total room count:      weight 2 (prefer similar total)
    - Living + Kitchen:      mandatory penalty if missing
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TemplateMatch:
    """Represents an analysed template with match scoring."""
    path: str
    score: float
    room_types: List[int]
    num_bedrooms: int
    num_bathrooms: int
    num_balconies: int
    has_living: bool
    has_kitchen: bool
    has_dining: bool
    has_study: bool
    has_storage: bool
    total_rooms: int          # real rooms only (excl. doors)
    total_nodes: int          # all nodes including doors
    num_interior_doors: int
    num_front_doors: int

    @property
    def has_balcony(self) -> bool:
        return self.num_balconies > 0


class TemplateSelector:
    """
    Selects the best template from available House-GAN++ JSON files.
    """

    def __init__(self, template_dir: str = None):
        if template_dir is None:
            self.template_dir = Path(__file__).parent.parent / 'houseganpp' / 'data' / 'json'
        else:
            self.template_dir = Path(template_dir)

        self.templates = self._load_all_templates()

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_all_templates(self) -> List[TemplateMatch]:
        templates = []
        if not self.template_dir.exists():
            print(f"Warning: Template directory not found: {self.template_dir}")
            return templates

        for json_file in self.template_dir.glob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                room_types = data.get('room_type', [])
                real_rooms = [r for r in room_types if r not in [15, 17]]

                template = TemplateMatch(
                    path=str(json_file),
                    score=0.0,
                    room_types=room_types,
                    num_bedrooms=sum(1 for r in real_rooms if r == 3),
                    num_bathrooms=sum(1 for r in real_rooms if r == 4),
                    num_balconies=sum(1 for r in real_rooms if r == 5),
                    has_living=any(r == 1 for r in real_rooms),
                    has_kitchen=any(r == 2 for r in real_rooms),
                    has_dining=any(r == 7 for r in real_rooms),
                    has_study=any(r == 8 for r in real_rooms),
                    has_storage=any(r == 10 for r in real_rooms),
                    total_rooms=len(real_rooms),
                    total_nodes=len(room_types),
                    num_interior_doors=sum(1 for r in room_types if r == 17),
                    num_front_doors=sum(1 for r in room_types if r == 15),
                )
                templates.append(template)

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        return templates

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_best_match(self, requirements: Dict) -> Optional[TemplateMatch]:
        """
        Find the template that best matches requirements.

        Scoring (lower = better):
            - |bedrooms_diff|  * 10
            - |bathrooms_diff| * 5
            - balcony mismatch * 3
            - |total_rooms_diff| * 2
            - missing living   * 50
            - missing kitchen  * 40
            - dining / study / storage mismatch * 2 each

        Args:
            requirements: Dict with num_bedrooms, num_bathrooms, etc.

        Returns:
            Best matching template or None.
        """
        if not self.templates:
            return None

        req_bed = requirements.get('num_bedrooms', 2)
        req_bath = requirements.get('num_bathrooms', 1)
        req_balcony = requirements.get('has_balcony', False)
        req_dining = not requirements.get('combined_living_dining', True)
        req_study = requirements.get('has_study', False)
        req_storage = requirements.get('has_storage', False)

        # Estimate expected real room count
        expected_total = 2 + req_bed + req_bath  # living + kitchen + beds + baths
        if req_balcony:
            expected_total += 1
        if req_dining:
            expected_total += 1
        if req_study:
            expected_total += 1
        if req_storage:
            expected_total += 1

        best_template = None
        best_score = float('inf')

        for t in self.templates:
            score = 0.0

            # --- Critical matches ---
            score += abs(t.num_bedrooms - req_bed) * 10
            score += abs(t.num_bathrooms - req_bath) * 5

            if not t.has_living:
                score += 50
            if not t.has_kitchen:
                score += 40

            # --- Important matches ---
            if req_balcony and not t.has_balcony:
                score += 3
            elif not req_balcony and t.has_balcony:
                score += 15  # strongly penalise unwanted balcony

            # --- Secondary matches ---
            if req_dining and not t.has_dining:
                score += 2
            if req_study and not t.has_study:
                score += 2
            if req_storage and not t.has_storage:
                score += 2

            # Total room count difference
            score += abs(t.total_rooms - expected_total) * 2

            t.score = score
            if score < best_score:
                best_score = score
                best_template = t

        return best_template

    def find_top_matches(self, requirements: Dict, top_k: int = 3) -> List[TemplateMatch]:
        """Return top-k best matching templates sorted by score (ascending)."""
        if not self.templates:
            return []

        # Score all templates
        self.find_best_match(requirements)  # sets .score on each
        scored = sorted(self.templates, key=lambda t: t.score)
        return scored[:top_k]

    def get_all_templates_info(self) -> List[Dict]:
        """Info about all available templates."""
        return [
            {
                'path': t.path,
                'bedrooms': t.num_bedrooms,
                'bathrooms': t.num_bathrooms,
                'balconies': t.num_balconies,
                'has_dining': t.has_dining,
                'has_study': t.has_study,
                'total_rooms': t.total_rooms,
                'total_nodes': t.total_nodes,
            }
            for t in self.templates
        ]


# ============================================================================
# Edge extraction from template ed_rm
# ============================================================================

def create_edges_from_ed_rm(ed_rm: List[List[int]], n_rooms: int) -> np.ndarray:
    """
    Create edge triples from ed_rm (edge-to-room mapping).

    The ``ed_rm`` structure encodes which rooms share each wall segment.
    We derive adjacency from wall sharing.

    Returns:
        Edge array [E, 3] with [src, relation, dst].
    """
    adjacency = set()

    for edge_rooms in ed_rm:
        if len(edge_rooms) >= 2:
            for i in range(len(edge_rooms)):
                for j in range(i + 1, len(edge_rooms)):
                    r1, r2 = edge_rooms[i], edge_rooms[j]
                    if r1 != r2:
                        adjacency.add((min(r1, r2), max(r1, r2)))

    edges = []
    for i in range(n_rooms):
        for j in range(i + 1, n_rooms):
            is_adj = (i, j) in adjacency
            edges.append([i, 1 if is_adj else -1, j])

    return np.array(edges, dtype=np.int64) if edges else np.zeros((0, 3), dtype=np.int64)


if __name__ == "__main__":
    selector = TemplateSelector()

    print(f"Found {len(selector.templates)} templates\n")
    for t in selector.templates:
        print(f"  {Path(t.path).name}: "
              f"{t.num_bedrooms}BR/{t.num_bathrooms}BA/{t.num_balconies}Bal  "
              f"rooms={t.total_rooms}  nodes={t.total_nodes}  "
              f"doors={t.num_interior_doors}+{t.num_front_doors}")

    for label, req in [
        ("1 BHK", {"num_bedrooms": 1, "num_bathrooms": 1, "has_balcony": True}),
        ("2 BHK", {"num_bedrooms": 2, "num_bathrooms": 1, "has_balcony": True}),
        ("3 BHK", {"num_bedrooms": 3, "num_bathrooms": 2, "has_balcony": True}),
    ]:
        best = selector.find_best_match(req)
        if best:
            print(f"\n  Best for {label}: {Path(best.path).name} "
                  f"({best.num_bedrooms}BR/{best.num_bathrooms}BA, score={best.score:.0f})")
