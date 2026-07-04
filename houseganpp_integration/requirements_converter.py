"""
Requirements Converter for House-GAN++ (Research Grade)
=======================================================

Converts user requirements to an architecturally-valid House-GAN++ graph.

Graph Design Principles (derived from RPLAN training-data analysis):
    1. Hub-and-spoke topology — Living Room is the central hub.
    2. Each non-hub real room gets exactly 1 interior door connecting it
       to its *primary neighbour* (usually the living room).
    3. Attached bathrooms connect exclusively to their parent bedroom
       (door opens into that bedroom only).
    4. Common bathrooms connect to the living room (corridor / common area).
    5. Kitchen is always adjacent to the living room.
    6. Balcony is accessible from the living room or a bedroom.
    7. Front door connects to the living room.
    8. Forbidden adjacencies (e.g. kitchen ↔ bathroom) are enforced.

Node ordering convention (matching RPLAN templates):
    [bedrooms..., bathrooms..., kitchen, balcony, other..., living_room,
     interior_doors..., front_door]

Edge format:  [source_idx, relation, dest_idx]
    relation =  1  → rooms are adjacent (share a wall / door)
    relation = -1  → rooms are NOT adjacent
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .architectural_constraints import (
    ROOM_TYPES, REAL_ROOM_TYPES, DOOR_TYPES,
    ROOM_DISPLAY_NAMES, ROOM_ALIASES,
    MANDATORY_ADJACENCIES, PREFERRED_ADJACENCIES, FORBIDDEN_ADJACENCIES,
    SINGLE_DOOR_ROOMS, MUST_BE_EXTERIOR,
    map_room_type, validate_requirements,
    RoomNode, DoorConnection,
)


# ============================================================================
# Room Role Constants
# ============================================================================

ROLE_HUB = 'living_hub'
ROLE_MASTER_BED = 'master_bedroom'
ROLE_BEDROOM = 'bedroom'
ROLE_ATTACHED_BATH = 'attached_bath'
ROLE_COMMON_BATH = 'common_bath'
ROLE_KITCHEN = 'kitchen'
ROLE_BALCONY = 'balcony'
ROLE_DINING = 'dining'
ROLE_ENTRANCE = 'entrance'
ROLE_STUDY = 'study'
ROLE_STORAGE = 'storage'
ROLE_DOOR = 'interior_door'
ROLE_FRONT_DOOR = 'front_door'


# Room class mapping (kept for backward compatibility)
ROOM_CLASS = {
    "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4,
    "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
    "storage": 10, "front door": 15, "unknown": 16, "interior_door": 17,
}
CLASS_ROOM = {v: k for k, v in ROOM_CLASS.items()}


class RequirementsConverter:
    """
    Converts chatbot / user requirements into a House-GAN++ graph.

    Usage::

        converter = RequirementsConverter()
        nodes, edges, room_names = converter.convert(requirements_dict)
    """

    def __init__(self):
        self.room_class = ROOM_CLASS
        self.class_room = CLASS_ROOM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_json(self, json_path: str) -> Dict:
        """Load requirements from a JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)

    def convert(self, requirements: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Full conversion pipeline.

        Args:
            requirements: Dict with keys like ``num_bedrooms``, ``num_bathrooms``,
                ``has_balcony``, ``combined_living_dining``, etc.

        Returns:
            nodes:      np.ndarray  [N]     — room-type IDs
            edges:      np.ndarray  [E, 3]  — edge triples
            room_names: List[str]           — display names for each node
        """
        # Step 1 — validate & normalise
        req, warnings = validate_requirements(requirements)
        for w in warnings:
            print(f"  [WARN] {w}")

        # Step 2 — build ordered room list with roles
        room_nodes = self._build_room_list(req)

        # Step 3 — plan door connections (1 per non-hub real room)
        door_connections = self._plan_door_connections(room_nodes, req)

        # Step 4 — append door nodes
        n_real = len(room_nodes)
        for i, dc in enumerate(door_connections):
            room_nodes.append(RoomNode(
                index=n_real + i,
                room_type=17,
                display_name=f'Door {i + 1}',
                role=ROLE_DOOR,
                door_target_idx=-1,
            ))

        # Append front door
        front_door_idx = len(room_nodes)
        room_nodes.append(RoomNode(
            index=front_door_idx,
            room_type=15,
            display_name='Front Door',
            role=ROLE_FRONT_DOOR,
            door_target_idx=-1,
        ))

        # Step 5 — rebuild index mapping
        for i, rn in enumerate(room_nodes):
            rn.index = i

        # Step 6 — build edge list
        nodes_arr = np.array([rn.room_type for rn in room_nodes])
        names = [rn.display_name for rn in room_nodes]
        edges = self._build_edges(room_nodes, door_connections, n_real, front_door_idx)

        return nodes_arr, edges, names

    # ------------------------------------------------------------------
    # Room list construction
    # ------------------------------------------------------------------

    def _build_room_list(self, req: Dict) -> List[RoomNode]:
        """
        Build an ordered list of real room nodes according to the convention:
        [bedrooms, bathrooms, kitchen, balcony, other, living_room].

        Bathroom role assignment:
            - If bathrooms >= 2: first bathroom is *attached* to master bedroom.
            - All other bathrooms are *common* (connect to living room).
            - If bathrooms == 1 and bedrooms == 1: attached to the sole bedroom.
            - If bathrooms == 1 and bedrooms > 1: common bathroom.
        """
        rooms: List[RoomNode] = []
        idx = 0

        num_bedrooms = req['num_bedrooms']
        num_bathrooms = req['num_bathrooms']

        # --- Bedrooms ---
        for b in range(num_bedrooms):
            role = ROLE_MASTER_BED if b == 0 else ROLE_BEDROOM
            name = 'Master Bedroom' if b == 0 else f'Bedroom {b + 1}'
            rooms.append(RoomNode(idx, 3, name, role, -1))
            idx += 1

        # --- Bathrooms (with role assignment) ---
        master_bed_idx = 0  # always first in the list
        for b in range(num_bathrooms):
            if b == 0 and (num_bathrooms >= 2 or num_bedrooms == 1):
                # First bathroom → attached to master bedroom
                rooms.append(RoomNode(idx, 4, 'Attached Bath', ROLE_ATTACHED_BATH, master_bed_idx))
                idx += 1
            else:
                bath_name = f'Bathroom {b + 1}' if num_bathrooms > 1 else 'Bathroom'
                rooms.append(RoomNode(idx, 4, bath_name, ROLE_COMMON_BATH, -1))
                idx += 1

        # --- Kitchen ---
        rooms.append(RoomNode(idx, 2, 'Kitchen', ROLE_KITCHEN, -1))
        idx += 1

        # --- Balcony (optional) ---
        if req.get('has_balcony', False):
            rooms.append(RoomNode(idx, 5, 'Balcony', ROLE_BALCONY, -1))
            idx += 1

        # --- Dining Room (optional, only if NOT combined with living) ---
        if not req.get('combined_living_dining', True):
            rooms.append(RoomNode(idx, 7, 'Dining Room', ROLE_DINING, -1))
            idx += 1

        # --- Study (optional) ---
        if req.get('has_study', False):
            rooms.append(RoomNode(idx, 8, 'Study', ROLE_STUDY, -1))
            idx += 1

        # --- Storage (optional) ---
        if req.get('has_storage', False):
            rooms.append(RoomNode(idx, 10, 'Storage', ROLE_STORAGE, -1))
            idx += 1

        # --- Living Room (ALWAYS last among real rooms — hub) ---
        rooms.append(RoomNode(idx, 1, 'Living Room', ROLE_HUB, -1))

        return rooms

    # ------------------------------------------------------------------
    # Door connection planning
    # ------------------------------------------------------------------

    def _plan_door_connections(self, room_nodes: List[RoomNode],
                               req: Dict) -> List[DoorConnection]:
        """
        Assign one interior door per non-hub real room.

        Placement rules (architectural best-practice):
            - Front door → living room (entrance opens into living area)
            - Attached bathroom → parent bedroom ONLY (single door access)
            - Common bathroom → living room (corridor / common area, NOT
              inside kitchen, dining, or a bedroom)
            - Kitchen → living room (adjacent, but NOT facing main entrance)
            - Dining → living room (between living and kitchen zone)
            - Balcony → living room OR master bedroom (external wall, NOT
              attached to bathroom)
            - Study → living room
            - Storage/Pooja → kitchen (utility proximity) or living/dining
            - Bedrooms → living room (hub-and-spoke; bedrooms should NOT
              open directly to main entrance)
            - Guest bedroom → living room (near common bath)
            - Children's bedroom → living room (near master bedroom in
              the graph; physical proximity is via adjacency edges)

        No bedroom should be used as a passage to reach another bedroom
        (each bedroom has a single door from the hub / corridor).
        """
        living_idx = self._find_hub(room_nodes)
        kitchen_idx = self._find_by_role(room_nodes, ROLE_KITCHEN)
        master_idx = self._find_by_role(room_nodes, ROLE_MASTER_BED)
        connections: List[DoorConnection] = []
        door_idx_start = len(room_nodes)
        dc_counter = 0

        # Determine balcony target: prefer master bedroom if > 1 bedroom,
        # otherwise living room.
        num_beds = req.get('num_bedrooms', 2)
        balcony_target = master_idx if (master_idx is not None and num_beds > 1) else living_idx

        for rn in room_nodes:
            if rn.role == ROLE_HUB:
                continue  # hub has no dedicated door

            if rn.role == ROLE_ATTACHED_BATH:
                # Rule: attached bath entry ONLY from parent bedroom
                target = rn.door_target_idx  # parent bedroom index
            elif rn.role == ROLE_COMMON_BATH:
                # Rule: common bath opens into passage/corridor (= living hub)
                # NOT inside kitchen, dining, or bedroom
                target = living_idx
            elif rn.role == ROLE_KITCHEN:
                # Rule: kitchen adjacent to living, NOT facing entrance
                target = living_idx
            elif rn.role == ROLE_DINING:
                # Rule: dining between living and kitchen
                target = living_idx
            elif rn.role == ROLE_BALCONY:
                # Rule: balcony on external wall; from living or master bedroom
                target = balcony_target
            elif rn.role == ROLE_STORAGE:
                # Rule: storage/pooja near kitchen (utility) or living
                target = kitchen_idx if kitchen_idx is not None else living_idx
            elif rn.role == ROLE_STUDY:
                # Rule: study connected to living room
                target = living_idx
            elif rn.role == ROLE_ENTRANCE:
                # Rule: foyer connects to living room
                target = living_idx
            else:
                # Bedrooms (master, guest, children's) → living room hub
                # This ensures no bedroom is a passage to another bedroom
                target = living_idx

            rn.door_target_idx = target
            connections.append(DoorConnection(
                door_index=door_idx_start + dc_counter,
                room_a_index=rn.index,
                room_b_index=target,
            ))
            dc_counter += 1

        return connections

    # ------------------------------------------------------------------
    # Edge list construction
    # ------------------------------------------------------------------

    def _build_edges(self, room_nodes: List[RoomNode],
                     door_connections: List[DoorConnection],
                     n_real: int,
                     front_door_idx: int) -> np.ndarray:
        """
        Build the complete edge list with proper architectural adjacencies.

        For every pair (i, j) we emit exactly one triple [i, rel, j].

        Adjacency (rel = 1) when:
            (a) Two rooms share an interior door
            (b) A door node and one of its two connected rooms
            (c) Front door and the living room
            (d) Mandatory / preferred architectural adjacencies

        Non-adjacency (rel = -1) for everything else.

        Enforced rules:
            - Forbidden adjacencies always override to -1.
            - Attached bath isolation: only adjacent to parent bedroom + own door.
            - Bedrooms NOT adjacent to entrance.
            - Kitchen NOT adjacent to entrance.
            - Bathroom NOT adjacent to entrance, kitchen, dining, or balcony.
            - No bedroom-to-bedroom adjacency (prevents pass-through rooms).
            - Single-door rooms (bedrooms, bathrooms) connect to hub only
              through their one assigned door.
        """
        N = len(room_nodes)
        types = {rn.index: rn.room_type for rn in room_nodes}
        roles = {rn.index: rn.role for rn in room_nodes}
        living_idx = self._find_hub(room_nodes)

        # Pre-compute adjacent pairs as a set of (min, max) tuples
        adj_pairs: set = set()

        # (a) Rooms connected by a door → mutually adjacent
        for dc in door_connections:
            a, b = dc.room_a_index, dc.room_b_index
            adj_pairs.add((min(a, b), max(a, b)))

        # (b) Door nodes adjacent to their two rooms
        for dc in door_connections:
            d = dc.door_index
            adj_pairs.add((min(d, dc.room_a_index), max(d, dc.room_a_index)))
            adj_pairs.add((min(d, dc.room_b_index), max(d, dc.room_b_index)))

        # (c) Front door ↔ living room
        adj_pairs.add((min(front_door_idx, living_idx), max(front_door_idx, living_idx)))

        # (d) Mandatory / preferred adjacencies between real rooms
        for i in range(n_real):
            for j in range(i + 1, n_real):
                ti, tj = types[i], types[j]
                pair_fwd = (ti, tj)
                pair_rev = (tj, ti)

                if pair_fwd in MANDATORY_ADJACENCIES or pair_rev in MANDATORY_ADJACENCIES:
                    adj_pairs.add((i, j))

                if pair_fwd in PREFERRED_ADJACENCIES or pair_rev in PREFERRED_ADJACENCIES:
                    adj_pairs.add((i, j))

        # ---- Remove forbidden adjacencies ----
        forbidden_rm = set()
        for i in range(n_real):
            for j in range(i + 1, n_real):
                ti, tj = types[i], types[j]
                if (ti, tj) in FORBIDDEN_ADJACENCIES or (tj, ti) in FORBIDDEN_ADJACENCIES:
                    forbidden_rm.add((i, j))
        adj_pairs -= forbidden_rm

        # ---- No bedroom-to-bedroom adjacency ----
        # Prevents using one bedroom as a passage to another.
        for i in range(n_real):
            for j in range(i + 1, n_real):
                if types[i] == 3 and types[j] == 3:
                    adj_pairs.discard((i, j))

        # ---- Enforce attached-bath isolation ----
        # Attached bath adjacent ONLY to parent bedroom and its own door node.
        for rn in room_nodes:
            if rn.role == ROLE_ATTACHED_BATH:
                bath_i = rn.index
                parent_bed = rn.door_target_idx
                for j in range(N):
                    if j == bath_i:
                        continue
                    pair = (min(bath_i, j), max(bath_i, j))
                    if pair in adj_pairs:
                        is_parent = (j == parent_bed)
                        is_own_door = any(
                            dc.door_index == j and dc.room_a_index == bath_i
                            for dc in door_connections
                        )
                        if not is_parent and not is_own_door:
                            adj_pairs.discard(pair)

        # ---- Common bath isolation ----
        # Common bath should NOT be adjacent to kitchen, dining, bedroom, balcony.
        # Only adjacent to living room (corridor) and its own door.
        for rn in room_nodes:
            if rn.role == ROLE_COMMON_BATH:
                bath_i = rn.index
                for j in range(N):
                    if j == bath_i:
                        continue
                    pair = (min(bath_i, j), max(bath_i, j))
                    if pair in adj_pairs:
                        oj = types.get(j, 0)
                        is_living = (j == living_idx)
                        is_own_door = any(
                            dc.door_index == j and dc.room_a_index == bath_i
                            for dc in door_connections
                        )
                        if not is_living and not is_own_door:
                            # Only keep adjacency to living room hub + own door
                            adj_pairs.discard(pair)

        # ---- Balcony isolation ----
        # Balcony must be on an exterior wall.  In the graph it should
        # only be adjacent to its door-target room (master bedroom or
        # living room) + the living hub + its own door node.
        # Remove adjacency with ALL other rooms (prevents the GAN from
        # pulling the balcony into the interior to satisfy many edges).
        for rn in room_nodes:
            if rn.role == ROLE_BALCONY:
                bal_i = rn.index
                bal_target = rn.door_target_idx  # master bed or living
                for j in range(N):
                    if j == bal_i:
                        continue
                    pair = (min(bal_i, j), max(bal_i, j))
                    if pair in adj_pairs:
                        is_target = (j == bal_target)
                        is_living = (j == living_idx)
                        is_own_door = any(
                            dc.door_index == j and dc.room_a_index == bal_i
                            for dc in door_connections
                        )
                        if not is_target and not is_living and not is_own_door:
                            adj_pairs.discard(pair)

        # Build full edge list
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                rel = 1 if (i, j) in adj_pairs else -1
                edges.append([i, rel, j])

        return np.array(edges, dtype=np.int64) if edges else np.zeros((0, 3), dtype=np.int64)

    # ------------------------------------------------------------------
    # One-hot encoding (House-GAN++ format)
    # ------------------------------------------------------------------

    @staticmethod
    def to_one_hot(nodes: np.ndarray, num_classes: int = 19) -> np.ndarray:
        """
        Convert room-type IDs to one-hot vectors (18-dim after dropping col 0).
        """
        one_hot = np.zeros((len(nodes), num_classes))
        for i, nid in enumerate(nodes):
            if 0 <= nid < num_classes:
                one_hot[i, nid] = 1
            else:
                one_hot[i, 1] = 1
        return one_hot[:, 1:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_hub(room_nodes: List[RoomNode]) -> int:
        for rn in room_nodes:
            if rn.role == ROLE_HUB:
                return rn.index
        raise ValueError("No living room hub found")

    @staticmethod
    def _find_by_role(room_nodes: List[RoomNode], role: str) -> Optional[int]:
        for rn in room_nodes:
            if rn.role == role:
                return rn.index
        return None

    def get_graph_summary(self, nodes: np.ndarray, edges: np.ndarray,
                          room_names: List[str]) -> str:
        """Human-readable graph summary."""
        n_total = len(nodes)
        real = [i for i in range(n_total) if int(nodes[i]) in REAL_ROOM_TYPES]
        doors = [i for i in range(n_total) if int(nodes[i]) == 17]
        front = [i for i in range(n_total) if int(nodes[i]) == 15]
        n_adj = sum(1 for e in edges if int(e[1]) == 1)
        n_nonadj = sum(1 for e in edges if int(e[1]) == -1)

        lines = [
            "=" * 55,
            "GRAPH SUMMARY",
            "=" * 55,
            f"Total nodes:       {n_total}",
            f"  Real rooms:      {len(real)}",
            f"  Interior doors:  {len(doors)}",
            f"  Front doors:     {len(front)}",
            f"Door:room ratio:   {len(doors)}:{len(real)} "
            f"({'OK (N-1):N' if len(doors) == len(real) - 1 else 'MISMATCH'})",
            f"Edge triples:      {len(edges)}",
            f"  Adjacent (+1):   {n_adj}",
            f"  Non-adj  (-1):   {n_nonadj}",
            "-" * 55,
            "Room List:",
        ]
        for i, name in enumerate(room_names):
            rt = int(nodes[i])
            label = ROOM_DISPLAY_NAMES.get(rt, f'type-{rt}')
            lines.append(f"  [{i:2d}] {name:<22s} (type {rt:2d}: {label})")

        lines.append("-" * 55)
        lines.append("Adjacent Pairs (relation = +1):")
        for e in edges:
            if int(e[1]) == 1:
                si, di = int(e[0]), int(e[2])
                lines.append(f"  {room_names[si]:<22s} <-> {room_names[di]}")
        lines.append("=" * 55)
        return "\n".join(lines)


# ============================================================================
# Convenience
# ============================================================================

def convert_requirements_file(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a requirements JSON and convert to graph."""
    converter = RequirementsConverter()
    requirements = converter.parse_json(json_path)
    return converter.convert(requirements)


if __name__ == "__main__":
    from .architectural_constraints import validate_graph

    configs = [
        {"label": "1 BHK", "num_bedrooms": 1, "num_bathrooms": 1,
         "has_balcony": True, "combined_living_dining": True},
        {"label": "2 BHK", "num_bedrooms": 2, "num_bathrooms": 2,
         "has_balcony": True, "combined_living_dining": False},
        {"label": "3 BHK", "num_bedrooms": 3, "num_bathrooms": 2,
         "has_balcony": True, "combined_living_dining": False, "has_study": True},
    ]

    converter = RequirementsConverter()

    for cfg in configs:
        label = cfg.pop('label')
        print(f"\n{'#' * 60}")
        print(f"  {label}")
        print(f"{'#' * 60}")
        nodes, edges, names = converter.convert(cfg)
        print(converter.get_graph_summary(nodes, edges, names))

        vr = validate_graph(nodes, edges, names)
        if vr.errors:
            print("ERRORS:", vr.errors)
        if vr.warnings:
            print("WARNINGS:", vr.warnings)
        print(f"Validation: {'PASS' if vr.is_valid else 'FAIL'}")
