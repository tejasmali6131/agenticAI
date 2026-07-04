"""
Requirements Converter for House-GAN++
======================================

Converts Gemini chatbot requirements JSON to House-GAN++ graph format.

Room Types in House-GAN++ (RPLAN dataset):
    1: living_room
    2: kitchen
    3: bedroom
    4: bathroom
    5: balcony
    6: entrance
    7: dining room
    8: study room
    10: storage
    15: front door (exterior connection)
    16: unknown
    17: interior_door
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# House-GAN++ room type mapping
ROOM_CLASS = {
    "living_room": 1, 
    "kitchen": 2, 
    "bedroom": 3, 
    "bathroom": 4, 
    "balcony": 5, 
    "entrance": 6, 
    "dining room": 7, 
    "study room": 8,
    "storage": 10, 
    "front door": 15, 
    "unknown": 16, 
    "interior_door": 17
}

# Reverse mapping
CLASS_ROOM = {v: k for k, v in ROOM_CLASS.items()}

# User-friendly aliases to House-GAN++ types
ROOM_ALIASES = {
    # Living areas
    'living': 1,
    'living_room': 1,
    'living room': 1,
    'hall': 1,
    'drawing': 1,
    'drawing room': 1,
    'lobby': 1,
    
    # Kitchen
    'kitchen': 2,
    'kitchenette': 2,
    'pantry': 2,
    
    # Bedrooms
    'bedroom': 3,
    'master_bedroom': 3,
    'master bedroom': 3,
    'guest_room': 3,
    'guest room': 3,
    'kids_room': 3,
    'kids room': 3,
    "children's room": 3,
    
    # Bathrooms
    'bathroom': 4,
    'toilet': 4,
    'washroom': 4,
    'restroom': 4,
    'wc': 4,
    'attached_bath': 4,
    'attached bath': 4,
    'common_bath': 4,
    'common bath': 4,
    
    # Balcony
    'balcony': 5,
    'terrace': 5,
    'patio': 5,
    'verandah': 5,
    'deck': 5,
    
    # Entrance
    'entrance': 6,
    'foyer': 6,
    'entry': 6,
    'vestibule': 6,
    
    # Dining
    'dining': 7,
    'dining room': 7,
    'dining_room': 7,
    'dining area': 7,
    
    # Study
    'study': 8,
    'study room': 8,
    'study_room': 8,
    'office': 8,
    'home office': 8,
    'workspace': 8,
    'library': 8,
    
    # Storage
    'storage': 10,
    'store': 10,
    'store room': 10,
    'store_room': 10,
    'storeroom': 10,
    'closet': 10,
    'wardrobe': 10,
    'utility': 10,
    'utility room': 10,
    
    # Special
    'pooja': 10,  # Map to storage (closest)
    'pooja room': 10,
    'prayer': 10,
    'prayer room': 10,
    'puja': 10,
    'puja room': 10,
    
    # Misc
    'unknown': 16,
    'other': 16,
}


def map_room_type(room_name: str) -> int:
    """
    Map a room name to House-GAN++ room type ID
    
    Args:
        room_name: Human-readable room name
        
    Returns:
        Room type ID (1-17)
    """
    name_lower = room_name.lower().strip()
    
    # Try exact match first
    if name_lower in ROOM_ALIASES:
        return ROOM_ALIASES[name_lower]
    
    # Try partial match
    for alias, room_id in ROOM_ALIASES.items():
        if alias in name_lower or name_lower in alias:
            return room_id
    
    # Default to unknown
    return 16


class RequirementsConverter:
    """
    Converts Gemini chatbot requirements to House-GAN++ input format.
    
    House-GAN++ expects:
    - nodes: One-hot encoded room types [N, 18]
    - edges: [source, edge_type, destination] where edge_type is 1 (adjacent) or -1 (not adjacent)
    """
    
    def __init__(self):
        self.room_class = ROOM_CLASS
        self.class_room = CLASS_ROOM
    
    def parse_json(self, json_path: str) -> Dict:
        """Load requirements from JSON file"""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def convert(self, requirements: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert requirements dict to House-GAN++ graph format
        
        Args:
            requirements: Dict with rooms, adjacencies, etc.
            
        Returns:
            nodes: Room type IDs [N]
            edges: Edge list [E, 3] with [source, edge_type, destination]
            room_names: List of room names for visualization
        """
        rooms = []
        room_names = []
        
        # Parse rooms from requirements
        if 'rooms' in requirements:
            for room in requirements['rooms']:
                room_name = room.get('name', room.get('type', 'unknown'))
                room_type = room.get('type', room_name)
                room_id = map_room_type(room_type)
                rooms.append(room_id)
                room_names.append(room_name)
        else:
            # Fallback: Generate from num_bedrooms, num_bathrooms, etc.
            rooms, room_names = self._generate_from_counts(requirements)
        
        # Count real rooms before adding doors
        n_real_rooms = len(rooms)
        
        # Add interior doors (type 17) - typically one between major room connections
        # This matches the training data format which has doors between rooms
        num_doors = min(4, n_real_rooms - 1)  # Reasonable number of interior doors
        for i in range(num_doors):
            rooms.append(17)
            room_names.append(f'Door {i+1}')
        
        # Add front door (type 15) - connects to outside
        rooms.append(15)
        room_names.append('Front Door')
        
        nodes = np.array(rooms)
        
        # Generate edges (including door connections)
        edges = self._generate_edges(nodes, room_names, requirements, n_real_rooms)
        
        return nodes, edges, room_names
    
    def _generate_from_counts(self, requirements: Dict) -> Tuple[List[int], List[str]]:
        """Generate room list from bedroom/bathroom counts"""
        rooms = []
        room_names = []
        
        # Living room (always)
        rooms.append(1)
        room_names.append('Living Room')
        
        # Entrance
        rooms.append(6)
        room_names.append('Entrance')
        
        # Kitchen
        rooms.append(2)
        room_names.append('Kitchen')
        
        # Dining (if not combined with living)
        if not requirements.get('combined_living_dining', False):
            rooms.append(7)
            room_names.append('Dining Room')
        
        # Bedrooms
        num_bedrooms = requirements.get('num_bedrooms', 2)
        for i in range(num_bedrooms):
            rooms.append(3)
            if i == 0:
                room_names.append('Master Bedroom')
            else:
                room_names.append(f'Bedroom {i + 1}')
        
        # Bathrooms
        num_bathrooms = requirements.get('num_bathrooms', 1)
        for i in range(num_bathrooms):
            rooms.append(4)
            if i == 0 and num_bathrooms > 1:
                room_names.append('Attached Bath')
            else:
                room_names.append(f'Bathroom {i + 1}')
        
        # Balcony
        if requirements.get('has_balcony', False):
            rooms.append(5)
            room_names.append('Balcony')
        
        # Study
        if requirements.get('has_study', False):
            rooms.append(8)
            room_names.append('Study')
        
        # Storage/Pooja
        if requirements.get('has_pooja', False):
            rooms.append(10)
            room_names.append('Pooja Room')
        
        return rooms, room_names
    
    def _generate_edges(self, nodes: np.ndarray, room_names: List[str], 
                       requirements: Dict, n_real_rooms: int = None) -> np.ndarray:
        """
        Generate adjacency edges based on architectural rules
        
        Rules:
        - Living room connects to entrance, dining, balcony
        - Kitchen connects to dining
        - Master bedroom connects to attached bath
        - Doors connect adjacent rooms
        """
        n_rooms = len(nodes)
        if n_real_rooms is None:
            n_real_rooms = sum(1 for n in nodes if n not in [15, 17])
        
        edges = []
        
        # Default adjacency rules by room type (for real rooms)
        adjacency_rules = {
            1: [6, 7, 5, 3],      # living → entrance, dining, balcony, bedroom
            2: [7, 1, 10],        # kitchen → dining, living, storage
            3: [4, 1, 5],         # bedroom → bathroom, living, balcony
            4: [3],               # bathroom → bedroom
            5: [1, 3],            # balcony → living, bedroom
            6: [1],               # entrance → living
            7: [1, 2],            # dining → living, kitchen
            8: [1, 3],            # study → living, bedroom
            10: [2, 3],           # storage → kitchen, bedroom
        }
        
        # Check user-defined adjacencies
        user_adjacencies = requirements.get('adjacencies', [])
        
        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                type_i = int(nodes[i])
                type_j = int(nodes[j])
                
                is_adjacent = False
                
                # Handle door nodes specially
                if type_i == 15 or type_j == 15:
                    # Front door connects to living (1) or entrance (6)
                    other = type_j if type_i == 15 else type_i
                    is_adjacent = other in [1, 6]
                elif type_i == 17 or type_j == 17:
                    # Interior doors connect to any real room
                    other = type_j if type_i == 17 else type_i
                    is_adjacent = other not in [15, 17]  # Connect to real rooms
                else:
                    # Real rooms - use adjacency rules
                    if type_j in adjacency_rules.get(type_i, []):
                        is_adjacent = True
                    if type_i in adjacency_rules.get(type_j, []):
                        is_adjacent = True
                    
                    # Check user adjacencies
                    if not is_adjacent and i < len(room_names) and j < len(room_names):
                        name_i = room_names[i].lower()
                        name_j = room_names[j].lower()
                        for adj in user_adjacencies:
                            r1, r2 = adj.get('room1', '').lower(), adj.get('room2', '').lower()
                            if (r1 in name_i and r2 in name_j) or (r2 in name_i and r1 in name_j):
                                is_adjacent = True
                                break
                
                # Add edge
                edge_type = 1 if is_adjacent else -1
                edges.append([i, edge_type, j])
        
        return np.array(edges) if edges else np.zeros((0, 3), dtype=np.int64)
    
    def to_one_hot(self, nodes: np.ndarray, num_classes: int = 19) -> np.ndarray:
        """
        Convert room type IDs to one-hot encoding (House-GAN++ format)
        
        House-GAN++ uses 19 classes (0-18) but removes column 0, giving 18 final columns.
        Room type IDs map to: living_room=1, kitchen=2, etc.
        After [:, 1:], living_room is at index 0, kitchen at index 1, etc.
        """
        # Create one-hot with 19 classes
        one_hot = np.zeros((len(nodes), num_classes))
        for i, node in enumerate(nodes):
            if 0 <= node < num_classes:
                one_hot[i, node] = 1
            else:
                one_hot[i, 1] = 1  # Default to living_room
        
        # Remove first column to match House-GAN++ format (18 columns)
        return one_hot[:, 1:]


def convert_requirements_file(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to convert a requirements JSON file
    
    Args:
        json_path: Path to requirements JSON
        
    Returns:
        nodes, edges, room_names
    """
    converter = RequirementsConverter()
    requirements = converter.parse_json(json_path)
    return converter.convert(requirements)


if __name__ == "__main__":
    # Test conversion
    test_requirements = {
        "num_bedrooms": 2,
        "num_bathrooms": 2,
        "has_balcony": True,
        "combined_living_dining": False
    }
    
    converter = RequirementsConverter()
    nodes, edges, names = converter.convert(test_requirements)
    
    print("Test Requirements Conversion:")
    print(f"  Rooms ({len(nodes)}): {names}")
    print(f"  Node types: {nodes}")
    print(f"  Edges ({len(edges)}): {edges[:5]}..." if len(edges) > 5 else f"  Edges: {edges}")
