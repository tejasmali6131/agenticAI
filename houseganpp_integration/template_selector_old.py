"""
Template-Based Floor Plan Generation
====================================

Uses pre-existing House-GAN++ templates that closely match user requirements.
This approach produces much better results because the model was trained on
these exact graph structures.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TemplateMatch:
    """Represents a matched template"""
    path: str
    score: float
    room_types: List[int]
    num_bedrooms: int
    num_bathrooms: int
    has_balcony: bool
    has_living: bool
    has_kitchen: bool
    total_rooms: int


class TemplateSelector:
    """
    Selects the best template from available House-GAN++ JSON files
    based on user requirements.
    """
    
    def __init__(self, template_dir: str = None):
        """
        Initialize selector with template directory.
        
        Args:
            template_dir: Path to directory containing JSON templates
        """
        if template_dir is None:
            # Default to houseganpp/data/json
            self.template_dir = Path(__file__).parent.parent / 'houseganpp' / 'data' / 'json'
        else:
            self.template_dir = Path(template_dir)
        
        self.templates = self._load_all_templates()
        
    def _load_all_templates(self) -> List[TemplateMatch]:
        """Load and analyze all templates"""
        templates = []
        
        if not self.template_dir.exists():
            print(f"Warning: Template directory not found: {self.template_dir}")
            return templates
        
        for json_file in self.template_dir.glob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                room_types = data.get('room_type', [])
                
                # Count room types (excluding doors)
                real_rooms = [r for r in room_types if r not in [15, 17]]
                
                template = TemplateMatch(
                    path=str(json_file),
                    score=0.0,
                    room_types=room_types,
                    num_bedrooms=sum(1 for r in real_rooms if r == 3),
                    num_bathrooms=sum(1 for r in real_rooms if r == 4),
                    has_balcony=any(r == 5 for r in real_rooms),
                    has_living=any(r == 1 for r in real_rooms),
                    has_kitchen=any(r == 2 for r in real_rooms),
                    total_rooms=len(real_rooms)
                )
                templates.append(template)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return templates
    
    def find_best_match(self, requirements: Dict) -> Optional[TemplateMatch]:
        """
        Find the template that best matches requirements.
        
        Args:
            requirements: Dict with num_bedrooms, num_bathrooms, etc.
            
        Returns:
            Best matching template or None
        """
        if not self.templates:
            return None
        
        req_bedrooms = requirements.get('num_bedrooms', 2)
        req_bathrooms = requirements.get('num_bathrooms', 1)
        req_balcony = requirements.get('has_balcony', False)
        
        best_template = None
        best_score = float('inf')
        
        for template in self.templates:
            # Calculate match score (lower is better)
            score = 0
            
            # Bedroom difference (weighted heavily)
            score += abs(template.num_bedrooms - req_bedrooms) * 10
            
            # Bathroom difference
            score += abs(template.num_bathrooms - req_bathrooms) * 5
            
            # Balcony match
            if req_balcony and not template.has_balcony:
                score += 3
            elif not req_balcony and template.has_balcony:
                score += 1  # Having extra balcony is less bad
            
            # Must have living room and kitchen
            if not template.has_living:
                score += 20
            if not template.has_kitchen:
                score += 15
            
            template.score = score
            
            if score < best_score:
                best_score = score
                best_template = template
        
        return best_template
    
    def get_all_templates_info(self) -> List[Dict]:
        """Get info about all available templates"""
        return [
            {
                'path': t.path,
                'bedrooms': t.num_bedrooms,
                'bathrooms': t.num_bathrooms,
                'has_balcony': t.has_balcony,
                'total_rooms': t.total_rooms
            }
            for t in self.templates
        ]


def create_edges_from_ed_rm(ed_rm: List[List[int]], n_rooms: int) -> np.ndarray:
    """
    Create edge triples from ed_rm (edge to room mapping).
    
    The ed_rm structure tells us which rooms share each edge segment.
    We convert this to [src, relation, dst] triples.
    
    Args:
        ed_rm: List of lists, where each inner list contains room indices
        n_rooms: Total number of rooms
        
    Returns:
        Edge array [E, 3]
    """
    # Build adjacency from ed_rm
    adjacency = set()
    
    for edge_rooms in ed_rm:
        if len(edge_rooms) >= 2:
            # This edge is shared by multiple rooms
            for i in range(len(edge_rooms)):
                for j in range(i + 1, len(edge_rooms)):
                    r1, r2 = edge_rooms[i], edge_rooms[j]
                    if r1 != r2:
                        adjacency.add((min(r1, r2), max(r1, r2)))
    
    # Create edge triples
    edges = []
    for i in range(n_rooms):
        for j in range(i + 1, n_rooms):
            is_adjacent = (i, j) in adjacency
            edges.append([i, 1 if is_adjacent else -1, j])
    
    return np.array(edges) if edges else np.zeros((0, 3), dtype=np.int64)


if __name__ == "__main__":
    # Test template selection
    selector = TemplateSelector()
    
    print(f"Found {len(selector.templates)} templates\n")
    
    # Show all templates
    for t in selector.templates:
        print(f"  {Path(t.path).name}: {t.num_bedrooms}BR/{t.num_bathrooms}BA, "
              f"balcony={t.has_balcony}, rooms={t.total_rooms}")
    
    # Test matching
    requirements = {
        "num_bedrooms": 2,
        "num_bathrooms": 1,
        "has_balcony": True
    }
    
    best = selector.find_best_match(requirements)
    if best:
        print(f"\nBest match for 2BR/1BA with balcony:")
        print(f"  File: {Path(best.path).name}")
        print(f"  Score: {best.score}")
        print(f"  Rooms: {best.num_bedrooms}BR/{best.num_bathrooms}BA, balcony={best.has_balcony}")
