"""
House-GAN++ Inference Module V2
================================

Properly implements the House-GAN++ inference pipeline matching the original paper.
Uses the exact same data format and rendering as the original code.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import webcolors
import json

from .models import Generator, load_generator
from .requirements_converter import RequirementsConverter
from .template_selector import TemplateSelector, create_edges_from_ed_rm
from pathlib import Path


# Room class mapping (matching original House-GAN++)
ROOM_CLASS = {
    "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, 
    "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
    "storage": 10, "front door": 15, "unknown": 16, "interior_door": 17
}

CLASS_ROOM = {v: k for k, v in ROOM_CLASS.items()}

# Color mapping (matching original)
ID_COLOR = {
    1: '#EE4D4D',   # living_room - red
    2: '#C67C7B',   # kitchen - brown/salmon
    3: '#FFD274',   # bedroom - yellow
    4: '#BEBEBE',   # bathroom - gray
    5: '#BFE3E8',   # balcony - light blue
    6: '#7BA779',   # entrance - green
    7: '#E87A90',   # dining room - pink
    8: '#FF8C69',   # study room - orange
    10: '#1F849B', # storage - teal
    15: '#727171', # front door - dark gray
    16: '#785A67', # unknown - mauve
    17: '#D3A2C7'  # interior_door - light purple
}

# Room names for display
ROOM_NAMES = {
    1: 'Living Room', 2: 'Kitchen', 3: 'Bedroom', 4: 'Bathroom',
    5: 'Balcony', 6: 'Entrance', 7: 'Dining Room', 8: 'Study',
    10: 'Storage', 15: 'Front Door', 17: 'Door'
}


def one_hot_embedding(labels: np.ndarray, num_classes: int = 19) -> np.ndarray:
    """
    Embedding labels to one-hot form (matching original House-GAN++).
    
    Args:
        labels: Class labels array
        num_classes: Number of classes (19 in House-GAN++)
        
    Returns:
        One-hot encoded array [N, num_classes-1] (column 0 is removed)
    """
    y = np.eye(num_classes)
    result = y[labels]
    return result[:, 1:]  # Remove first column (index 0)


def fix_nodes(prev_mks: torch.Tensor, ind_fixed_nodes: List[int]) -> torch.Tensor:
    """
    Prepare mask input with fixed/non-fixed node indicators.
    Exactly matches original House-GAN++ implementation.
    
    Args:
        prev_mks: Previous masks [N, 64, 64]
        ind_fixed_nodes: Indices of fixed nodes
        
    Returns:
        Prepared masks [N, 2, 64, 64] with indicator channel
    """
    given_masks = prev_mks.clone()
    
    # Set non-fixed masks to -1.0
    all_nodes = set(range(len(given_masks)))
    ind_not_fixed = list(all_nodes - set(ind_fixed_nodes))
    
    if ind_not_fixed:
        given_masks[ind_not_fixed] = -1.0
    
    given_masks = given_masks.unsqueeze(1)
    
    # Add channel to indicate given nodes
    inds_masks = torch.zeros_like(given_masks)
    if ind_fixed_nodes is not None and len(ind_fixed_nodes) > 0:
        inds_masks[list(ind_fixed_nodes)] = 1.0
    
    given_masks = torch.cat([given_masks, inds_masks], 1)
    return given_masks


def draw_masks(masks: np.ndarray, real_nodes: np.ndarray, im_size: int = 256) -> Image.Image:
    """
    Draw room masks to image - EXACT copy of original House-GAN++ implementation.
    
    Args:
        masks: Room masks [N, 64, 64] with values in [-1, 1]
        real_nodes: Room type indices (0-based, where 0=living, 1=kitchen, etc.)
        im_size: Output image size
        
    Returns:
        PIL Image with rendered floor plan
    """
    bg_img = Image.new("RGBA", (im_size, im_size), (255, 255, 255, 255))
    
    for m, nd in zip(masks, real_nodes):
        # Threshold mask
        m = m.copy()
        m[m > 0] = 255
        m[m < 0] = 0
        
        # Resize to output size
        m_lg = cv2.resize(m.astype(np.float32), (im_size, im_size), 
                          interpolation=cv2.INTER_AREA)
        
        # Get color for room type (nd is 0-based index, add 1 for ID_COLOR key)
        color = ID_COLOR.get(nd + 1, '#808080')
        r, g, b = webcolors.hex_to_rgb(color)
        
        # Draw filled region
        dr_bkg = ImageDraw.Draw(bg_img)
        m_pil = Image.fromarray(m_lg.astype(np.uint8))
        dr_bkg.bitmap((0, 0), m_pil.convert('L'), fill=(r, g, b, 255))
        
        # Draw contour (walls)
        m_cv = m_lg[:, :, np.newaxis].astype(np.uint8)
        ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = np.zeros((im_size, im_size, 3), dtype=np.uint8)
            # Original uses white (255, 255, 255) but fills with black
            cv2.drawContours(cnt, contours, -1, (255, 255, 255), 1)
            cnt_pil = Image.fromarray(cnt)
            dr_bkg.bitmap((0, 0), cnt_pil.convert('L'), fill=(0, 0, 0, 255))
    
    return bg_img


def draw_masks_with_legend(masks: np.ndarray, real_nodes: np.ndarray, 
                           room_names: List[str], im_size: int = 512) -> Image.Image:
    """
    Draw room masks with a legend panel.
    
    Args:
        masks: Room masks [N, 64, 64]
        real_nodes: Room type indices (0-based)
        room_names: Display names for rooms
        im_size: Output floor plan size
        
    Returns:
        PIL Image with floor plan and legend
    """
    # Filter out door types for rendering
    DOOR_TYPES = {14, 16}  # 0-based indices for front_door (15-1) and interior_door (17-1)
    
    # Draw main floor plan
    fp_img = draw_masks(masks, real_nodes, im_size)
    
    # Create final image with legend
    legend_width = 200
    total_width = im_size + legend_width
    
    final_img = Image.new("RGBA", (total_width, im_size), (255, 255, 255, 255))
    final_img.paste(fp_img, (0, 0))
    
    # Draw legend
    dr = ImageDraw.Draw(final_img)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        item_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        item_font = ImageFont.load_default()
    
    # Legend position
    x_start = im_size + 15
    y_start = 20
    
    # Title
    dr.text((x_start, y_start), "Room Legend", fill=(0, 0, 0), font=title_font)
    dr.line([(x_start, y_start + 25), (x_start + legend_width - 30, y_start + 25)], 
            fill=(100, 100, 100), width=2)
    
    # Legend items
    y = y_start + 40
    box_size = 20
    spacing = 28
    
    # Track shown rooms to avoid duplicates
    shown = set()
    
    for nd, name in zip(real_nodes, room_names):
        # Skip doors in legend
        if nd in DOOR_TYPES:
            continue
            
        # Skip if already shown
        if name in shown:
            continue
        shown.add(name)
        
        # Get color (nd is 0-based, ID_COLOR uses 1-based keys)
        color = ID_COLOR.get(nd + 1, '#808080')
        r, g, b = webcolors.hex_to_rgb(color)
        
        # Color box
        dr.rectangle([x_start, y, x_start + box_size, y + box_size], 
                    fill=(r, g, b), outline=(50, 50, 50))
        
        # Room name
        display_name = name if len(name) <= 18 else name[:16] + '..'
        dr.text((x_start + box_size + 10, y + 3), display_name, fill=(0, 0, 0), font=item_font)
        
        y += spacing
        
        if y > im_size - 40:
            break
    
    return final_img


class HouseGANPPInferenceV2:
    """
    Proper House-GAN++ inference following the original paper implementation.
    """
    
    def __init__(self, checkpoint_path: str = None, device: str = 'auto'):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to pretrained.pth
            device: 'cuda', 'cpu', or 'auto'
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_generator(checkpoint_path, self.device)
        self.converter = RequirementsConverter()
        self.mask_size = 64
        
    def _init_input(self, graph: Tuple, prev_state: Dict = None):
        """
        Initialize input tensors - matches original _init_input function.
        
        Args:
            graph: Tuple of (nodes_onehot, edges)
            prev_state: Dict with 'masks' and 'fixed_nodes'
            
        Returns:
            z, given_masks_in, given_nds, given_eds
        """
        given_nds, given_eds = graph
        given_nds = torch.FloatTensor(given_nds) if not isinstance(given_nds, torch.Tensor) else given_nds.float()
        given_eds = torch.LongTensor(given_eds) if not isinstance(given_eds, torch.Tensor) else given_eds.long()
        
        n_nodes = len(given_nds)
        z = torch.randn(n_nodes, 128).float()
        
        # Unpack previous state
        fixed_nodes = prev_state.get('fixed_nodes', []) if prev_state else []
        prev_mks = prev_state.get('masks', None) if prev_state else None
        
        if prev_mks is None:
            prev_mks = torch.zeros((n_nodes, self.mask_size, self.mask_size)) - 1.0
        else:
            prev_mks = torch.tensor(prev_mks).float()
        
        # Fix nodes and prepare input
        given_masks_in = fix_nodes(prev_mks, fixed_nodes)
        
        return z, given_masks_in, given_nds, given_eds
    
    def _infer(self, graph: Tuple, prev_state: Dict = None) -> np.ndarray:
        """
        Run single inference step - matches original _infer function.
        
        Args:
            graph: Tuple of (nodes_onehot, edges)
            prev_state: Dict with 'masks' and 'fixed_nodes'
            
        Returns:
            Generated masks [N, 64, 64]
        """
        z, given_masks_in, given_nds, given_eds = self._init_input(graph, prev_state)
        
        with torch.no_grad():
            masks = self.model(
                z.to(self.device),
                given_masks_in.to(self.device),
                given_nds.to(self.device),
                given_eds.to(self.device)
            )
            masks = masks.detach().cpu().numpy()
        
        return masks
    
    def generate_from_graph(self, nodes: np.ndarray, edges: np.ndarray,
                           room_names: List[str] = None,
                           num_iterations: int = 5,
                           seed: int = None) -> Dict:
        """
        Generate floor plan from graph specification.
        
        This follows the exact same iterative refinement process as the original
        House-GAN++ test.py.
        
        Args:
            nodes: Room type IDs (e.g., [1, 2, 3, 4, ...])
            edges: Edge triples [src, relation, dst]
            room_names: Optional display names
            num_iterations: Additional refinement iterations
            seed: Random seed
            
        Returns:
            Dict with 'image', 'masks', 'nodes', 'room_names'
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Convert node types to one-hot encoding (matching original)
        # nodes are room type IDs (1-based), one_hot_embedding expects these directly
        nodes_onehot = one_hot_embedding(np.array(nodes))
        
        # Get "real_nodes" - the room type indices (0-based, used for coloring)
        # This is what np.where(nds==1)[-1] returns in the original code
        real_nodes = np.array(nodes) - 1  # Convert 1-based IDs to 0-based indices
        
        # Prepare graph
        graph = (nodes_onehot, edges)
        
        # Get unique room types (excluding doors 15, 17)
        room_types_only = [t for t in nodes if t not in {15, 17}]
        unique_types = sorted(list(set(room_types_only)))
        
        # Build selection schedule (add rooms type by type)
        selected_types = [unique_types[:k+1] for k in range(min(10, len(unique_types)))]
        
        # Step 1: Initial generation (no fixed nodes)
        state = {'masks': None, 'fixed_nodes': []}
        masks = self._infer(graph, state)
        
        # Step 2: Iterative refinement by room type (key to House-GAN++)
        for _types in selected_types:
            # Find indices of rooms with these types
            _fixed_nds = []
            for _t in _types:
                indices = np.where(np.array(nodes) == _t)[0]
                _fixed_nds.extend(indices.tolist())
            _fixed_nds = np.array(_fixed_nds) if _fixed_nds else np.array([])
            
            state = {'masks': masks, 'fixed_nodes': _fixed_nds.tolist()}
            masks = self._infer(graph, state)
        
        # Step 3: Additional refinement passes
        all_room_indices = [i for i, t in enumerate(nodes) if t not in {15, 17}]
        for _ in range(num_iterations):
            state = {'masks': masks, 'fixed_nodes': all_room_indices}
            masks = self._infer(graph, state)
        
        # Generate room names if not provided
        if room_names is None:
            room_names = []
            bedroom_count = 0
            bathroom_count = 0
            for nd in nodes:
                if nd == 3:
                    bedroom_count += 1
                    name = 'Master Bedroom' if bedroom_count == 1 else f'Bedroom {bedroom_count}'
                elif nd == 4:
                    bathroom_count += 1
                    name = 'Attached Bath' if bathroom_count == 1 else f'Bathroom {bathroom_count}'
                elif nd in {15, 17}:
                    name = 'Door'
                else:
                    name = ROOM_NAMES.get(nd, f'Room {nd}')
                room_names.append(name)
        
        # Render image
        image = draw_masks_with_legend(masks, real_nodes, room_names)
        
        return {
            'image': image,
            'masks': masks,
            'nodes': np.array(nodes),
            'edges': edges,
            'room_names': room_names,
            'real_nodes': real_nodes,  # 0-based indices for coloring
        }
    
    def generate_from_requirements(self, requirements: Dict,
                                   num_iterations: int = 5,
                                   seed: int = None,
                                   use_template: bool = True) -> Dict:
        """
        Generate floor plan from requirements dict.
        
        Uses template-based generation by default for best results, as the model
        performs best with graph structures it was trained on.
        
        Args:
            requirements: Dict with num_bedrooms, num_bathrooms, etc.
            num_iterations: Refinement iterations
            seed: Random seed
            use_template: If True (default), find best matching template
            
        Returns:
            Dict with 'image', 'masks', 'nodes', 'room_names'
        """
        if use_template:
            # Try template-based generation first (much better results)
            try:
                selector = TemplateSelector()
                best_match = selector.find_best_match(requirements)
                
                if best_match is not None:
                    print(f"Using template: {Path(best_match.path).name} "
                          f"({best_match.num_bedrooms}BR/{best_match.num_bathrooms}BA)")
                    return self.generate_from_json_file(
                        best_match.path,
                        num_iterations=num_iterations,
                        seed=seed
                    )
            except Exception as e:
                print(f"Template selection failed: {e}, falling back to direct generation")
        
        # Fallback: Convert requirements to graph directly
        nodes, edges, room_names = self.converter.convert(requirements)
        
        return self.generate_from_graph(
            nodes=nodes,
            edges=edges,
            room_names=room_names,
            num_iterations=num_iterations,
            seed=seed
        )
    
    def generate_from_json_file(self, json_path: str,
                                num_iterations: int = 5,
                                seed: int = None) -> Dict:
        """
        Generate floor plan from a House-GAN++ JSON template file.
        
        These JSON files contain pre-defined layouts with proper room
        configurations that are known to work well with the model.
        
        Args:
            json_path: Path to JSON file
            num_iterations: Refinement iterations
            seed: Random seed
            
        Returns:
            Dict with 'image', 'masks', 'nodes', 'room_names'
        """
        with open(json_path) as f:
            data = json.load(f)
        
        nodes = np.array(data['room_type'])
        
        # Build edges from template
        # The template has detailed edge info, but we need simple triples
        n_rooms = len(nodes)
        edges = []
        
        # Check if 'ed_rm' exists for adjacency info
        if 'ed_rm' in data:
            ed_rm = data['ed_rm']
            for i in range(n_rooms):
                for j in range(i + 1, n_rooms):
                    # Check if rooms share an edge
                    is_adjacent = any(
                        (i in edge_rooms and j in edge_rooms) 
                        for edge_rooms in ed_rm
                    )
                    relation = 1 if is_adjacent else -1
                    edges.append([i, relation, j])
        else:
            # Fallback: connect all rooms as adjacent
            for i in range(n_rooms):
                for j in range(i + 1, n_rooms):
                    edges.append([i, 1, j])
        
        edges = np.array(edges)
        
        return self.generate_from_graph(
            nodes=nodes.tolist(),
            edges=edges,
            num_iterations=num_iterations,
            seed=seed
        )
    
    def save_result(self, result: Dict, output_dir: str, 
                    filename: str = 'floorplan') -> Dict[str, str]:
        """Save generation result to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save image
        img_path = output_path / f'{filename}.png'
        result['image'].save(str(img_path))
        paths['image'] = str(img_path)
        
        # Save masks
        masks_path = output_path / f'{filename}_masks.npy'
        np.save(str(masks_path), result['masks'])
        paths['masks'] = str(masks_path)
        
        # Save metadata
        meta = {
            'room_names': result['room_names'],
            'nodes': result['nodes'].tolist(),
            'edges': result['edges'].tolist() if len(result['edges']) > 0 else []
        }
        meta_path = output_path / f'{filename}_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        paths['metadata'] = str(meta_path)
        
        return paths


# Convenience function
def generate_floorplan(requirements: Dict = None,
                       json_path: str = None,
                       output_dir: str = 'generated_floorplans',
                       seed: int = None) -> Tuple[Image.Image, str]:
    """
    Generate a floor plan.
    
    Args:
        requirements: Requirements dict (num_bedrooms, etc.)
        json_path: Path to House-GAN++ JSON template
        output_dir: Output directory
        seed: Random seed
        
    Returns:
        (image, output_path)
    """
    inference = HouseGANPPInferenceV2()
    
    if json_path:
        result = inference.generate_from_json_file(json_path, seed=seed)
    elif requirements:
        result = inference.generate_from_requirements(requirements, seed=seed)
    else:
        raise ValueError("Provide either requirements or json_path")
    
    # Save
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = inference.save_result(result, output_dir, f'floorplan_{timestamp}')
    
    return result['image'], paths['image']


if __name__ == "__main__":
    # Test with a template file
    print("Testing House-GAN++ Inference V2...")
    
    inference = HouseGANPPInferenceV2()
    
    # Test with template
    result = inference.generate_from_json_file(
        'houseganpp/data/json/18477.json',
        seed=42
    )
    result['image'].save('test_v2_template.png')
    print(f"✅ Template test saved to test_v2_template.png")
    
    # Test with requirements
    requirements = {
        "num_bedrooms": 2,
        "num_bathrooms": 1,
        "has_balcony": True,
        "combined_living_dining": True
    }
    result = inference.generate_from_requirements(requirements, seed=42)
    result['image'].save('test_v2_requirements.png')
    print(f"✅ Requirements test saved to test_v2_requirements.png")
