# Agentic AI Floor Plan Generator - Requirements & Implementation Guide

## Project Overview
An agentic AI system that interacts with customers through a chatbot to gather bungalow requirements, generates custom floor plans using House-GAN, and creates 3D visualizations of the proposed structure.

---

## ✅ Feasibility Assessment

### Can this be implemented in Jupyter Notebook (.ipynb)?
**YES!** This project is perfectly suited for Jupyter notebooks because:
- Interactive chatbot can be implemented using `ipywidgets` for UI
- All ML models (Gemini API, House-GAN) can be integrated in notebooks
- Google Colab provides free GPU for model inference
- Visualization libraries work seamlessly in notebooks
- Progressive development and testing is easier

### Google Colab Compatibility
**FULLY COMPATIBLE** - The implementation will support:
- Free tier GPU/TPU runtime
- Persistent storage via Google Drive
- API key management through Colab secrets
- Interactive widgets for chat interface

---

## 📋 Step-by-Step Implementation Roadmap

### **Phase 1: Conversational Agent (Requirements Gathering)**
**Goal:** Create an intelligent chatbot that extracts all necessary bungalow specifications

#### Technologies Required:
- **Google Gemini API** (Free tier: 15 requests/min)
- **ipywidgets** for interactive chat UI
- **Python libraries:** google-generativeai, json, typing

#### Implementation Steps:
1. **Set up Gemini API integration**
   - Configure API key (user-provided)
   - Initialize Gemini model (gemini-pro)
   - Create conversational context management

2. **Design conversation flow**
   - Define structured prompt templates
   - Implement multi-turn dialogue system
   - Extract and validate user requirements

3. **Information to Extract:**
   - **Basic Layout:**
     - Type of bungalow (single-story, duplex, villa)
     - Total land area (in sq ft / sq meters)
     - Building coverage ratio preference
   
   - **Room Requirements:**
     - Number of bedrooms (with sizes)
     - Number of bathrooms
     - Living room (yes/no + size preference)
     - Kitchen type (open/closed)
     - Dining room (separate/combined)
     - Additional rooms: study, prayer room, guest room
   
   - **Utility Spaces:**
     - Parking (2-wheeler/4-wheeler capacity)
     - Balconies/Terraces
     - Store room/utility room
     - Laundry area
   
   - **Layout Preferences:**
     - Open floor plan vs traditional
     - Room adjacencies (e.g., master bedroom near bathroom)
     - Natural light preferences
     - Entrance facing direction
   
   - **Special Requirements:**
     - Accessibility features
     - Outdoor spaces (garden, patio)
     - Future expansion possibilities

4. **Output Format:**
   - Structured JSON with validated requirements
   - Summary display for user confirmation
   - Edit/refine capability before generation

#### Estimated Complexity: **Medium**
#### Estimated Development Time: **2-3 days**

---

### **Phase 2: Floor Plan Generation**
**Goal:** Generate custom floor plans using House-GAN based on extracted requirements

#### Technologies Required:
- **House-GAN Model** (already in workspace: `exp_demo_D_500000.pth`)
- **PyTorch** for model inference
- **NumPy** for data processing
- **Pillow/OpenCV** for image generation

#### Implementation Steps:
1. **Prepare House-GAN environment**
   - Load pre-trained model weights (`exp_demo_D_500000.pth`)
   - Set up graph structure for room relationships
   - Configure room type mapping (ROOM_CLASS from dataset)

2. **Translate requirements to graph structure**
   - Convert user requirements to room nodes
   - Define room adjacency constraints
   - Set room size constraints (bounding boxes)
   - Add door/corridor connections

3. **Generate floor plan**
   - Feed graph structure to House-GAN
   - Generate multiple variations (3-5 options)
   - Post-process output for clarity
   - Add annotations (room labels, dimensions)

4. **Validate generated plan**
   - Check if all requirements are met
   - Verify spatial constraints
   - Calculate total area coverage
   - Present to user for selection

5. **Export options:**
   - Vector format (SVG/PDF) for editing
   - Raster image (PNG) for visualization
   - Structured data (JSON) for 3D generation

#### Required Model Architecture Components:
```python
- Graph Neural Network (GNN) for room relationships
- Generator network from House-GAN
- Room bounding box predictor
- Edge/wall generator
- Door placement module
```

#### Dataset Utilization:
- **Training data:** `housegan_clean_data.npy` (145,811 floor plans)
- **Pre-trained weights:** `exp_demo_D_500000.pth`
- **CubiCasa5K:** Optional additional training data

#### Estimated Complexity: **High**
#### Estimated Development Time: **5-7 days**

---

### **Phase 3: 3D Visualization & Rendering**
**Goal:** Generate realistic 3D renders of the bungalow exterior/interior

#### Technologies Required:
- **Option A - Simple:** matplotlib/plotly for 2.5D isometric views
- **Option B - Advanced:** Blender Python API or Three.js for full 3D
- **Option C - AI-based:** Stable Diffusion/ControlNet for photorealistic renders

#### Implementation Steps (Recommended: Hybrid Approach):

**Step 1: Basic 3D Structure Generation**
- Use matplotlib/plotly to create 3D wireframe from floor plan
- Extrude walls to standard height (10-12 ft)
- Add roof structure (sloped/flat based on bungalow type)
- Color-code rooms by type

**Step 2: AI-Powered Realistic Rendering** (Recommended)
- Use **Stable Diffusion + ControlNet** to generate photorealistic images
- Input: Floor plan sketch + text prompt (from user requirements)
- Output: Multiple architectural renders (front elevation, 3D view)
- Models to use:
  - ControlNet with canny edge detection
  - Architectural fine-tuned SD models

**Step 3: Interactive Visualization**
- Create interactive 3D viewer in notebook
- Allow rotation, zoom, pan
- Toggle layers (walls, furniture, dimensions)

#### Alternative Approaches:
1. **Text-to-3D Models:**
   - DreamFusion, Point-E (requires GPU)
   - Limited architectural control

2. **Pre-rendered Templates:**
   - Match generated floor plan to template library
   - Apply textures and materials
   - Fastest but less customizable

3. **Parametric 3D Modeling:**
   - Generate Blender script from floor plan
   - Automatic scene setup and rendering
   - Best quality but complex setup

#### Estimated Complexity: **High**
#### Estimated Development Time: **4-6 days**

---

## 🛠️ Technical Requirements

### Python Libraries (pip install)
```python
# Core ML & AI
google-generativeai>=0.3.0    # Gemini API
torch>=2.0.0                   # PyTorch for House-GAN
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0

# Computer Vision & Graphics
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
plotly>=5.17.0
scikit-image>=0.21.0

# 3D Rendering (Optional)
trimesh>=4.0.0                # 3D mesh processing
pyrender>=0.1.45              # 3D rendering
open3d>=0.17.0                # Point cloud & 3D visualization

# Image Generation (Optional)
diffusers>=0.25.0             # Stable Diffusion
transformers>=4.35.0          # For ControlNet
accelerate>=0.25.0            # Inference optimization

# UI & Interaction
ipywidgets>=8.1.0             # Interactive widgets
ipython>=8.12.0
gradio>=4.0.0                 # Alternative UI framework

# Utilities
requests>=2.31.0
tqdm>=4.66.0
pyyaml>=6.0.0
jsonschema>=4.19.0            # Validate requirement JSON
```

### System Requirements

**Minimum (for Development):**
- Python 3.8+
- 8GB RAM
- 5GB storage
- CPU-only mode (slower)

**Recommended (for Production):**
- Python 3.10+
- 16GB+ RAM
- 10GB storage
- GPU with 6GB+ VRAM (for faster inference)

**Google Colab (Free Tier):**
- ✅ T4 GPU (15GB VRAM)
- ✅ 12GB RAM
- ✅ 100GB temporary storage
- ✅ Perfect for this project!

---

## 📦 Project Structure

```
AgenticAI/
├── notebooks/
│   ├── 01_chatbot_agent.ipynb           # Phase 1: Conversational agent
│   ├── 02_floorplan_generator.ipynb     # Phase 2: House-GAN integration
│   ├── 03_3d_visualizer.ipynb           # Phase 3: 3D rendering
│   └── 04_full_pipeline.ipynb           # Complete integrated system
│
├── src/                                  # Reusable Python modules
│   ├── agent/
│   │   ├── gemini_agent.py              # Gemini API wrapper
│   │   ├── conversation_manager.py      # Dialogue flow control
│   │   └── requirement_extractor.py     # Parse user inputs
│   │
│   ├── generation/
│   │   ├── housegan_model.py            # House-GAN wrapper
│   │   ├── graph_builder.py             # Convert requirements → graph
│   │   ├── floorplan_generator.py       # Main generation logic
│   │   └── postprocessor.py             # Clean up outputs
│   │
│   ├── visualization/
│   │   ├── render_2d.py                 # 2D floor plan rendering
│   │   ├── render_3d.py                 # 3D structure generation
│   │   └── ai_render.py                 # Stable Diffusion integration
│   │
│   └── utils/
│       ├── config.py                    # Configuration management
│       ├── validators.py                # Input validation
│       └── file_handlers.py             # Save/load utilities
│
├── data/
│   ├── 2018-house_gan/                  # Existing House-GAN data
│   ├── generated_plans/                 # User-generated floor plans
│   └── user_sessions/                   # Conversation history
│
├── models/
│   ├── housegan_weights/                # Pre-trained models
│   └── controlnet_weights/              # (Optional) For rendering
│
├── configs/
│   ├── room_templates.json              # Standard room configurations
│   ├── style_presets.json               # Architectural styles
│   └── generation_params.yaml           # Model hyperparameters
│
├── REQUIREMENTS.md                       # This file
├── README.md                             # Project documentation
├── environment.yml                       # Conda environment (optional)
└── requirements.txt                      # Pip dependencies
```

---

## 🎯 Implementation Priority & Timeline

### Week 1: Foundation
- ✅ Set up development environment
- ✅ Integrate Gemini API with basic chat
- ✅ Design requirement extraction schema
- ✅ Create conversational prompts

### Week 2: Core Generation
- ✅ Load and test House-GAN model
- ✅ Implement requirement → graph conversion
- ✅ Generate basic floor plans
- ✅ Add validation and refinement

### Week 3: Visualization
- ✅ Implement 2D floor plan rendering
- ✅ Create basic 3D structure extrusion
- ✅ (Optional) Integrate AI rendering

### Week 4: Integration & Polish
- ✅ Combine all phases into single notebook
- ✅ Add error handling and edge cases
- ✅ Optimize for Colab deployment
- ✅ Create user documentation

---

## 🔑 Configuration Required

### API Keys & Credentials
```python
# Store in Google Colab Secrets or .env file
GEMINI_API_KEY = "your-gemini-api-key-here"

# Optional (for advanced features)
HUGGINGFACE_TOKEN = "your-hf-token"  # For Stable Diffusion models
```

### Model Weights
- **House-GAN:** Already available at `2018-house_gan/exp_demo_D_500000.pth`
- **Training Data:** Available at `2018-house_gan/housegan_clean_data.npy`

---

## 🚀 Getting Started

### Step 1: Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download additional models (if needed)
# House-GAN weights already present
```

### Step 2: Configure API
```python
# In notebook or Colab secrets
import google.generativeai as genai
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

### Step 3: Run Notebooks Sequentially
1. Start with `01_chatbot_agent.ipynb` - Test conversation flow
2. Move to `02_floorplan_generator.ipynb` - Generate sample plans
3. Test `03_3d_visualizer.ipynb` - Visualize outputs
4. Use `04_full_pipeline.ipynb` - Complete end-to-end workflow

---

## 🎨 User Experience Flow

```
1. User opens notebook/Colab
   ↓
2. Chatbot welcomes user
   ↓
3. Conversational requirement gathering (5-10 exchanges)
   ↓
4. System displays extracted requirements for confirmation
   ↓
5. User approves/edits requirements
   ↓
6. System generates 3-5 floor plan variations
   ↓
7. User selects preferred design
   ↓
8. System generates 3D visualization
   ↓
9. User can download:
   - Floor plan (PNG/SVG/PDF)
   - 3D render (PNG/JPG)
   - Requirement summary (JSON/TXT)
   ↓
10. Option to refine or start new design
```

---

## ⚠️ Challenges & Solutions

### Challenge 1: House-GAN Model Adaptation
**Issue:** Pre-trained model may not perfectly match user requirements  
**Solution:** 
- Fine-tune on specific bungalow layouts
- Use post-processing to adjust room sizes
- Implement constraint-based refinement

### Challenge 2: Gemini API Rate Limits
**Issue:** Free tier has 15 requests/minute limit  
**Solution:**
- Implement request throttling
- Cache conversation context
- Optimize prompts to reduce API calls

### Challenge 3: 3D Rendering Complexity
**Issue:** High-quality 3D rendering is computationally expensive  
**Solution:**
- Prioritize 2.5D visualization (faster)
- Use pre-rendered templates for common layouts
- Offer AI-generated images as alternative

### Challenge 4: Requirement Ambiguity
**Issue:** Users may provide vague specifications  
**Solution:**
- Use Gemini to ask clarifying questions
- Provide default/suggested values
- Show examples during conversation

---

## 🔍 Testing Strategy

### Phase 1 Testing (Chatbot)
- [ ] Test with various user input styles (detailed/brief)
- [ ] Validate JSON schema for all edge cases
- [ ] Ensure graceful handling of unclear inputs

### Phase 2 Testing (Generation)
- [ ] Generate plans for 10+ diverse requirements
- [ ] Validate architectural feasibility
- [ ] Check compliance with spatial constraints

### Phase 3 Testing (Visualization)
- [ ] Verify 3D outputs match floor plans
- [ ] Test rendering on different GPU/CPU configs
- [ ] Ensure Colab compatibility

---

## 📚 Learning Resources

### House-GAN
- Paper: "House-GAN: Relational Generative Adversarial Networks"
- Dataset structure explained in `2018-house_gan/README.txt`

### Gemini API
- Google AI Studio: https://makersuite.google.com/
- Documentation: https://ai.google.dev/docs

### 3D Visualization
- Matplotlib 3D: https://matplotlib.org/stable/gallery/mplot3d/
- Plotly 3D: https://plotly.com/python/3d-charts/

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- ✅ Functional chatbot that extracts requirements
- ✅ Generate valid floor plan from requirements
- ✅ Basic 2D visualization of floor plan
- ✅ Runs on Google Colab without errors

### Full Featured Product
- ✅ Natural conversational flow with clarifications
- ✅ Multiple floor plan variations generated
- ✅ High-quality 3D renders or AI-generated images
- ✅ Export functionality (multiple formats)
- ✅ User can refine and regenerate designs
- ✅ Professional UI/UX in notebook

---

## 💡 Future Enhancements (Post-MVP)

1. **Cost Estimation Module**
   - Calculate construction cost based on local rates
   - Material quantity estimation

2. **Furniture Layout**
   - Auto-place furniture in generated floor plans
   - Multiple style options (modern, traditional)

3. **Building Code Compliance**
   - Check against local building regulations
   - Suggest modifications for compliance

4. **Multi-Story Support**
   - Generate plans for 2-3 story buildings
   - Staircase placement optimization

5. **VR/AR Integration**
   - Export to VR-compatible formats
   - AR preview on mobile devices

6. **Collaborative Features**
   - Share designs via link
   - Comments and revision tracking

---

## 📝 Notes for Developer

- **Start Simple:** Build Phase 1 completely before moving to Phase 2
- **Test Frequently:** Validate each component in isolation
- **Document Code:** Add clear comments for notebook users
- **Optimize Later:** Focus on functionality first, performance second
- **User Feedback:** Add checkpoints where user can validate/modify

---

## ✅ Pre-Implementation Checklist

- [ ] Obtain Gemini API key from user
- [ ] Verify House-GAN model weights are accessible
- [ ] Test Google Colab environment setup
- [ ] Install all required Python packages
- [ ] Create initial notebook structure
- [ ] Set up Git repository (optional but recommended)

---

**Total Estimated Development Time:** 3-4 weeks (full-time)  
**Estimated Cost:** $0 (using free tier services)  
**Primary Development Environment:** VS Code + Jupyter + Google Colab

