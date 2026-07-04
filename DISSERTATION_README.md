# Agentic AI for Customer-Centric Architectural Design Assistance

## 📋 Project Overview

This dissertation project develops an **Agentic AI system** that assists users in generating architectural floor plans through natural language conversation. The system combines a conversational AI chatbot with a deep learning-based floor plan generator (House-GAN++) to create personalized home layouts based on user requirements.

### Problem Statement
Traditional architectural design requires expensive professional consultations and multiple iterations. This project democratizes floor plan design by allowing users to describe their needs in plain English and receive AI-generated layouts instantly.

### Solution
An end-to-end pipeline where:
1. Users chat with an AI agent to describe their requirements
2. The agent extracts structured specifications automatically
3. A neural network generates multiple floor plan variations
4. The system auto-selects the best layout based on optimization criteria

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Conversational AI** | Google Gemini 2.0 Flash | Natural language understanding & requirement extraction |
| **Floor Plan Generator** | House-GAN++ (CVPR 2021) | Deep learning-based layout generation |
| **Deep Learning Framework** | PyTorch | Neural network operations |
| **Graph Processing** | NetworkX | Room adjacency graph visualization |
| **Image Processing** | OpenCV, Pillow | Floor plan rendering with room colors |
| **Development Environment** | Python 3.8+, Jupyter Notebook | Implementation platform |
| **API Integration** | Google Generative AI SDK | Chatbot communication |

### Key Libraries
- `torch`, `torchvision` - Neural network operations
- `google-generativeai` - Gemini API integration
- `networkx` - Graph-based room relationships
- `matplotlib` - Visualization
- `numpy`, `opencv-python`, `pillow` - Image processing

---

## 🎯 Desired Outcomes

1. **Natural Language Interface**: Users describe floor plan needs conversationally
2. **Automated Requirement Extraction**: AI converts conversation to structured JSON
3. **Multiple Layout Variations**: Generate 6+ different layouts per request
4. **Intelligent Selection**: Auto-select best layout based on room balance
5. **Professional Output**: High-quality rendered floor plans with room labels
6. **Complete Session Export**: Save images, metadata, and conversation history

---

## 🔄 Project Flow Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERACTION LAYER                        │
│  User describes: "I need 2 bedrooms, 2 bathrooms, and a balcony"│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 1: AI CHATBOT AGENT                        │
│  • Gemini 2.0 Flash processes natural language                  │
│  • Multi-turn conversation to gather requirements               │
│  • Extracts: bedrooms, bathrooms, balcony, study, dining, etc.  │
│  • Outputs: Structured JSON requirements                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2: GRAPH CONSTRUCTION                      │
│  • RequirementsConverter transforms JSON to graph format        │
│  • Creates room nodes (living, kitchen, bedroom, bathroom...)   │
│  • Defines adjacency edges (which rooms connect)                │
│  • TemplateSelector finds best matching training template       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 3: HOUSE-GAN++ GENERATION                  │
│  • Neural network generator with CMP (Convolutional MP)         │
│  • Iterative refinement by room type                            │
│  • One-hot encoding (19 room classes)                           │
│  • Generates room masks on 32x32 → 256x256 canvas               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 4: VARIATION & SELECTION                   │
│  • Generate 6 variations with different random seeds            │
│  • Score each based on room coverage and balance                │
│  • Auto-select best layout                                      │
│  • Render with professional colors and room labels              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 5: EXPORT & DOCUMENTATION                  │
│  • Save PNG floor plan images                                   │
│  • Export room masks (NumPy arrays)                             │
│  • Store session metadata (JSON)                                │
│  • Archive conversation history                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Why This Project is "Agentic AI"

### Definition of Agentic AI
Agentic AI systems autonomously perform tasks, make decisions, and take actions with minimal human intervention.

### Agentic Characteristics in This Project

| Characteristic | Implementation |
|----------------|----------------|
| **Autonomy** | System independently extracts requirements, generates layouts, and selects best output |
| **Goal-Oriented** | Works toward user's floor plan goal through multi-step reasoning |
| **Adaptive** | Handles varied user inputs and adjusts to different requirement combinations |
| **Decision Making** | Auto-selects best layout from multiple variations using scoring algorithm |
| **Multi-Step Planning** | Orchestrates chatbot → graph → generation → selection → export pipeline |
| **No Hardcoded Values** | All parameters derived dynamically from user conversation |
| **Self-Contained Workflow** | End-to-end execution without manual intervention |

### Agent Workflow
```
User Input → Agent Reasoning → Action Execution → Result Evaluation → Output
```

---

## ✅ Benefits of the Project

1. **Accessibility**: No architectural expertise required from users
2. **Cost-Effective**: Eliminates need for expensive initial consultations
3. **Speed**: Generate multiple layouts in seconds vs. days with professionals
4. **Personalization**: Tailored to exact user specifications
5. **Iteration-Friendly**: Easy to regenerate with different preferences
6. **Educational**: Users learn about spatial relationships through the graph
7. **Reproducibility**: Session data allows exact recreation of results

---

## 👨‍💻 My Work in This Project

### Completed Tasks
1. **Integrated House-GAN++ (CVPR 2021)** - Replaced original House-GAN with improved version
2. **Built Custom Inference Engine** - `inference_v2.py` with proper one-hot encoding
3. **Created Requirements Converter** - Transform chatbot JSON to graph format
4. **Implemented Template Selector** - Match user requirements to best training template
5. **Developed Gemini Chatbot Agent** - Natural conversation for requirement gathering
6. **Built Graph Visualization** - NetworkX-based room adjacency display
7. **Created End-to-End Pipeline** - Single notebook for complete workflow
8. **Implemented Auto-Selection** - Scoring algorithm for best layout selection
9. **Added Variation Generation** - Multiple layouts from same requirements

### Technical Achievements
- Fixed room mask rendering to match original paper quality
- Implemented correct 19-class one-hot encoding
- Created proper iterative refinement by room type
- Built professional color-coded floor plan output

---

## 📊 Current Project Status: 50% Complete

### ✅ Completed Phases (Phase 1-2)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Done | AI Chatbot for requirement gathering |
| Phase 2 | ✅ Done | House-GAN++ floor plan generation |

### 🔄 In Progress (Phase 3)

| Task | Status |
|------|--------|
| Quality improvements | 🔄 Ongoing |
| Edge case handling | 🔄 Ongoing |
| Performance optimization | 🔄 Ongoing |

---

## 📅 Remaining Work (December 2024 - January 2025)

### Phase 3: Enhancement & Optimization (December)

| Week | Task | Details |
|------|------|---------|
| Week 1 | **Dimension Integration** | Add real-world measurements (sq.ft) to rooms |
| Week 2 | **Constraint System** | Implement architectural rules (min room sizes, adjacencies) |
| Week 3 | **User Feedback Loop** | Allow users to refine/modify generated layouts |
| Week 4 | **Testing & Validation** | Test with diverse requirement combinations |

### Phase 4: Advanced Features (January)

| Week | Task | Details |
|------|------|---------|
| Week 1 | **Multi-Floor Support** | Extend to 2-story home layouts |
| Week 2 | **Style Preferences** | Add modern/traditional/open-plan style options |
| Week 3 | **Export Formats** | Add PDF, DXF (CAD-compatible) export |
| Week 4 | **Documentation & Demo** | Final documentation, demo video, dissertation writing |

### Planned Improvements
1. **3D Visualization** - Basic 3D view of generated floor plan
2. **Comparison Mode** - Side-by-side comparison of variations
3. **Web Interface** - Streamlit/Gradio demo deployment

---

## 📁 Project Structure

```
AgenticAI/
├── notebooks/
│   └── 06_complete_floorplan_pipeline.ipynb  # Main notebook
├── houseganpp/                                # House-GAN++ model
│   ├── checkpoints/pretrained.pth            # Pretrained weights
│   ├── data/json/                            # Template files
│   └── models/                               # Model definitions
├── houseganpp_integration/                   # Custom integration
│   ├── inference_v2.py                       # Main inference engine
│   ├── models.py                             # Generator wrapper
│   ├── requirements_converter.py             # JSON to graph
│   └── template_selector.py                  # Template matching
├── .env                                      # API key configuration
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## 🔬 Research Contributions

1. **Novel Integration**: First integration of conversational AI with House-GAN++ for end-to-end floor plan generation
2. **Agentic Architecture**: Demonstrates agentic AI principles in architectural design domain
3. **Template Matching**: Improved generation quality through intelligent template selection
4. **User-Centric Design**: Natural language interface for non-technical users

---

## 📚 References

1. House-GAN++: Generative Adversarial Layout Refinement Network (CVPR 2021)
2. Google Gemini API Documentation
3. Graph Neural Networks for Architectural Layout Generation

---

