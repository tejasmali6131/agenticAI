# Project Summary: Agentic AI for Customer-Centric Architectural Design Assistance

## Brief Summary

This is an **M.Tech dissertation project** that builds an **Agentic AI system** to generate architectural floor plans from natural language conversations. Instead of hiring an expensive architect, a user simply chats with an AI agent ("I need 2 bedrooms, 2 bathrooms, and a balcony"), and the system autonomously generates professional floor plan layouts.

### How It Works (5-Phase Pipeline)

1. **Phase 1 – AI Chatbot Agent (Google Gemini 2.0 Flash)**  
   A conversational agent acts as a "virtual architectural consultant." It conducts a 5–7 turn dialogue, asks follow-up questions, and extracts structured requirements (number of bedrooms, bathrooms, balcony, study, etc.) into a JSON format.

2. **Phase 2 – Graph Construction**  
   A `RequirementsConverter` transforms the JSON into a room adjacency graph (nodes = rooms, edges = which rooms share walls/doors). A `TemplateSelector` finds the best-matching template from ~1000 pre-analyzed RPLAN dataset templates.

3. **Phase 3 – House-GAN++ Deep Learning Generation (CVPR 2021)**  
   The graph is fed into a pre-trained House-GAN++ neural network (Generator with Convolutional Message Passing). It generates room masks on a 32×32 → 256×256 canvas using iterative refinement by room type.

4. **Phase 4 – Variation Generation & Auto-Selection**  
   The system generates 6 layout variations using different random seeds, scores each based on room coverage and spatial balance, and **autonomously selects the best one** — no human intervention needed.

5. **Phase 5 – Export & Documentation**  
   Saves PNG floor plan images (color-coded with legend), NumPy mask arrays, JSON metadata, and full conversation history.

---

## What Makes It "Agentic"?

| Agentic Property | How It's Implemented |
|------------------|---------------------|
| **Autonomy** | System independently extracts requirements, builds graphs, generates layouts, and selects the best output — all without human intervention after initial chat |
| **Goal-Oriented Reasoning** | Works toward producing the user's ideal floor plan through multi-step planning |
| **Adaptive Behavior** | Handles varied user inputs, infers missing information using architectural best practices |
| **Decision Making** | Auto-selects best layout from 6 variations using a scoring algorithm (coverage + balance) |
| **Multi-Step Orchestration** | Coordinates 5 distinct phases: Chatbot → Graph → GAN → Scoring → Export |
| **No Hardcoded Values** | All parameters derived dynamically from user conversation |
| **Self-Contained Workflow** | End-to-end execution: one user conversation produces complete floor plan outputs |

---

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash (conversational AI, requirement extraction)
- **Deep Learning**: House-GAN++ (CVPR 2021), PyTorch
- **Graph Processing**: NetworkX (room adjacency visualization)
- **Image Processing**: OpenCV, Pillow, matplotlib
- **Environment**: Python 3.8+, Jupyter Notebook
- **Dataset**: RPLAN (60K+ residential floor plans)

---

## Key Code Modules

| Module | Purpose |
|--------|---------|
| `houseganpp_integration/inference_v2.py` | Main inference engine — generates floor plans, handles post-processing, best-of-N selection |
| `houseganpp_integration/requirements_converter.py` | Converts chatbot JSON → House-GAN++ graph (nodes + edges) |
| `houseganpp_integration/template_selector.py` | Finds best matching RPLAN template for user requirements |
| `houseganpp_integration/architectural_constraints.py` | Enforces real-world architectural rules (adjacency, forbidden pairs, area ratios) |
| `houseganpp_integration/models.py` | House-GAN++ Generator with CMP (Convolutional Message Passing) |
| `notebooks/06_complete_floorplan_pipeline.ipynb` | End-to-end pipeline notebook |

---

## 10 Important Questions & Answers for Presentation

### Q1: What is the problem this project solves?

**A:** Traditional architectural design is expensive ($5,000–$20,000 for residential projects), time-intensive (2–4 weeks for initial iterations), and requires specialized expertise. This project democratizes floor plan design — a non-expert user describes their needs in plain English, and the AI generates professional layouts in seconds.

---

### Q2: What makes this system "Agentic AI" and not just a regular chatbot?

**A:** Unlike a regular chatbot that only answers questions, this system **autonomously performs a complete multi-step workflow**: it conducts a conversation, extracts structured data, constructs a graph, runs a neural network, generates multiple variations, scores them, selects the best one, and exports results — all without human intervention after the initial chat. It exhibits autonomy, goal-oriented reasoning, adaptive behavior, and decision-making.

---

### Q3: What is House-GAN++ and why was it chosen?

**A:** House-GAN++ (published at CVPR 2021) is a Generative Adversarial Network specifically designed for floor plan layout generation. It uses **Convolutional Message Passing (CMP)** to propagate spatial information between adjacent rooms via a graph structure. It was chosen because:
- It generates realistic residential layouts trained on 60K+ real floor plans (RPLAN dataset)
- It accepts a room adjacency graph as input, which naturally maps to user requirements
- It supports iterative refinement by room type for higher quality output

---

### Q4: How does the conversational agent extract requirements?

**A:** The Google Gemini 2.0 Flash model is given a system prompt defining it as a "professional architectural consultant." It conducts a 5–7 turn conversation, then uses a separate extraction prompt to parse the entire conversation into a structured JSON:
```json
{
  "num_bedrooms": 2,
  "num_bathrooms": 2,
  "has_balcony": true,
  "has_study": false,
  "has_storage": false,
  "combined_living_dining": true
}
```
If extraction fails, it falls back to sensible architectural defaults.

---

### Q5: How does the graph construction work?

**A:** The `RequirementsConverter` uses a **hub-and-spoke topology** where the Living Room is the central hub. Rules include:
- Each room gets exactly 1 interior door connecting it to its primary neighbour
- Attached bathrooms connect exclusively to their parent bedroom
- Kitchen is always adjacent to the living room
- Forbidden adjacencies (e.g., kitchen ↔ bathroom) are enforced
- The output is a nodes array (room type IDs) and edges array [source, relation, destination] where relation = 1 (adjacent) or -1 (not adjacent).

---

### Q6: Why is a Template Selector used instead of building graphs from scratch?

**A:** House-GAN++ was trained on specific graph structures from the RPLAN dataset. Using a matching template ensures the graph structure is one the model has **actually seen during training**, producing much higher quality outputs than a synthetically constructed graph. The selector scores templates on bedroom count match (weight 10), bathroom count (weight 5), balcony presence (weight 3), and total room similarity (weight 2).

---

### Q7: How does the system select the "best" floor plan from multiple variations?

**A:** The system generates 6 variations using different random seeds, then scores each on a 0–100 scale evaluating:
- **Area coverage**: What percentage of the canvas is filled by rooms
- **Room balance**: Whether rooms have proportional sizes (no oversized/tiny rooms)
- **Shape quality**: Regularity of room shapes
- **Overlap**: Penalizes overlapping room masks
- **Adjacency fidelity**: Whether the generated layout respects the input graph

The highest-scoring variation is automatically selected.

---

### Q8: What are the architectural constraints enforced by the system?

**A:** The `architectural_constraints.py` module enforces rules based on the National Building Code of India and Neufert Architects' Data:
- **Mandatory adjacencies**: Living room ↔ Entrance, Kitchen ↔ Living Room
- **Forbidden adjacencies**: Kitchen ↔ Bathroom (hygiene), Bedroom ↔ Kitchen
- **Hub-and-spoke topology**: Living room must be the central connecting node
- **Door ratios**: N-1 interior doors + 1 front door for N real rooms
- **Post-processing**: Connected component analysis, hole filling, overlap resolution

---

### Q9: What is Convolutional Message Passing (CMP) and why is it important?

**A:** CMP is the core innovation in House-GAN++. It's a graph neural network layer that:
1. Takes room feature maps and an adjacency graph as input
2. Pools features from positively-connected neighbours (adjacent rooms) and negatively-connected neighbours (non-adjacent rooms) separately
3. Concatenates own features + positive pool + negative pool
4. Passes through an encoder to produce updated features

This allows the generator to **reason about spatial relationships** — adjacent rooms get placed next to each other, non-adjacent rooms get separated. It's run iteratively during generation so room placements progressively improve.

---

### Q10: What are the limitations and future scope of this project?

**A:** Current limitations:
- Output is 2D only (no 3D visualization yet)
- No real-world dimensions (sq.ft) on rooms
- Limited to single-floor residential layouts
- Requires internet for Gemini API calls
- Quality depends on how close requirements match training templates

Future scope:
- Multi-floor support (2-story homes)
- Real dimension integration with minimum room size enforcement
- User feedback loop for iterative refinement
- Style preferences (modern/traditional/open-plan)
- Export to CAD formats (DXF) for professional use
- 3D visualization of generated layouts

---

## Quick Talking Points for Presentation

- "This is an end-to-end agentic system — user talks, AI designs."
- "It bridges three domains: NLP, Graph Neural Networks, and Architecture."
- "The agent makes autonomous decisions at every stage — no manual intervention."
- "House-GAN++ from CVPR 2021 provides research-grade generation quality."
- "Template matching ensures generated layouts match patterns the model was trained on."
- "The scoring algorithm mimics what an architect would evaluate: balance, coverage, adjacency."
- "This democratizes architectural design for non-experts."
