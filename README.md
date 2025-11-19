# Agentic AI Bungalow Designer

An intelligent AI system that helps customers design their dream bungalow through conversational interaction and generates floor plans with 3D visualizations.

## 🎯 Project Overview

This project uses:
- **Google Gemini API** for intelligent conversation
- **House-GAN** for floor plan generation
- **Python/Jupyter** for interactive development
- **Google Colab** compatible for easy deployment

## 📁 Project Structure

```
AgenticAI/
├── notebooks/
│   ├── 01_chatbot_agent.ipynb          # ✅ Phase 1: Requirements gathering
│   ├── 02_floorplan_generator.ipynb    # 🚧 Phase 2: Floor plan generation
│   ├── 03_3d_visualizer.ipynb          # 🚧 Phase 3: 3D rendering
│   └── 04_full_pipeline.ipynb          # 🚧 Complete integrated system
├── src/
│   └── agent/
│       └── gemini_agent.py             # Gemini API wrapper
├── 2018-house_gan/                      # House-GAN model & dataset
├── cubicasa5k/                          # Additional floor plan data
├── requirements.txt                     # Python dependencies
└── REQUIREMENTS.md                      # Detailed project requirements
```

## 🚀 Getting Started

### Phase 1: Requirements Gathering (✅ Ready)

1. **Open the notebook:**
   ```bash
   code notebooks/01_chatbot_agent.ipynb
   ```
   Or upload to Google Colab: https://colab.research.google.com/

2. **Get Gemini API Key:**
   - Visit: https://makersuite.google.com/app/apikey
   - Create a free API key
   - Free tier: 15 requests/minute

3. **Run the notebook:**
   - Follow the cells sequentially
   - Start a conversation with the AI agent
   - Extract and export requirements

### Phase 2: Floor Plan Generation (Coming Soon)
- Uses House-GAN model
- Generates custom floor plans
- Multiple design variations

### Phase 3: 3D Visualization (Coming Soon)
- Creates 3D structure previews
- Photorealistic renders
- Interactive viewing

## 📦 Installation

### Local Development (VS Code)
```bash
pip install -r requirements.txt
```

### Google Colab
No installation needed! The notebook installs dependencies automatically.

## 🔑 Configuration

Set your Gemini API key:

**In Code:**
```python
GEMINI_API_KEY = "your-api-key-here"
```

**As Environment Variable:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**In Google Colab:**
Use Colab secrets or input directly in the notebook.

## 📚 Documentation

- **REQUIREMENTS.md** - Comprehensive implementation guide
- **Phase 1 Notebook** - Interactive chatbot with inline documentation
- **src/agent/gemini_agent.py** - API integration documentation

## 🎯 Features

### Phase 1: ✅ Complete
- [x] Interactive conversational AI
- [x] Natural requirement extraction
- [x] Structured data output (JSON)
- [x] Beautiful UI with ipywidgets
- [x] Save/export functionality
- [x] Google Colab compatible

### Phase 2: 🚧 In Development
- [ ] House-GAN integration
- [ ] Graph-based floor plan generation
- [ ] Constraint validation
- [ ] Multiple design variations
- [ ] 2D floor plan visualization

### Phase 3: 🚧 Planned
- [ ] 3D structure generation
- [ ] AI-powered rendering
- [ ] Interactive 3D viewer
- [ ] Export to various formats

## 🛠️ Tech Stack

- **AI/ML:** Google Gemini API, PyTorch, House-GAN
- **UI:** ipywidgets, Jupyter Notebook
- **Visualization:** Matplotlib, Plotly, PIL
- **Data:** NumPy, Pandas, JSON
- **Platform:** Python 3.8+, Google Colab

## 📖 Usage Example

```python
# Initialize agent
from src.agent.gemini_agent import BungalowDesignAgent

agent = BungalowDesignAgent(api_key="your-key")

# Start conversation
response = agent.start_conversation()
print(response)

# Send user message
user_input = "I want a 3-bedroom bungalow"
response = agent.send_message(user_input)

# Extract requirements
requirements = agent.extract_requirements()
print(requirements)
```

## 🎓 Learn More

- [House-GAN Paper](https://arxiv.org/abs/2003.06988)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Project Requirements](REQUIREMENTS.md)

## 📊 Project Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Chatbot | ✅ Complete | 100% |
| Phase 2: Floor Plan | 🚧 Next | 0% |
| Phase 3: 3D Render | ⏳ Planned | 0% |

## 🤝 Contributing

This is a personal project for bungalow design automation. Suggestions welcome!

## 📝 License

Uses House-GAN dataset under SFU license. See `2018-house_gan/LICENSE.txt`

## 🎉 Acknowledgments

- House-GAN by Nelson Nauata et al.
- Google Gemini API
- CubiCasa5K dataset

---

**Ready to design your dream bungalow?** Start with `notebooks/01_chatbot_agent.ipynb`! 🏠✨
