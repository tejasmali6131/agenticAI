# 🤖 Agentic AI System for Bungalow Floor Plan Generation

## What Makes This System "Agentic AI"?

This project implements a **true multi-agent AI system** where multiple AI agents work together autonomously to achieve a goal: generating perfect bungalow floor plans based on customer requirements.

---

## 🧠 The Two AI Agents

### **Agent 1: Gemini Requirements Gatherer** (Phase 1)
**Type:** Large Language Model (LLM) Agent  
**Role:** Conversational requirement extraction

**Agentic Capabilities:**
- ✅ **Autonomous conversation** - Decides what questions to ask
- ✅ **Context understanding** - Interprets vague user inputs
- ✅ **Decision-making** - Determines when enough information is gathered
- ✅ **Structured output** - Converts unstructured text to JSON
- ✅ **Adaptive behavior** - Asks follow-up questions based on user answers

**Why it's agentic:** It doesn't just respond to prompts; it actively guides the conversation to achieve the goal of collecting complete bungalow requirements.

---

### **Agent 2: House-GAN Floor Plan Generator** (Phase 2)
**Type:** Generative Adversarial Network (GAN) Agent  
**Role:** Intelligent floor plan generation with quality control

**Agentic Capabilities:**
- ✅ **Learned intelligence** - Trained on 145,811 real floor plans
- ✅ **Quality evaluation** - Uses discriminator to assess its own outputs
- ✅ **Autonomous regeneration** - Tries again if quality is below threshold
- ✅ **Best selection** - Generates multiple candidates, picks the best
- ✅ **Adaptive parameters** - Can adjust generation based on constraints
- ✅ **Goal-oriented** - Keeps trying until quality threshold is met

**Why it's agentic:** It doesn't just generate once and stop. It evaluates, makes decisions, and takes actions (regenerate/accept) to ensure high-quality output.

---

## 🔄 The Agentic Workflow

```
User Input: "I want a 2-bedroom bungalow"
         ↓
    [AGENT 1: Gemini]
    • Analyzes input
    • Decides: "Need more info"
    • Asks: "How many bathrooms?"
    • Continues conversation
    • Extracts structured data
         ↓
    Requirements JSON
         ↓
    [AGENT 2: House-GAN]
    • Generates floor plan (Attempt 1)
    • Evaluates quality: 0.45 (low)
    • Decision: "Regenerate"
    • Generates floor plan (Attempt 2)
    • Evaluates quality: 0.72 (good!)
    • Decision: "Accept"
         ↓
    High-quality floor plan output
```

---

## 🎯 Key Agentic AI Characteristics

### 1. **Autonomy**
- Agents make decisions without human intervention
- No manual quality checks needed
- Automatic regeneration until goals are met

### 2. **Goal-Oriented Behavior**
- **Goal:** Generate architecturally sound floor plan
- **Actions:** Generate → Evaluate → Decide → Regenerate/Accept
- Persists until goal is achieved

### 3. **Learning from Experience**
- House-GAN learned from 145K real floor plans
- Gemini learned from massive text corpora
- Both use learned knowledge to make intelligent decisions

### 4. **Adaptive Decision-Making**
- Adjusts strategy based on feedback
- Changes generation parameters if needed
- Responds to quality metrics

### 5. **Multi-Agent Collaboration**
- Agent 1 output becomes Agent 2 input
- Seamless handoff between agents
- Each specializes in its task

---

## 📊 What's NOT Agentic AI

### ❌ Pure Algorithmic Generator (Previous Version)
```python
# Not agentic - just follows rules
def place_rooms(rooms):
    for room in rooms:
        x, y = random_position()
        place_at(x, y)
    return layout
```
- No learning
- No quality evaluation
- No adaptive behavior
- Fixed rules only

### ❌ Simple ML Model Usage
```python
# Not agentic - one-shot generation
layout = model.predict(requirements)
return layout
```
- No decision-making
- No self-evaluation
- No regeneration logic

---

## 🚀 The Complete System

```
┌─────────────────────────────────────────────┐
│         AGENTIC AI SYSTEM                   │
├─────────────────────────────────────────────┤
│                                             │
│  Phase 1: Gemini Agent                      │
│  ├─ Conversational AI                       │
│  ├─ Requirement extraction                  │
│  ├─ Context understanding                   │
│  └─ Structured output generation            │
│                                             │
│  Phase 2: House-GAN Agent                   │
│  ├─ Neural network (145K trained)           │
│  ├─ Quality evaluation (Discriminator)      │
│  ├─ Autonomous regeneration                 │
│  ├─ Best candidate selection                │
│  └─ Quality scoring                         │
│                                             │
│  Phase 3: 3D Visualization (Coming)         │
│  └─ Render floor plans in 3D               │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 💡 Why This Matters

**Traditional Approach:**
1. User fills form
2. System generates once
3. User accepts or rejects
4. Manual regeneration

**Agentic AI Approach:**
1. AI extracts requirements conversationally
2. AI generates, evaluates, regenerates automatically
3. AI delivers high-quality result
4. Minimal human intervention

**Result:** Higher quality, better user experience, intelligent automation

---

## 🔧 Technical Implementation

### Agent 1 (Gemini)
- **Framework:** Google Generative AI API
- **Model:** gemini-2.5-flash
- **Decision Logic:** Built into LLM
- **Output:** JSON with structured requirements

### Agent 2 (House-GAN)
- **Framework:** PyTorch
- **Architecture:** Generator + Discriminator
- **Training:** 145,811 real floor plans
- **Decision Logic:** Custom `AgenticFloorPlanGenerator` wrapper
  - Quality threshold: 0.6
  - Max attempts: 10
  - Automatic best selection

---

## 📈 Quality Metrics

The system provides transparency into AI decision-making:

- **Quality Score (0-1):** Discriminator's confidence
  - < 0.4: Poor quality, regenerate
  - 0.4-0.6: Acceptable
  - > 0.6: High quality, accept

- **Attempt Count:** Shows AI persistence
- **Average Quality:** Overall performance metric

---

## 🎓 Learning Resources

**What is Agentic AI?**
- Autonomous decision-making systems
- Goal-oriented behavior
- Self-evaluation and adaptation
- Multi-agent collaboration

**This Project Demonstrates:**
- LLM agents (Gemini)
- Generative model agents (GAN)
- Agent orchestration
- Quality-driven decision loops

---

## 🏆 Conclusion

This is **TRUE Agentic AI** because:
1. ✅ Multiple AI agents working together
2. ✅ Autonomous decision-making at each phase
3. ✅ Self-evaluation and quality control
4. ✅ Adaptive regeneration behavior
5. ✅ Goal-oriented (perfect floor plans)
6. ✅ Minimal human intervention needed

**Not just AI tools, but AI agents that think, decide, and act!** 🤖
