# 🚀 Quick Start Guide - Phase 1

## What You Have Now

✅ **Fully functional conversational AI chatbot** that:
- Asks intelligent questions about your bungalow requirements
- Extracts all necessary specifications
- Validates and structures data into JSON
- Saves requirements for Phase 2 (floor plan generation)

---

## 🎯 Two Ways to Run

### Option 1: Google Colab (Recommended for Beginners)

1. **Upload the notebook:**
   - Go to: https://colab.research.google.com/
   - Click **File → Upload notebook**
   - Upload: `notebooks/01_chatbot_agent.ipynb`

2. **Run it:**
   - The notebook will auto-install dependencies
   - Enter your API key when prompted: `AIzaSyBRzjI1yGlMz0ou9tlD-MMuZKQzmVVICfk`
   - Follow the instructions in each cell

3. **Download results:**
   - After extraction, download the JSON files
   - Keep them for Phase 2

---

### Option 2: VS Code (Your Current Setup)

1. **Open the notebook in VS Code:**
   - You're already in the right place!
   - Open: `notebooks/01_chatbot_agent.ipynb`

2. **Select Python kernel:**
   - Click "Select Kernel" at top right
   - Choose your Python 3.8+ environment

3. **Install dependencies:**
   ```powershell
   pip install google-generativeai ipywidgets
   ```

4. **Run the notebook:**
   - Execute cells sequentially (Shift+Enter)
   - Your API key is already saved: `AIzaSyBRzjI1yGlMz0ou9tlD-MMuZKQzmVVICfk`

---

## 📋 What to Expect

### Step-by-Step Flow:

1. **Installation** (Cell 1-2)
   - Installs required packages
   - Takes ~30 seconds

2. **API Configuration** (Cell 3-4)
   - Sets up your Gemini API key
   - Tests connection

3. **Agent Initialization** (Cell 5-7)
   - Loads the conversational AI
   - Creates the chat interface

4. **Interactive Chat** (Cell 8)
   - Click "New Chat" to start
   - Answer questions naturally
   - Example conversation:
     ```
     Agent: Hello! I'm excited to help you design your dream bungalow...
     You: I want a 3-bedroom house with a garden
     Agent: Great! How much land area do you have?
     You: About 2000 square feet
     Agent: Perfect! Would you like open or closed kitchen?
     ... (continues)
     ```

5. **Extract Requirements** (Cell 9)
   - Click "Extract Requirements" when done
   - Gets structured JSON output
   - Displays summary

6. **Export** (Cell 10)
   - Saves requirements for Phase 2
   - Creates JSON files in `generated_requirements/`

---

## 🎨 Sample Conversation

Here's what a typical interaction looks like:

```
🤖 Agent: Hello! I'm thrilled to help you design your dream bungalow. 
Let's start with some basic questions. What type of bungalow are you 
envisioning? For example, single-story, duplex, villa, or cottage?

👤 You: I want a single-story bungalow

🤖 Agent: Excellent choice! A single-story bungalow is great for 
accessibility. Now, could you tell me:
1. How much land area do you have? (in square feet or square meters)
2. How many bedrooms would you like?

👤 You: I have 2500 sqft land and need 3 bedrooms

🤖 Agent: Perfect! For the 3 bedrooms:
- Would you like one to be a master bedroom with attached bathroom?
- Any specific size preferences (small, medium, or large)?

... (conversation continues)
```

---

## 📊 Output Example

After extraction, you'll get:

```json
{
  "bungalow_type": "single-story",
  "land_area": {"value": 2500, "unit": "sqft"},
  "bedrooms": {
    "count": 3,
    "sizes": ["master - large", "medium", "medium"],
    "master_bedroom": true
  },
  "bathrooms": {"count": 3, "attached": 1, "common": 2},
  "living_room": {"required": true, "size": "large"},
  "kitchen": {"type": "open", "size": "medium"},
  "parking": {"two_wheeler": 2, "four_wheeler": 1},
  ...
}
```

---

## ❓ Troubleshooting

### Issue: API Key Error
**Solution:** Check that your key is correct:
```python
AIzaSyBRzjI1yGlMz0ou9tlD-MMuZKQzmVVICfk
```

### Issue: Module Not Found
**Solution:** Install dependencies:
```powershell
pip install google-generativeai ipywidgets
```

### Issue: Widgets Not Showing
**Solution:** In VS Code, install Jupyter extension:
- Ctrl+Shift+X → Search "Jupyter" → Install

### Issue: Rate Limit Error
**Solution:** You're on free tier (15 requests/min). Wait 1 minute and retry.

---

## 🎯 Success Checklist

After completing Phase 1, you should have:

- [x] Successfully started a conversation with the agent
- [x] Answered all questions about your bungalow
- [x] Extracted requirements into JSON format
- [x] Saved files in `generated_requirements/` folder
- [x] Reviewed the summary of your requirements

---

## 🚀 Next Steps

Once Phase 1 is complete:

1. **Keep your JSON file** - You'll need it for Phase 2
2. **Wait for Phase 2 notebook** - Floor plan generation with House-GAN
3. **Phase 3 will follow** - 3D visualization

---

## 💡 Tips for Best Results

1. **Be specific:** "3 bedrooms with one master" vs "some bedrooms"
2. **Provide context:** "Open kitchen for modern look" vs just "open"
3. **Ask questions:** The agent is designed to clarify and suggest
4. **Take your time:** No rush, think about what you really want
5. **Review carefully:** Check the extracted requirements before Phase 2

---

## 🎓 Understanding the Output

The JSON file structure:
- **Required fields:** Bungalow type, land area, bedrooms, bathrooms
- **Optional fields:** Additional rooms, preferences, special requirements
- **Phase 2 will use:** Room counts, sizes, adjacencies, parking
- **Phase 3 will use:** Style preferences, outdoor spaces

---

## 📞 Need Help?

**Common Questions:**

**Q: How long should the conversation be?**  
A: Usually 8-15 exchanges. The agent will ask until it has enough info.

**Q: Can I edit the JSON manually?**  
A: Yes! After extraction, you can open the JSON file and modify it.

**Q: What if I want to start over?**  
A: Click "New Chat" button to restart the conversation.

**Q: Can I save multiple designs?**  
A: Yes! Each extraction creates a new timestamped file.

---

## ✨ You're Ready!

Open `notebooks/01_chatbot_agent.ipynb` and start designing your dream bungalow! 🏠

**Your API Key:** `AIzaSyBRzjI1yGlMz0ou9tlD-MMuZKQzmVVICfk`

---

*Generated: November 19, 2025*
