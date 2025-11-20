# 🔄 Update Instructions for Phase 2

## After Training Your Model

Once you've trained your custom House-GAN model, follow these steps to use it in Phase 2:

---

## ✅ Step 1: Verify Model File Exists

Make sure you have:
```
AgenticAI/
└── trained_models/
    └── trained_housegan_model.pth  ← Your trained model
```

**If using Colab**: Download the model and place it here first.

---

## ✅ Step 2: Update Phase 2 Notebook

Open: `notebooks/02_floorplan_generator.ipynb`

### Find Cell 9 (Step 3B) - around line 420

**Current code:**
```python
agentic_generator = AgenticFloorPlanGenerator(
    model_path="../2018-house_gan/exp_demo_D_500000.pth",
    device='cpu'
)
```

**Change to:**
```python
agentic_generator = AgenticFloorPlanGenerator(
    model_path="../trained_models/trained_housegan_model.pth",  # ✅ YOUR MODEL
    device='cpu'
)
```

---

## ✅ Step 3: Restart and Run

In Jupyter:
1. **Kernel** → **Restart & Run All**
2. Wait for execution to complete
3. Watch as your custom model generates PERFECT floor plans!

---

## 🎯 What Changed?

### Before (Pre-trained Model):
- ❌ Architecture mismatch errors
- ❌ Failed to load weights
- ❌ Blank/gray floor plans
- ❌ Quality score: 0.50 (random)

### After (Your Trained Model):
- ✅ Perfect architecture match
- ✅ Loads weights successfully
- ✅ Realistic, detailed floor plans
- ✅ Quality score: 0.7-0.9 (excellent!)

---

## 📊 Expected Results

After the update, you should see:

### Step 3B Output:
```
🤖 Initializing Agentic AI Floor Plan Generator...

📂 Loading pre-trained model from: ../trained_models/trained_housegan_model.pth
✅ Pre-trained model loaded successfully!
   • Trained on 145,811 real floor plans
   • Custom trained for bungalow layouts
   • Your model from [training date]
```

### Step 5 Output:
```
🎨 Generating 6 floor plan(s) with agentic intelligence...

   ✓ Variation 1: High quality achieved (score: 0.847) in 2 attempts
   ✓ Variation 2: High quality achieved (score: 0.792) in 1 attempts
   ✓ Variation 3: High quality achieved (score: 0.823) in 3 attempts
   ...
```

**Key differences:**
- Quality scores now **0.7-0.9** instead of 0.5
- Floor plans show **clear room structures**
- **Realistic layouts** with proper adjacencies
- **Fewer regeneration attempts** needed

---

## 🔍 Visual Comparison

### Before (Untrained):
- Gray/blank squares
- No discernible rooms
- Random patterns
- Quality: 0.50

### After (Trained):
- Colorful room layouts
- Clear room boundaries
- Proper adjacencies (bedroom near bathroom, etc.)
- Quality: 0.7-0.9

---

## 🎨 Adjusting Quality Threshold

If your model is very good, you can increase quality standards:

In **Cell 9** (AgenticFloorPlanGenerator class):

```python
class AgenticFloorPlanGenerator:
    def __init__(self, model_path="...", device='cpu'):
        # ...
        
        # Quality thresholds (agentic decision parameters)
        self.quality_threshold = 0.7  # ✅ Increase from 0.5 to 0.7
        self.max_attempts = 10        # Keep at 10
```

**Higher threshold = Better quality but more attempts**

---

## 🚨 Troubleshooting

### Issue: "FileNotFoundError: trained_housegan_model.pth"
**Solution**: 
- Check file path is correct
- Make sure model file exists in `trained_models/`
- If using Colab-trained model, download it first

### Issue: "RuntimeError: Error(s) in loading state_dict"
**Solution**: 
- Model was trained with different architecture
- Re-train model using the latest training notebook
- Ensure you're using `03_train_housegan.ipynb` notebook

### Issue: Quality scores still around 0.5
**Solution**: 
- Model may need more training epochs
- Check training curves - was training successful?
- Try training for 200-500 epochs instead of 100

### Issue: Floor plans still look random
**Solution**: 
- Model didn't train properly
- Check training notebook outputs for errors
- Ensure data files loaded correctly during training
- Try training again with more epochs

---

## ✨ Optimization Tips

### For Best Results:

1. **Quality Threshold**: Start at 0.7, adjust based on results
2. **Max Attempts**: 10 is good balance (speed vs quality)
3. **Variations**: Generate 6-12 variations, pick best ones
4. **Regenerate**: If not satisfied, just re-run Step 5

### For Faster Generation:

```python
self.quality_threshold = 0.6  # Lower threshold
self.max_attempts = 5         # Fewer attempts
```

### For Highest Quality:

```python
self.quality_threshold = 0.85  # Higher threshold
self.max_attempts = 20         # More attempts
```

---

## 🎯 Complete Workflow

1. ✅ Train model (Phase 2.5 - `03_train_housegan.ipynb`)
2. ✅ Download model if using Colab
3. ✅ Update Phase 2 with model path
4. ✅ Restart kernel and run all
5. ✅ Generate perfect floor plans
6. ✅ Move to Phase 3 for 3D visualization

---

## 🎉 Success Indicators

You know it's working when:

- ✅ Model loads without errors
- ✅ Quality scores are 0.7-0.9
- ✅ Floor plans show clear room structures
- ✅ Layouts look realistic and usable
- ✅ Fewer regeneration attempts needed
- ✅ User is satisfied with output quality

---

**🚀 Your custom-trained model is now powering the agentic AI floor plan generator!**
