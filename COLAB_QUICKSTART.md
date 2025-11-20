# 🚀 Google Colab Quick Start - 5 Minutes to Training!

## Step 1: Open Colab
👉 https://colab.research.google.com/

## Step 2: Upload Notebook
- **File** → **Upload notebook**
- Choose: `03_train_housegan.ipynb`

## Step 3: Enable FREE GPU 🎯
- **Runtime** → **Change runtime type**
- Hardware accelerator: **GPU**
- GPU type: **T4**
- Click **Save**

## Step 4: Upload Data Files 📁
Click folder icon on left, then upload:
- `train_data.npy` (from `2018-house_gan/dataset_paper/`)
- `valid_data.npy` (from `2018-house_gan/dataset_paper/`)

**⏱️ Upload time: 5-10 minutes (files are large)**

## Step 5: Start Training 🏃
- **Runtime** → **Run all**
- Or press: **Ctrl+F9** (Windows) / **Cmd+F9** (Mac)

**⏱️ Training time: 2-3 hours**

## Step 6: Download Trained Model 💾
When done:
- Find: `trained_models/trained_housegan_model.pth`
- Right-click → **Download**
- Copy to: `AgenticAI/trained_models/` on your computer

## Step 7: Use Your Model ✨
In `02_floorplan_generator.ipynb` (Step 3B), change:

```python
model_path="../trained_models/trained_housegan_model.pth"
```

**🎉 Done! Now generate PERFECT floor plans!**

---

## ⚡ Pro Tips:

1. **Don't close browser** during training (2-3 hours)
2. **Move mouse occasionally** to prevent timeout
3. **Check progress** - you'll see epoch updates
4. **Training complete?** Look for "🎉 Training complete!" message

---

## 🐛 Quick Fixes:

**"Out of memory"?**
- In Step 2, change `batch_size: 32` to `batch_size: 16`

**"File not found"?**
- Make sure you uploaded both .npy files in Step 4

**Colab disconnected?**
- Reconnect and look for checkpoints in `trained_models/`
- Download the latest `checkpoint_epoch_XX.pth`

---

## 📊 How to Know Training is Working:

You'll see output like:
```
Epoch [1/100] | G Loss: 1.234 | D Loss: 0.567 | ETA: 2.3h
Epoch [2/100] | G Loss: 1.156 | D Loss: 0.589 | ETA: 2.2h
...
```

✅ **Good signs:**
- Losses are decreasing
- ETA countdown is working
- No error messages

❌ **Warning signs:**
- Red error messages
- Training stops unexpectedly
- "Out of memory" errors

---

## 🎯 What You're Training:

- **Input**: 145,811 real floor plans
- **Output**: AI that generates perfect bungalow layouts
- **Time**: 2-3 hours on free Colab GPU
- **Result**: Custom model trained for YOUR project

---

**💡 First time using Colab? Don't worry - just follow the steps above!**
