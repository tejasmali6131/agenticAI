# 🎓 Training Your Custom House-GAN Model

## 📋 Quick Start Guide

This guide will help you train your own House-GAN model for perfect bungalow floor plan generation.

---

## 🚀 Option 1: Google Colab (Recommended - Free GPU!)

### Step-by-Step Instructions:

#### 1. **Open Google Colab**
- Go to: https://colab.research.google.com/
- Sign in with your Google account (free)

#### 2. **Upload Training Notebook**
- Click: **File → Upload notebook**
- Select: `notebooks/03_train_housegan.ipynb`
- Wait for upload to complete

#### 3. **Enable Free GPU**
- Click: **Runtime → Change runtime type**
- Hardware accelerator: Select **GPU**
- GPU type: Select **T4** (free tier)
- Click **Save**

#### 4. **Upload Training Data**
- Look for the **folder icon** 📁 on the left sidebar
- Click the **upload button** ⬆️
- Upload these 2 files from your computer:
  - `2018-house_gan/dataset_paper/train_data.npy`
  - `2018-house_gan/dataset_paper/valid_data.npy`
- **Note**: These files are large (~500MB each), upload may take 5-10 minutes

#### 5. **Start Training**
- Click: **Runtime → Run all**
- Or press: **Ctrl+F9** (Windows) / **Cmd+F9** (Mac)
- **Estimated time**: 2-3 hours on T4 GPU

#### 6. **Monitor Progress**
- Watch the training progress in real-time
- You'll see epoch updates like:
  ```
  Epoch [1/100] | G Loss: 1.234 | D Loss: 0.567 | ETA: 2.3h
  ```
- **Don't close the browser tab** while training

#### 7. **Download Trained Model**
- When training completes, look for: `trained_models/trained_housegan_model.pth`
- Click the **three dots** ⋮ next to the file
- Select **Download**
- File size: ~50-100 MB

#### 8. **Use Your Model**
- Copy downloaded file to your local project:
  ```
  AgenticAI/trained_models/trained_housegan_model.pth
  ```
- Update Phase 2 notebook (see below)

---

## 💻 Option 2: Local Training (Overnight on Your Computer)

### Requirements:
- Python 3.8+
- 8GB+ RAM
- 10GB+ free disk space
- **Time**: 8-12 hours on CPU (overnight)

### Steps:

#### 1. **Activate Virtual Environment**
```powershell
cd C:\AgenticAI
.venv\Scripts\Activate.ps1
```

#### 2. **Open Training Notebook**
```powershell
jupyter notebook notebooks/03_train_housegan.ipynb
```

#### 3. **Run All Cells**
- Click: **Cell → Run All**
- Or use **Shift+Enter** to run cell by cell

#### 4. **Let It Run Overnight**
- Training will take 8-12 hours on CPU
- Your computer can do other tasks while training
- Model saves automatically every 10 epochs

#### 5. **Check Results**
- Trained model will be in: `trained_models/trained_housegan_model.pth`
- Training curves will show if model learned successfully

---

## 🎯 Using Your Trained Model in Phase 2

After training, update `notebooks/02_floorplan_generator.ipynb`:

### Find Step 3B (Cell 9):
```python
# OLD CODE (line ~420):
agentic_generator = AgenticFloorPlanGenerator(
    model_path="../2018-house_gan/exp_demo_D_500000.pth",  # ❌ Don't use this
    device='cpu'
)

# NEW CODE:
agentic_generator = AgenticFloorPlanGenerator(
    model_path="../trained_models/trained_housegan_model.pth",  # ✅ YOUR MODEL!
    device='cpu'
)
```

### Then Run Phase 2:
1. Restart kernel: **Kernel → Restart & Run All**
2. Watch as it generates **PERFECT** floor plans!
3. Your custom model understands bungalow layouts

---

## 📊 Training Quality Indicators

### Good Training Signs ✅:
- Generator Loss decreases steadily
- Discriminator Loss stabilizes around 0.5-0.7
- D(real) stays near 1.0
- D(fake) approaches 0.5
- Generated samples look like real floor plans

### Warning Signs ⚠️:
- Losses oscillate wildly
- D(fake) stays near 0.0 (generator not learning)
- D(real) drops below 0.7 (discriminator too weak)
- Generated samples are random noise

**If you see warning signs**: Train for more epochs (increase `num_epochs` in Step 2)

---

## 🔧 Training Configuration Options

In **Step 2** of the training notebook, you can adjust:

```python
CONFIG = {
    'num_epochs': 100,    # Adjust based on time:
                          # - 10 epochs: 10 min (test)
                          # - 100 epochs: 2 hours (good)
                          # - 500 epochs: 8 hours (best)
    
    'batch_size': 32,     # Reduce to 16 if out of memory
    
    'learning_rate': 0.0001,  # Default works well
}
```

### Recommended Settings:

| Hardware | Epochs | Batch Size | Time | Quality |
|----------|--------|------------|------|---------|
| Colab GPU (T4) | 100 | 32 | 2-3h | Good |
| Colab GPU (T4) | 300 | 32 | 6-8h | Excellent |
| Local CPU | 50 | 16 | 6-8h | Decent |
| Local CPU | 100 | 16 | 12-16h | Good |
| Local GPU (NVIDIA) | 200 | 32 | 3-4h | Excellent |

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory" (Colab)
**Solution**: In Step 2, change `batch_size` from 32 to 16

### Issue: "FileNotFoundError: train_data.npy"
**Solution**: Make sure you uploaded the data files in Step 4

### Issue: Colab disconnects during training
**Solution**: 
- Stay on the tab (move mouse occasionally)
- Use browser extension like "Colab Keep Alive"
- Download intermediate checkpoints every 10 epochs

### Issue: Generated samples look random even after training
**Solution**: 
- Train for more epochs (100 → 200 → 500)
- Check training curves - losses should decrease
- Ensure data loaded correctly (check Step 4 output)

### Issue: Training is too slow locally
**Solution**: 
- Use Google Colab instead (free GPU)
- Reduce `num_epochs` for faster test run
- Let it run overnight on CPU

---

## 💾 Important Files

After training, you'll have:

```
AgenticAI/
├── trained_models/
│   ├── trained_housegan_model.pth          ← YOUR FINAL MODEL (use this!)
│   ├── checkpoint_epoch_10.pth              ← Backup checkpoint
│   ├── checkpoint_epoch_20.pth              ← Backup checkpoint
│   └── ...
└── notebooks/
    └── 03_train_housegan.ipynb              ← Training notebook
```

**Keep all checkpoints** in case you want to resume training or compare different epochs.

---

## 🎉 Success Checklist

After training, you should have:

- ✅ Training completed without errors
- ✅ Training curves showing learning progress
- ✅ Generated samples that look like floor plans
- ✅ Final model saved: `trained_housegan_model.pth`
- ✅ Model integrated into Phase 2
- ✅ Phase 2 generating high-quality floor plans

---

## 🚀 Next Steps

1. ✅ **Train model** using this guide
2. ✅ **Download trained model** (if using Colab)
3. ✅ **Update Phase 2** with your model path
4. ✅ **Generate perfect floor plans** in Phase 2
5. ✅ **Move to Phase 3** for 3D visualization

---

## 💡 Pro Tips

1. **Save your Colab notebook**: File → Save a copy in Drive
2. **Download checkpoints**: Save intermediate epochs as backup
3. **Monitor training**: Check generated samples every 20 epochs
4. **Experiment**: Try different epochs to find best quality
5. **Document**: Note which epoch gave best results

---

## 📞 Need Help?

If you encounter issues:

1. Check the troubleshooting section above
2. Review training curves for abnormal patterns
3. Try the test configuration first (10 epochs)
4. Compare your results with the quality indicators

---

**🎓 Happy Training! You're about to create a custom AI model specifically for your bungalow floor plan generation!**
