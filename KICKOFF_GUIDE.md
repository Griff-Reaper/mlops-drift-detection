# 🚀 Project 3 Kickoff Guide - Start Today!

**Target:** Get your MLOps pipeline project set up and running Phase 1 this week

**Time Required Today:** 2-3 hours for initial setup

---

## 🎯 What We're Building

An automated machine learning pipeline that:
1. Trains a network traffic classifier
2. Tracks experiments with MLflow  
3. Monitors for data drift
4. Automatically retrains when drift is detected
5. Has full CI/CD integration

This project demonstrates **production MLOps skills** that separate junior from senior ML engineers.

---

## 📅 Today's Goals (2-3 hours)

By the end of today, you'll have:
- [x] Project repository created locally
- [ ] Virtual environment set up
- [ ] All dependencies installed
- [ ] Dataset downloaded
- [ ] MLflow tracking server running
- [ ] First experiment tracked successfully

---

## Step-by-Step: Let's Get Started!

### Step 1: Copy Project to Your Machine (5 minutes)

You'll receive a ZIP file with the project structure. Extract it to your development folder:

```bash
# Extract to your preferred location
cd ~/projects  # or wherever you keep code
# Extract the mlops-drift-detection folder here
```

Or if you want to start fresh:

```bash
cd ~/projects
mkdir mlops-drift-detection
cd mlops-drift-detection
```

---

### Step 2: Create Virtual Environment (2 minutes)

```bash
cd mlops-drift-detection
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 3: Install Dependencies (5-10 minutes)

This will install all the tools we need (Prefect, MLflow, Evidently, etc.):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This might take a few minutes. Grab some water! 💧

**Verify installation:**
```bash
python -c "import mlflow, prefect, evidently; print('All imports successful!')"
```

You should see "All imports successful!" with no errors.

---

### Step 4: Set Up Configuration (2 minutes)

```bash
# Copy the example environment file
cp .env.example .env

# You can leave it as-is for now, or edit if you want
```

The defaults are fine for local development.

---

### Step 5: Download the Dataset (10-15 minutes)

We're using the **CICIDS2017** network traffic dataset. It's about 2GB.

I've created a download script for you:

```bash
python scripts/download_data.py
```

This will:
- Download the dataset from the official source
- Extract it to `data/raw/`
- Show a progress bar

**Alternative if download is slow:** 
Manual download link: https://www.unb.ca/cic/datasets/ids-2017.html
- Download "Friday-WorkingHours.pcap_ISCX.csv" (smallest file, ~500MB)
- Place in `data/raw/` folder

---

### Step 6: Start MLflow Tracking Server (2 minutes)

Open a **NEW terminal window** (keep your main one open):

```bash
cd mlops-drift-detection
source venv/bin/activate  # Windows: venv\Scripts\activate
mlflow server --host 127.0.0.1 --port 5000
```

You should see:
```
[INFO] Starting gunicorn 20.1.0
[INFO] Listening at: http://127.0.0.1:5000
```

**Open your browser** to http://localhost:5000

You should see the MLflow UI! 🎉

Leave this terminal running - MLflow needs to stay active.

---

### Step 7: Run Your First Experiment (15-20 minutes)

Back in your main terminal (with venv activated):

```bash
python scripts/test_setup.py
```

This script will:
1. Load a sample of the data
2. Train a simple model
3. Log metrics to MLflow
4. Save the model

Watch the output! You'll see:
- Data loading progress
- Training metrics
- MLflow experiment ID

**Then check MLflow UI:**
- Go to http://localhost:5000
- Click on the "network_traffic_classification" experiment
- You should see your first run with metrics!

---

##  ✅ Success Checklist

By the end of today, verify:

- [ ] Virtual environment created and activated
- [ ] All Python packages installed successfully
- [ ] Dataset downloaded to `data/raw/`
- [ ] MLflow server running on port 5000
- [ ] MLflow UI accessible in browser
- [ ] First test experiment logged in MLflow
- [ ] Can see experiment metrics and parameters in UI

---

## 🎉 If You Got This Far - You're Crushing It!

You've just set up a **production-grade MLOps environment**. This alone puts you ahead of most candidates.

---

## 🔜 What's Next?

### Tomorrow/This Weekend (Phase 1 continuation):

1. **Explore the data** (30 min)
   - Run the EDA notebook: `jupyter notebook notebooks/01_eda.ipynb`
   - Look at feature distributions
   - Check for class imbalance

2. **Set up Prefect** (30 min)
   - Initialize Prefect workspace
   - Create your first flow
   - Test local orchestration

3. **Start data preprocessing** (1-2 hours)
   - Implement `src/data/preprocessing.py`
   - Handle missing values
   - Feature engineering
   - Save processed data

### This Week (Phase 1 completion):

- [ ] Complete data pipeline
- [ ] Build baseline model
- [ ] Set up experiment tracking workflow
- [ ] Configure DVC for data versioning
- [ ] Create training pipeline with Prefect

---

## 🆘 Troubleshooting

### "Module not found" errors
```bash
# Make sure venv is activated (you should see (venv) in prompt)
# Reinstall requirements:
pip install -r requirements.txt
```

### MLflow won't start
```bash
# Check if port 5000 is in use:
# Windows:
netstat -ano | findstr :5000

# Mac/Linux:
lsof -i :5000

# Kill the process or use a different port:
mlflow server --host 127.0.0.1 --port 5001
```

### Dataset download fails
- Use the manual download link above
- Or try a smaller file from CICIDS2017
- Alternative: Use UNSW-NB15 dataset instead

### "Permission denied" errors
- Make sure you have write access to the project folder
- On Mac/Linux, avoid using sudo with pip

---

## 💡 Tips for Success

**1. Keep MLflow Running**
- Always have MLflow server running when working on the project
- It tracks all your experiments automatically

**2. Small Iterations**
- Don't try to build everything at once
- Get one piece working, then move to the next
- Commit to git frequently

**3. Use the Tracker**
- Check off tasks in the project tracker as you complete them
- It's motivating to see progress!

**4. Ask for Help**
- If stuck for >30 minutes, ask Claude for help
- Screenshot errors for faster debugging

**5. Document As You Go**
- Take notes on what worked/didn't
- You'll use these for your portfolio blog post

---

## 📊 Phase 1 Roadmap (This Week)

```
Day 1 (Today):     ✅ Setup + MLflow
Day 2-3:           📊 Data exploration + preprocessing
Day 4-5:           🤖 Baseline model + Prefect integration  
Day 6-7:           📦 DVC setup + training pipeline
End of Week 1:     🎉 Complete Phase 1!
```

---

## 🎯 Remember the Goal

This project demonstrates:
- **MLOps expertise** - Not just ML, but production ML systems
- **Tool proficiency** - Prefect, MLflow, Evidently, DVC
- **Production mindset** - Monitoring, versioning, automation
- **Portfolio value** - Stands out to hiring managers

**Every hour you invest here is an hour closer to your target role at CrowdStrike or a defense contractor.**

---

## 🚀 Ready? Let's Do This!

Start with Step 1 above. Take it one step at a time. You've already crushed two projects - you've got this! 💪

Questions? Just ask. I'm here to help.

**Let's build something awesome!** 🔥
