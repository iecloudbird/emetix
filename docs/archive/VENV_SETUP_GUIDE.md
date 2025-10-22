# Quick Setup and Testing Guide

## Activate Virtual Environment First!

Before running any scripts, always activate your virtual environment:

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Command Prompt (CMD)

```cmd
venv\Scripts\activate.bat
```

You should see `(venv)` prefix in your terminal prompt when activated.

---

## Automated Setup

We've created setup scripts for easy installation:

### Option 1: PowerShell (Recommended)

```powershell
.\setup_ml_models.ps1
```

### Option 2: Command Prompt

```cmd
setup_ml_models.bat
```

These scripts will:

1. ✓ Activate virtual environment (or create if missing)
2. ✓ Upgrade pip
3. ✓ Install PyTorch (CPU version)
4. ✓ Install all ML/DL packages
5. ✓ Verify installation

---

## Manual Setup (If scripts fail)

### Step 1: Activate Virtual Environment

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Or CMD
venv\Scripts\activate.bat
```

### Step 2: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 3: Install PyTorch

```powershell
# CPU version (recommended for most users)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR GPU version (if you have CUDA)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install ML Packages

```powershell
python -m pip install pytorch-lightning scikit-learn scikit-optimize shap joblib statsmodels
```

### Step 5: Verify Installation

```powershell
python -c "import torch; import pytorch_lightning; import sklearn; print('All packages installed!')"
```

---

## Running Scripts (Always in venv!)

### 1. Test Models

```powershell
# Make sure venv is activated first!
python scripts\test_ml_models.py
```

### 2. Fetch Training Data

```powershell
python scripts\fetch_historical_data.py
```

### 3. Train LSTM-DCF

```powershell
python scripts\train_lstm_dcf.py
```

### 4. Train RF Ensemble

```powershell
python scripts\train_rf_ensemble.py
```

---

## Common Issues

### Issue 1: "venv not found"

**Solution**: Create virtual environment first

```powershell
python -m venv venv
```

### Issue 2: "Execution policy error" (PowerShell)

**Solution**: Allow script execution

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 3: "pip not found"

**Solution**: Use python -m pip

```powershell
python -m pip install --upgrade pip
```

### Issue 4: "Import error: torch not found"

**Solution**: Make sure venv is activated and PyTorch is installed

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Check Current Environment

To verify you're in the virtual environment:

```powershell
# PowerShell
Get-Command python | Select-Object Source

# CMD
where python
```

Should show path to `venv\Scripts\python.exe`

---

## Package Verification

After installation, verify packages:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import pytorch_lightning as pl; print('Lightning:', pl.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import joblib; print('joblib:', joblib.__version__)"
```

Expected output:

```
PyTorch: 2.1.0+cpu (or similar)
Lightning: 2.1.0 (or similar)
scikit-learn: 1.3.0 (or similar)
joblib: 1.3.0 (or similar)
```

---

## Installation Size

Expected disk space requirements:

- PyTorch (CPU): ~700 MB
- PyTorch Lightning: ~50 MB
- Other packages: ~200 MB
- **Total**: ~1 GB

Installation time: 5-15 minutes (depends on internet speed)

---

## Deactivate Virtual Environment

When you're done:

```powershell
deactivate
```

---

## Quick Command Reference

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Check you're in venv
python --version  # Should show Python 3.11.x
which python      # Should show venv/Scripts/python.exe (PowerShell: Get-Command python)

# Install a package
python -m pip install <package-name>

# List installed packages
python -m pip list

# Deactivate
deactivate
```
