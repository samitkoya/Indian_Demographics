# Requirements & Installation Guide

## Overview

This document provides step-by-step instructions to install all required dependencies for the UIDAI Aadhaar Analysis project.

---

## 📋 System Requirements

### Python Version

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10 or higher

### Check Your Python Version

Open a terminal and run:

```bash
python --version
```

or

```bash
python3 --version
```

Expected output: `Python 3.x.x`

---

## 📦 Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.5.0 | Data manipulation and analysis |
| numpy | >= 1.21.0 | Numerical computations |
| matplotlib | >= 3.5.0 | Data visualization |
| seaborn | >= 0.11.0 | Statistical visualizations |
| scikit-learn | >= 1.0.0 | Machine learning (clustering, regression) |
| scipy | >= 1.7.0 | Statistical functions |

---

## 🔧 Installation Commands

### Option 1: Install All Libraries at Once

Open your terminal (Command Prompt, PowerShell, or Terminal) and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Option 2: Install Libraries Individually

If you prefer to install each library separately:

```bash
# Install pandas for data manipulation
pip install pandas

# Install numpy for numerical operations
pip install numpy

# Install matplotlib for basic plotting
pip install matplotlib

# Install seaborn for statistical visualizations
pip install seaborn

# Install scikit-learn for machine learning
pip install scikit-learn

# Install scipy for statistical functions
pip install scipy
```

### Option 3: Using requirements.txt

Create a file named `requirements.txt` with the following content:

```text
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

Then install using:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Platform-Specific Instructions

### Windows

1. Open **Command Prompt** or **PowerShell**
2. Run the installation command:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

If `pip` is not recognized, try:

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

### macOS

1. Open **Terminal**
2. Run the installation command:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

### Linux (Ubuntu/Debian)

1. Open **Terminal**
2. First, ensure pip is installed:

```bash
sudo apt update
sudo apt install python3-pip
```

3. Then install the libraries:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## ✅ Verify Installation

After installation, verify that all libraries are installed correctly:

```bash
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"
python -c "import seaborn; print(f'seaborn: {seaborn.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import scipy; print(f'scipy: {scipy.__version__}')"
```

Or run a single verification script:

```python
# Save as verify_install.py and run: python verify_install.py
libraries = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scipy']

print("Checking installed libraries...")
print("-" * 40)

for lib in libraries:
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'Version not found')
        print(f"✓ {lib}: {version}")
    except ImportError:
        print(f"✗ {lib}: NOT INSTALLED")

print("-" * 40)
print("Verification complete!")
```

---

## 🏃 Running the Scripts

After installation, navigate to the scripts folder and run the analysis:

### Step 1: Navigate to Scripts Directory

```bash
cd path/to/UIDAI_new/scripts
```

**Example for Windows:**
```powershell
cd C:\Users\Samit Reddy\Desktop\UIDAI\UIDAI_new\scripts
```

**Example for macOS/Linux:**
```bash
cd ~/Desktop/UIDAI/UIDAI_new/scripts
```

### Step 2: Run Main Analysis

```bash
python data_cleaning.py
```

**Expected Output:**
```
================================================================================
       UIDAI AADHAAR ENROLLMENT & UPDATES ANALYSIS
       Unlocking Societal Trends in Aadhaar Data
================================================================================

[STEP 1] Loading datasets...
----------------------------------------

  Loading Biometric Update data...
  Loading: api_data_aadhar_biometric_0_500000.csv
  ...
  Total records: 1,861,108

[STEP 2] Preprocessing data...
----------------------------------------
  [OK] Biometric data: 1,861,108 records, 36 states
  ...
```

### Step 3: Run Advanced Analytics

```bash
python advanced_analytics.py
```

**Expected Output:**
```
================================================================================
       UIDAI AADHAAR - ADVANCED ANALYTICS
       Predictive Indicators & Pattern Analysis
================================================================================

[LOAD] Loading datasets...
  [OK] Loaded 1,861,108 biometric, 2,071,700 demographic, 1,006,029 enrollment records

[CLUSTER] Performing state clustering analysis...
  [OK] State clustering complete - 4 clusters identified
...
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: pip not found

**Error:**
```
'pip' is not recognized as an internal or external command
```

**Solution:**
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

### Issue 2: Permission denied

**Error:**
```
Permission denied: '/usr/local/lib/python3.x/...'
```

**Solution:**
```bash
pip install --user pandas numpy matplotlib seaborn scikit-learn scipy
```

---

### Issue 3: SSL Certificate error

**Error:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org pandas numpy matplotlib seaborn scikit-learn scipy
```

---

### Issue 4: Old pip version

**Solution:**
```bash
python -m pip install --upgrade pip
```

---

## 📁 Project File Check

Before running, ensure your project has this structure:

```
UIDAI_new/
├── scripts/
│   ├── data_cleaning.py    ✓
│   ├── analysis.py         ✓
│   └── advanced_analytics.py ✓
├── data/
│   └── *.csv files         ✓ (12 files)
├── images/
│   └── (will be created)
└── docs/
    └── (documentation)
```

---

## 🆘 Need Help?

If you encounter any issues:

1. Check that Python is installed correctly
2. Verify all libraries are installed using the verification script
3. Ensure you're in the correct directory when running scripts
4. Check that CSV data files are present in the `data/` folder
