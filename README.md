# 🧠 Sorting Hat Classifier  
> Logistic Regression Project — Hogwarts Edition  
> Built for the 42 ML module

---

## 📚 Overview

This project implements a **multiclass logistic regression classifier** from scratch to sort Hogwarts students into houses based on their course performance. It fulfills **all mandatory requirements** and completes the **bonus tasks** listed in the subject.

You can train the model using **Batch**, **Stochastic**, or **Mini-Batch Gradient Descent**, visualize training performance, and explore dataset statistics through custom scripts.

---

## ✅ Features

### 🎯 Mandatory
- Custom `describe.py`: manually computes basic stats:
  - Count, Mean, Std, Min, 25%, 50%, 75%, Max
- Logistic Regression (One-vs-All)
- Feature normalization & NaN handling
- Saves weights to `weights.csv`
- CLI usability for training and inspecting data

### 🎁 Bonus
- `--bonus` mode in `describe.py`:
  - Adds Range, Mode, Skewness, Kurtosis
- `--mode sgd`, `--mode minibatch` support in training
- Visualizations:
  - Histograms
  - Pair plots by house
  - Scatter plots for most correlated features
- Plots training loss for each class over time
  - Saved as `cost_plot_<mode>.png`
- Fully reproducible training + prediction flow

---

## 📁 Project Structure

```
sorting-hat/
├── dataset_train.csv          # Training dataset
├── dataset_test.csv           # Unlabeled test dataset
├── describe.py                # Manual describe() tool
├── histogram.py               # Histograms by feature
├── scatter_plot.py            # Most correlated feature scatter
├── pair_plot.py               # Full seaborn-style pair plot
├── logreg_train.py            # Logistic Regression training
├── logreg_predict.py          # Makes predictions using weights
├── weights.csv                # Saved model weights
├── houses.csv                 # Prediction output file
├── cost_plot_batch.png        # Training loss (Batch GD)
├── cost_plot_sgd.png          # Training loss (SGD)
├── cost_plot_minibatch.png    # Training loss (Mini-Batch)
├── requirements.txt           # Python dependencies
├── setup_project.sh           # Shell script to set up virtual env
└── README.md                  # This file
```

---

## 🚀 Usage

### 0. Quick Setup

To set up the project environment and install dependencies:

```bash
bash setup_project.sh
source .venv/bin/activate
```

---

### 1. Train the Model

```bash
python logreg_train.py dataset_train.csv              # Default (batch)
python logreg_train.py dataset_train.csv --mode sgd   # Stochastic GD
python logreg_train.py dataset_train.csv --mode minibatch  # Mini-batch GD
```

Weights are saved to `weights.csv`. Cost plots are saved per mode.

---

### 2. Describe the Dataset

```bash
python describe.py dataset_train.csv 
python describe.py dataset_train.csv --bonus
```

---

### 3. Predict Hogwarts Houses

```bash
python logreg_predict.py dataset_test.csv weights.csv
```

Predictions are written to `houses.csv`.

---

### 4. Visualize Data

```bash
python histogram.py dataset_train.csv
python pair_plot.py dataset_train.csv
python scatter_plot.py dataset_train.csv
```

---

## 🖼️ Sample Output

- **Loss Plots**:
  - `cost_plot_batch.png`
  - `cost_plot_sgd.png`
  - `cost_plot_minibatch.png`

- **Predictions**:
  - `houses.csv`

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
numpy
pandas
matplotlib
seaborn
```

Or simply run:
```bash
bash setup_project.sh
```

---

## 🏁 Notes

- Project built from scratch — **no scikit-learn**
- All gradient methods tested and validated
- Bonus-ready and submission-ready
- Follows 42 school's coding principles

---

## 📦 Submission Tips

To submit:

1. Make sure all `.py` scripts are working.
2. Clear any large cache files, `.ipynb_checkpoints`, etc.
3. Zip the folder:
   ```bash
   zip -r sorting-hat.zip sorting-hat/
   ```

---

✨ Good luck, and may the Hat sort you wisely.