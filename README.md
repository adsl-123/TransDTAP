# TransDTAP

## A Multimodal Transformer Architecture for Drug–Target Affinity Prediction

Official implementation and reproducibility repository for:

**TransDTAP: A Multimodal Transformer Architecture for Drug–Target Affinity Prediction Using Sequence and Biochemical Properties**
*Computational Biology and Chemistry (Under Revision)*

---

## 🔬 Overview

TransDTAP is a multimodal deep learning framework designed to predict drug–target binding affinity (pIC₅₀) by integrating complementary information sources:

* **SMILES ligand sequences** (Transformer encoder)
* **Protein amino-acid sequences** (Transformer encoder)
* **Molecular physicochemical descriptors**
* **Protein biochemical features**

The architecture employs a four-branch encoding strategy with **late multimodal fusion** and a regression head for affinity prediction.

The model is designed for scalable virtual screening without requiring molecular graph construction or three-dimensional protein structures.

---

## 📊 Dataset Summary

The curated dataset used in the study includes:

* **4,793** protein targets
* **23,531** unique ligands
* **33,457** experimentally measured IC₅₀ interactions
* Target type: *Single protein* (ChEMBL)
* Protein sequences: UniProt canonical entries
* Assay type: Binding assays
* Units: Nanomolar (nM)

Binding affinities are transformed as:

[
pIC_{50} = 9 - \log_{10}(IC_{50},[nM])
]

Both data splitting strategies are provided:

* **Random split (70/15/15)**
* **Bemis–Murcko scaffold-based split**

---

## 📁 Repository Structure

```
TransDTAP/
│
├── data/
│   ├── processed_dataset.csv
│   ├── random_split_indices.json
│   ├── scaffold_split_indices.json
│
├── preprocessing/
│   ├── descriptor_generation.py
│   ├── protein_feature_extraction.py
│   ├── tokenization.py
│
├── models/
│   ├── transformer_encoder.py
│   ├── descriptor_encoder.py
│   ├── multimodal_model.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│
├── experiments/
│   ├── default_config.yaml
│   ├── ablation_configs.yaml
│
├── interpretability/
│   ├── shap_analysis.py
│
├── results/
│   ├── loss_curves.png
│   ├── validation_r2.png
│   ├── true_vs_pred.png
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/adsl-123/TransDTAP.git
cd TransDTAP
```

### 2️⃣ Install dependencies

Python ≥ 3.9 recommended.

```
pip install -r requirements.txt
```

Main dependencies:

* PyTorch
* RDKit
* Scikit-learn
* SHAP
* NumPy
* Pandas
* Matplotlib

---

## 🚀 Reproducing Experiments

### Training the full model

```
python training/train.py --config experiments/default_config.yaml
```

### Running evaluation

```
python training/evaluate.py --checkpoint path/to/best_model.pt
```

### Running ablation experiments

```
python training/train.py --config experiments/ablation_configs.yaml
```

---

## 🧠 Model Architecture

TransDTAP consists of four parallel branches:

1. **SMILES Transformer Encoder**

   * Embedding dimension: 128
   * 2 Transformer layers
   * 4 attention heads
   * Attention-based pooling

2. **Protein Transformer Encoder**

   * Same configuration as ligand encoder
   * Maximum length: 1024 residues

3. **Molecular Descriptor Encoder**

   * 10-dimensional input
   * Fully connected projection (128-dim)

4. **Protein Biochemical Feature Encoder**

   * 8-dimensional input
   * Fully connected projection (128-dim)

The four representations are concatenated (late fusion) and passed to a two-layer MLP regression head.

Loss function: **Smooth L1 (Huber)**
Optimizer: **AdamW**
Scheduler: **ReduceLROnPlateau**
Regularization: Dropout, weight decay, gradient clipping

---

## 📈 Reported Performance

### Random Split (70/15/15)

* **R² = 0.7794**
* **MSE = 0.2142**
* **MAE (pIC₅₀) = 0.2827**
* **MAE (nM) = 0.8556**

### Scaffold-Based Split

* **R² = 0.689**
* **MSE = 0.253**
* **MAE = 0.325**

These results demonstrate robust generalization beyond scaffold memorization.

---

## 🔬 Ablation Study

The repository includes configurations for:

* Sequence-only model
* Descriptor-only model
* No molecular descriptors
* No protein descriptors

Results confirm that multimodal integration provides additive predictive improvement.

---

## 🔍 Interpretability

SHAP analysis is provided for descriptor branches to quantify:

* Molecular feature importance (logP, TPSA, molecular weight, etc.)
* Protein biochemical feature importance (GRAVY score, pI, net charge, etc.)

Run:

```
python interpretability/shap_analysis.py
```

---

## 🔁 Reproducibility

This repository includes:

* Processed dataset
* Exact split indices
* Tokenization logic
* Descriptor normalization pipeline (RobustScaler)
* Hyperparameter configurations
* Ablation settings
* Evaluation scripts

All reported results can be reproduced using the provided configurations.

---

## 🔐 Data Sources

The dataset was constructed from publicly available databases:

* ChEMBL
* UniProt

No proprietary or confidential data were used.

Users are responsible for complying with ChEMBL and UniProt licensing terms when reusing raw data.

---

## 📜 Citation

If you use this repository, please cite:

Bouguessa A., Mostefaoui S.A.M., Daoud M.A.
**TransDTAP: A Multimodal Transformer Architecture for Drug–Target Affinity Prediction Using Sequence and Biochemical Properties**
Computational Biology and Chemistry.

---

## 📬 Contact

**Corresponding Author**
Abdelkader Bouguessa
University of Tiaret – LRIAS Laboratory
Email: [abdelkader.bouguessa@univ-tiaret.dz](mailto:abdelkader.bouguessa@univ-tiaret.dz)

---

## 📄 License

This repository is released for research and academic purposes.
Please cite the associated manuscript when using this work.

