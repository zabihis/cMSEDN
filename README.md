# cMSEDN: Contextual Multiscale Expected Density of Nucleotide Encoding for Sequence‑Specific Node Embeddings in De Bruijn Graph Learning

## 🧬 Abstract

**Background:** Graph neural networks (GNNs) have become a powerful framework for biological sequence classification, especially when paired with **De Bruijn graphs** that capture local k‑mer connectivity. However, most existing node representations—one‑hot, Word2Vec, or pretrained genomic foundation models—assign **static embeddings** to k‑mers, ignoring the sequence‑specific contexts in which they occur.

**Results:** We introduce **cMSEDN (Contextual Multiscale Expected Density of Nucleotide)**, a lightweight and deterministic adaptation of the multiscale EDN/MSEDN sequence encoder. cMSEDN transforms multiscale sequence‑level EDN representations into **sequence‑specific node embeddings**, enabling identical k‑mers to receive distinct representations depending on their nucleotide‑density environments. Across seven benchmark datasets and two GNN architectures (GCN, GIN), cMSEDN consistently outperforms static encodings and large genomic foundation models while maintaining compact dimensionality and high computational efficiency.

**Conclusions:** cMSEDN demonstrates that adapting multiscale sequence‑level representations into contextual node embeddings yields richer graph representations without modifying the underlying GNN architecture. This repository provides the implementation of cMSEDN, baseline encodings, De Bruijn graph construction.

---

## 🛠 Project Structure

The repository is organized for clarity, reproducibility, and modular experimentation:

```
root/
│   main.py                     # Entry point 
│   requirements.txt            # Python dependencies
│
├── checkpoints/                # Saved pretrained models, node embeddings, and graph objects
│   ├── kmerEmbeddings_DNABERT2_k4.npy
│   ├── kmerEmbeddings_NTv2_k4.npy
│   ├── dglgraph/
│   ├── dglmodel/
│   │   model_GCN_cMSEDN_mouse_tf3_4mer.pkl
│   │   model_GCN_DNABERT2_mouse_tf3_4mer.pkl
│   │   model_GCN_NTv2_mouse_tf3_4mer.pkl
│   │   model_GCN_onehot_mouse_tf3_4mer.pkl
│   │   model_GCN_sMSEDN_mouse_tf3_4mer.pkl
│   │   model_GIN_cMSEDN_mouse_tf3_4mer.pkl
│   │   model_GIN_DNABERT2_mouse_tf3_4mer.pkl
│   │   model_GIN_NTv2_mouse_tf3_4mer.pkl
│   │   model_GIN_onehot_mouse_tf3_4mer.pkl
│   │   model_GIN_sMSEDN_mouse_tf3_4mer.pkl
│   ├── Final_model/
│   └── Node_feature/
│
├── data/                       # All datasets and preprocessing utilities
│   ├── seqProcessing.py
│   ├── SubLoc_BM.txt
│   ├── human_prom_core_tata/
│   │   dev.csv, test.csv, train.csv
│   ├── mouse_tf/, mouse_tf0/, mouse_tf1/, mouse_tf2/, mouse_tf3/, mouse_tf4/
│       dev.csv, test.csv, train.csv
│
├── models/                     # GNN architectures and classifier modules
│   ├── classifier.py
│   └── MLP.py
│
├── output/                     # Logs, predictions, and evaluation results
│
└── utils/                      # Configuration utilities
    ├── config.py
    └── FocalLoss.py
```

---

## 🚀 Installation & Usage

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Run 
```
python main.py
```
Note: edit this file if required.

### 3. Dataset Sources
All datasets used in this project are publicly available:

- **GUE Benchmark (Promoters, TF Binding):**  
  [https://huggingface.co/datasets/leannmlindsey/GUE](https://huggingface.co/datasets/leannmlindsey/GUE)  
- **SubLoc_BM (RNA Localization):**  
  [https://github.com/CSUBioGroup/GraphLncLoc](https://github.com/CSUBioGroup/GraphLncLoc)

### 4. Pretrained Models
Pretrained GCN and GIN models for the node representations are stored under:

```
checkpoints/dglmodel/
```

---

## 📧 Contact

* **Name:** Saman Zabihi  
* **Email:** szabihi@hotmail.com 
* **GitHub:** [https://github.com/zabihis/cMSEDN] (https://github.com/zabihis/cMSEDN)

---

## 📚 Citation

If you use **cMSEDN** or this repository in your research, please cite:

```
[placeholder]
```

**BibTeX:**
```
[placeholder]
```

---

