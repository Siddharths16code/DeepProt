# 🧬 DeepProt
## Protein Function Prediction Using Deep Learning

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://deepprot.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

**DeepProt** is a deep-learning-based bioinformatics project that predicts potential Gene Ontology (GO) annotations from protein amino-acid sequences using a trained Multi-Kernel Convolutional Neural Network (CNN).

The application allows users to enter a protein sequence or upload a FASTA file and obtain predicted GO terms with their corresponding model scores.

**Live Demo:** https://deepprot.streamlit.app/  
**GitHub Repository:** https://github.com/Siddharths16code/DeepProt

---

## 1. Project Overview

Proteins perform essential biological functions, including catalysis, transport, structural support, regulation, and cellular signaling.

Their amino-acid sequences contain information that can help computational models identify potential biological functions.

DeepProt uses a Multi-Kernel CNN to learn sequence patterns and predict associated Gene Ontology terms.

The task is formulated as **multi-label classification**, because a single protein may be associated with multiple biological functions.

The model predicts 226 selected GO terms belonging to three major namespaces:

- **Biological Process (BP):** Biological processes in which a protein participates.
- **Cellular Component (CC):** Cellular locations or complexes associated with a protein.
- **Molecular Function (MF):** Molecular activities performed by a protein.

## 2. Motivation

Experimental characterization of protein functions can be time-consuming and resource-intensive.

Computational prediction methods can help generate functional hypotheses for further biological investigation.

The objectives of DeepProt are:

1. To develop a deep-learning-based protein function prediction system.
2. To understand the relationship between amino-acid sequences and protein functions.
3. To implement multi-label classification using a CNN architecture.
4. To investigate class imbalance in Gene Ontology annotations.
5. To compare different loss-weighting strategies.
6. To evaluate model performance using validation and independent test data.
7. To develop a web application for interactive protein function prediction.

## 3. Key Features

- Protein sequence input through a text area.
- FASTA file upload support.
- Supports 20 standard amino acids.
- Maximum processed sequence length of 3,000 residues.
- Prediction of 226 selected GO terms.
- GO IDs, GO names and GO namespaces.
- Model scores for predicted annotations.
- Complete GO-term score table.
- CSV export of prediction results.
- Publicly accessible Streamlit application.

## 4. System Workflow

```text
Protein Amino-Acid Sequence
             |
             v
      Sequence Validation
             |
             v
       FASTA Parsing
             |
             v
      Amino-Acid Encoding
             |
             v
    Padding / Truncation
       (Length = 3000)
             |
             v
       Embedding Layer
        (32 Dimensions)
             |
       +-----+-----+
       |     |     |
       v     v     v
     Conv5  Conv9  Conv15
       |     |     |
       +-----+-----+
             |
             v
    Masked Global Max Pooling
             |
             v
      Feature Concatenation
        (96 Dimensions)
             |
             v
       Fully Connected
          (64 Units)
             |
             v
          Dropout
           (0.4)
             |
             v
       Output Layer
        (226 Logits)
             |
             v
       Sigmoid Scores
             |
             v
    Threshold Classification
         (0.46)
             |
             v
       Predicted GO Terms
```

## 5. Dataset Description

The project uses a pilot dataset containing:

- 500 reviewed human protein sequences.
- Protein functional annotations processed from a Gene Association File (GAF).
- Gene Ontology metadata from `go-basic.obo`.

### Dataset statistics

| Parameter | Value |
|---|---:|
| Total proteins | 500 |
| Unique positive protein-GO associations | 10,012 |
| Unique GO terms before selection | 3,860 |
| Positive-negative conflicts removed | 3 |
| MMseqs2 sequence clusters | 484 |
| Selected GO terms | 226 |
| Maximum sequence length | 3,000 |

### Dataset splitting

Protein sequences were clustered using MMseqs2 before dataset splitting.

Clustering configuration:

- Minimum sequence identity: 0.30
- Coverage: 0.80
- Coverage mode: 0

A cluster-aware split was used to reduce sequence-similarity leakage between partitions.

| Dataset | Number of proteins |
|---|---:|
| Training | 351 |
| Validation | 49 |
| Testing | 100 |
| **Total** | **500** |

No cluster overlap was observed between the three partitions.

## 6. Data Preprocessing

### 6.1 Amino-acid encoding

The 20 standard amino acids were converted into integer representations.

Alphabet:

`ACDEFGHIKLMNPQRSTVWY`

Each amino acid was assigned an integer ID from 1 to 20.

- 0 represents padding.
- 1–20 represent standard amino acids.

### 6.2 Sequence length normalization

All sequences were represented using a maximum length of 3,000 residues.

- Shorter sequences were padded.
- Longer sequences were truncated.

### 6.3 GO vocabulary selection

Only GO terms having at least five positive training proteins were selected.

This resulted in a final vocabulary of 226 GO terms.

The vocabulary was derived from the training dataset to avoid using validation or test information during label selection.

### 6.4 Target matrix generation

Each protein was represented by a 226-dimensional multi-hot target vector.

| Matrix | Shape |
|---|---|
| Y_train | 351 × 226 |
| Y_val | 49 × 226 |
| Y_test | 100 × 226 |

## 7. Model Architecture

DeepProt uses a **Multi-Kernel Convolutional Neural Network (CNN)** implemented using PyTorch.

### Architecture details

| Layer | Configuration |
|---|---|
| Embedding | 32 dimensions |
| Conv1D Branch 1 | 32 filters, kernel size 5 |
| Conv1D Branch 2 | 32 filters, kernel size 9 |
| Conv1D Branch 3 | 32 filters, kernel size 15 |
| Activation | ReLU |
| Pooling | Masked Global Max Pooling |
| Concatenation | 96 features |
| Fully Connected Layer | 64 units |
| Dropout | 0.4 |
| Output Layer | 226 units |

**Total trainable parameters: 51,362**

### Why Multi-Kernel CNN?

Different kernel sizes allow the network to learn local sequence patterns over different sequence-window widths.

The three convolution branches operate in parallel, and their extracted features are combined before the final classification layer.

This provides a compact architecture for sequence-based multi-label prediction.

## 8. Algorithms and Techniques Used

| Algorithm / Technique | Purpose |
|---|---|
| MMseqs2 Clustering | Protein sequence similarity clustering |
| Integer Encoding | Numerical representation of amino acids |
| Embedding Layer | Learnable amino-acid representations |
| 1D Convolution | Extract local sequence patterns |
| ReLU | Non-linear activation |
| Global Max Pooling | Fixed-size feature extraction |
| Dropout | Regularization |
| BCEWithLogitsLoss | Multi-label classification loss |
| Weighted BCE | Handling class imbalance |
| Adam Optimizer | Model optimization |
| Early Stopping | Reduce overfitting |
| Sigmoid | Convert logits into scores |
| Threshold Classification | Generate predicted GO labels |

## 9. Experimental Configurations

Four model configurations were investigated.

| Experiment | Configuration |
|---|---|
| A | Weighted BCE |
| B | Unweighted BCE |
| C | Moderate Weighted BCE |
| D | Multi-Kernel CNN with Weighted BCE |

### Experiment A — Weighted BCE

Positive-class weights were calculated using the ratio of negative to positive examples and capped at 10.

- Best validation loss: 0.6334
- Selected threshold: 0.44

### Experiment B — Unweighted BCE

Standard binary cross-entropy was used without additional positive-class weighting.

- Best validation loss: 0.1402
- Selected threshold: 0.20

### Experiment C — Moderate Weighting

Square-root-based positive weights were used and capped at 3.

- Best validation loss: 0.2956
- Selected threshold: 0.32

### Experiment D — Multi-Kernel CNN

The final architecture used three parallel convolution branches with kernel sizes 5, 9 and 15.

- Best validation loss: 0.5439
- Best checkpoint: Epoch 12
- Selected threshold: 0.46

## 10. Validation Results

All thresholds were selected using validation data only.

| Model | Threshold | Micro F1 | Macro F1 | Precision | Recall | Micro AUPR | Macro AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| A - Weighted BCE | 0.44 | 0.2507 | 0.0253 | 0.1830 | 0.3981 | 0.1166 | 0.1137 |
| B - Unweighted BCE | 0.20 | 0.3071 | 0.0179 | 0.3441 | 0.2773 | 0.2095 | 0.0927 |
| C - Moderate Weighting | 0.32 | 0.3018 | 0.0178 | 0.3051 | 0.2986 | 0.2104 | 0.0964 |
| D - Multi-Kernel CNN | 0.46 | 0.2755 | 0.0220 | 0.2476 | 0.3104 | 0.1408 | 0.1163 |

### Model Selection

Model D was selected provisionally because it achieved:

- Highest validation Macro AUPR: 0.1163.
- Highest validation Macro F1: 0.0220.
- Strongest support-stratified ranking performance.

Model B achieved the highest Micro F1, while Model C achieved the highest Micro AUPR.

Therefore, Model D was selected based on the project's emphasis on ranking performance across GO terms, not because it performed best on every metric.

## 11. Final Test Results

The selected model was evaluated on the untouched test dataset using the fixed threshold of 0.46.

| Metric | Test Result |
|---|---:|
| Micro F1 | **0.3256** |
| Macro F1 | 0.0232 |
| Precision | 0.2844 |
| Recall | 0.3807 |
| Micro AUPR | 0.1924 |
| Macro AUPR | 0.0742 |

Additional observations:

- Predicted GO terms: 22 out of 226.
- Average predicted labels per protein: 10.55.
- Average observed labels per protein: 7.88.

The model shows a tendency toward overprediction at the selected threshold.

### Namespace-wise analysis

| Namespace | Precision | Recall | F1 | Micro AP |
|---|---:|---:|---:|---:|
| Biological Process | 0.1102 | 0.0774 | 0.0909 | 0.0421 |
| Cellular Component | 0.2463 | 0.5025 | 0.3306 | 0.1851 |
| Molecular Function | 0.6822 | 0.3929 | 0.4986 | 0.4408 |

Molecular Function achieved the strongest performance in this pilot evaluation.

## 12. Technology Stack

| Technology | Application |
|---|---|
| Python 3.11.18 | Main programming language |
| PyTorch 2.15.0 | Deep learning model |
| NumPy 2.4.6 | Numerical computation |
| Pandas 3.1.1 | Dataset and annotation processing |
| Streamlit 1.67.0 | Web application |
| MMseqs2 | Protein sequence clustering |
| Jupyter Notebook | Model development and experimentation |
| Git | Version control |
| GitHub | Source code hosting |
| Streamlit Community Cloud | Public deployment |

## 13. Application Features

The Streamlit application provides two ways to submit protein sequences:

1. Directly paste a protein sequence.
2. Upload a FASTA file.

The application then:

1. Validates the input.
2. Extracts the sequence if a FASTA file is uploaded.
3. Converts the sequence into numerical tokens.
4. Loads the trained CNN model.
5. Generates scores for 226 GO terms.
6. Applies the selected threshold.
7. Displays predicted GO annotations.
8. Allows CSV downloads.

### Application URLs

**Live Application:**  
https://deepprot.streamlit.app/

**GitHub Repository:**  
https://github.com/Siddharths16code/DeepProt

## 14. Repository Structure

```text
DeepProt/
│
├── app.py
├── inference.py
├── model.py
├── requirements.txt
├── .gitignore
│
├── models/
│   └── deepprot_multikernel_weighted_best.pth
│
└── data/
    └── processed/
        └── go_vocabulary_aligned.csv
```

### Important Files

- `app.py`: Streamlit application.
- `model.py`: Multi-Kernel CNN architecture.
- `inference.py`: Standalone prediction pipeline.
- `requirements.txt`: Python dependencies.
- `models/`: Selected trained model checkpoint.
- `data/processed/`: Aligned GO vocabulary.

## 15. How to Run Locally

### Step 1: Clone repository

```bash
git clone https://github.com/Siddharths16code/DeepProt.git

cd DeepProt
```

### Step 2: Create environment

```bash
python3.11 -m venv .venv

source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run application

```bash
streamlit run app.py
```

The application will open locally in the browser.

## 16. Limitations

DeepProt is an experimental prototype and has several limitations:

1. The dataset contains only 500 reviewed human proteins.
2. Only 226 selected GO terms can be predicted.
3. GO annotations are incomplete, so missing annotations cannot be treated as confirmed biological negatives.
4. Rare GO terms remain difficult to predict.
5. The model tends to overpredict labels at the selected threshold.
6. The model uses sequence-based features without explicit protein structure or evolutionary information.
7. Sigmoid outputs are model scores, not calibrated probabilities.
8. Sequences longer than 3,000 residues are truncated.
9. The model has not undergone independent experimental biological validation.

Predictions should be treated as computational hypotheses rather than experimentally confirmed annotations.

## 17. Future Scope

Potential future improvements include:

- Expanding the protein dataset.
- Incorporating protein language model embeddings such as ESM.
- Exploring transformer-based architectures.
- Using GO hierarchy-aware learning.
- Improving rare-label prediction.
- Applying positive-unlabelled learning.
- Adding confidence calibration and uncertainty estimation.
- Supporting batch protein prediction.
- Evaluating on larger independent test datasets.
- Improving model interpretability.

## 18. Conclusion

DeepProt demonstrates an end-to-end deep-learning workflow for protein function prediction.

The project combines bioinformatics data processing, protein sequence encoding, multi-label classification, Multi-Kernel CNN architecture, model comparison, evaluation, standalone inference, and public deployment.

The final selected model achieved a test Micro F1 score of 0.3256 and Micro AUPR of 0.1924.

Although the results show promising pilot-level ranking performance in some GO namespaces, further work is required to improve rare-label prediction, reduce overprediction, and establish biological generalization.

DeepProt provides a foundation for further exploration of deep learning in computational biology and bioinformatics.

## 19. References

1. Ashburner, M. et al. (2000). Gene Ontology: tool for the unification of biology. *Nature*, 405, 193–197. https://doi.org/10.1038/35011509

2. The Gene Ontology Consortium (2021). The Gene Ontology resource: enriching a GOld mine. *Nucleic Acids Research*, 49(D1), D325–D334. https://doi.org/10.1093/nar/gkaa1113

3. UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase in 2023. *Nucleic Acids Research*, 51(D1), D523–D531. https://doi.org/10.1093/nar/gkac1052

4. Steinegger, M. and Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology*, 35, 1026–1028. https://doi.org/10.1038/nbt.3988

5. Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980

6. PyTorch Documentation: https://docs.pytorch.org/

7. Streamlit Documentation: https://docs.streamlit.io/

---

## Author

**Siddharth**  
B.Tech — Computer Engineering With A Minor Degree in Cyber Security
Shri Vishwakarma Skill University (SVSU)

**Project:** DeepProt  
**GitHub:** https://github.com/Siddharths16code/DeepProt  
**Live Demo:** https://deepprot.streamlit.app/
