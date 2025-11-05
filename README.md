# 🧠 Deep Learning Beginner Project

This project walks through my first steps in deep learning — from manually coding gradient descent to fine-tuning a transformer for sentiment analysis.

---

## 🚀 Project Overview
I began with a bare-bones PyTorch implementation of **gradient descent** to fit a simple linear curve.  
Then I explored the `torch.nn` module to model **non-linear relationships**.  
Finally, I applied these ideas to a real-world NLP task by **fine-tuning DistilBERT** on the IMDb movie reviews dataset for sentiment classification.

---

## 🧩 Project Phases

### 1️⃣ Manual Gradient Descent (PyTorch from scratch)
- Implemented a single-weight linear regression model  
- Computed gradients with `loss.backward()`  
- Updated weights manually using learning rate and gradient  

### 2️⃣ Non-Linear Function Fitting
- Used `torch.nn.Linear` and `torch.nn.Tanh` layers  
- Trained on a `sin(x)` dataset to visualize how neural nets approximate curves  

### 3️⃣ Text Classification with DistilBERT
- Loaded and tokenized the IMDb dataset using `datasets` + `transformers`
- Fine-tuned `distilbert-base-uncased` for binary sentiment prediction  
- Evaluated with accuracy, precision, recall, and F1-score  

---
## Results 

### 📋 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
|-------------|:----------:|:-------:|:---------:|:--------:|
| **Negative** | 0.87 | 0.82 | 0.84 | 1511 |
| **Positive** | 0.83 | 0.87 | 0.85 | 1489 |
| **Accuracy** |  |  | **0.85** | 3000 |
| **Macro Avg** | 0.85 | 0.85 | 0.85 | 3000 |
| **Weighted Avg** | 0.85 | 0.85 | 0.85 | 3000 |

### Confusion Matrix
![IMDB Confusion Matrix](IMDB_CM%20(1).png)

The model performs symmetrically well on positive and negative reviews, with low misclassification rates.

---

## 🧠 Key Learnings
- Understanding how `loss.backward()` computes gradients internally  
- Seeing how optimizers (`SGD`, `Adam`) abstract manual updates  
- Learning how tokenization turns raw text into numerical tensors  
- Interpreting precision, recall, and F1-scores for classification tasks  
- Managing GPU memory (batch size, FP16) during fine-tuning  

---

## 🛠️ Tech Stack
- **PyTorch** – core deep learning framework  
- **Transformers (Hugging Face)** – pretrained DistilBERT model  
- **Datasets** – IMDb text dataset  
- **Matplotlib / Scikit-learn** – evaluation plots and metrics  
