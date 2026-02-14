# Anchored Email Summarization with DistilBART

This repository contains the implementation for fine-tuning **DistilBART-CNN-12-6** on the **MailEx** dataset (a subset of Enron). Our approach leverages **Knowledge Distillation** from GPT-4 and **Feature Priming** (using event triggers and arguments) to generate concise, action-oriented email summaries.

## 📌 Project Overview

Traditional summarization often misses actionable "events" in long email threads. This project uses structured event metadata to "anchor" the model, ensuring summaries focus on intents like `Request_Action` or `Request_Data`.

- **Model:** DistilBART (300M parameters)
- **Dataset:** MailEx (Enron subset)
- **Methodology:** Feature Priming + Knowledge Distillation

## 🛠️ Installation

```bash
pip install torch transformers peft accelerate bitsandbytes
```

## 🚀 Usage
(see ipynb).

## 🧬 Methodology
### Feature Priming (Anchoring)
Instead of feeding raw text, we prepend identified event triggers to the input. This guides the model's attention mechanism to focus on relevant tokens.Input Format:summarize: [Request_Action: we need] [Request_Data: what about] THREAD: Email 0: we need a fourth again Email 1: what about hull?Knowledge DistillationDue to the lack of gold-standard summaries in the original MailEx set, we used GPT-4 to generate "silver-label" summaries. These professional summaries served as the ground truth for fine-tuning our lightweight DistilBART model.
### 📊Hyperparameters
- Base Modelsshleifer/distilbart-cnn-12-6
- Learning Rate: 5e-5
- Batch Size: 4
- Epochs: 5
- Beam Searchk: 5

## 📚 Citations
If you use this code or the MailEx summaries, please cite:

@article{shleifer2020pre,
  title={Pre-trained Summarization Distillation},
  author={Shleifer, Sam and Lewis, Mike},
  journal={arXiv preprint arXiv:2010.13002},
  year={2020}
}

@misc{enron_dataset,
  author = {Enron Corp. and Cohen, William W.},
  title = {Enron Email Dataset},
  year = {2015},
  url = {[https://www.loc.gov/item/2018487913/](https://www.loc.gov/item/2018487913/)}
}
