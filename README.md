# Distributed Image Classification Training using Vision Transformers

This project demonstrates **distributed deep learning training using PyTorch, Hugging Face Transformers, and the Accelerate library**. The goal is to train a **pretrained Vision Transformer model** on the CIFAR-10 dataset while comparing the performance of multiple optimizers.

The project explores how distributed training pipelines can improve the efficiency and scalability of deep learning models used in real-world AI systems.

---

# Project Overview

NeuScale Innovations aims to accelerate deep learning workflows by leveraging **distributed computing techniques**. As deep learning models become larger and more complex, efficient training strategies become critical.

In this project:

- A **pretrained Swin Transformer model** is used for image classification
- Training is implemented using the **Accelerate library for distributed training**
- Multiple optimizers are tested and compared
- Model performance is evaluated after training
- The best optimizer configuration is identified

This project demonstrates how distributed training can improve the scalability of machine learning workflows.

---

# Dataset

The project uses the **CIFAR-10 dataset**, a well-known benchmark dataset for image classification.

CIFAR-10 contains **60,000 color images** of size **32×32 pixels** categorized into **10 classes**.

| Class | Label |
|------|------|
Airplane | 0 |
Automobile | 1 |
Bird | 2 |
Cat | 3 |
Deer | 4 |
Dog | 5 |
Frog | 6 |
Horse | 7 |
Ship | 8 |
Truck | 9 |

For faster experimentation, this project trains on a **small subset of 50 images**.

The dataset is downloaded automatically using `torchvision.datasets`.

---

# Model

The model used in this project is a **pretrained Swin Transformer**:


microsoft/swin-tiny-patch4-window7-224


The model is loaded using the **Hugging Face Transformers library** and adapted to classify the **10 CIFAR-10 categories**.

Vision Transformers apply **self-attention mechanisms to image data**, allowing the model to capture global image relationships more effectively than traditional convolutional networks.

---

# Distributed Training

Distributed training is implemented using the **Accelerate library**.

Accelerate simplifies multi-device training by automatically preparing the model, optimizer, and dataloader for distributed execution.

This allows the same code to run across different hardware environments, including:

- CPUs
- GPUs
- multi-device training systems

---

# Optimizers Compared

The project compares three different optimizers:

| Optimizer | Learning Rate | Weight Decay |
|------|------|------|
SGD | 0.01 | 0 |
Adam | 0.001 | 1e-4 |
AdamW | 0.001 | 1e-4 |

These optimizer configurations are stored in the dictionary:


optimizer_summary


Each optimizer trains the model independently so their performance can be evaluated and compared.

---

# Training Process

For each optimizer:

1. Reset model weights  
2. Train the model for **2 epochs**  
3. Compute loss using **CrossEntropyLoss**  
4. Perform backpropagation  
5. Evaluate model accuracy  

Training results are stored in a dictionary named:


training_report


The optimizer that achieves the **highest accuracy** is selected as the best-performing configuration.

---

# Files Included

| File | Description |
|-----|-------------|
train.py | Main script implementing distributed model training |
evaluate_model.py | Model evaluation function |
optimizer_configs.py | Optimizer configurations |
README.md | Project documentation |

---

# Dependencies

Install the required Python packages before running the project:

```bash
pip install torch torchvision transformers accelerate evaluate
Running the Project

Run the training pipeline using:

python train.py

The script will:

Download the dataset if needed

Load the pretrained model

Train the model using three optimizers

Evaluate model accuracy

Print a summary of training results

Identify the best optimizer

First Run Note

When running the project for the first time, the following resources will be downloaded automatically:

CIFAR-10 dataset

Pretrained Swin Transformer model from Hugging Face

This may take a few minutes depending on your internet connection. After the first run, the files will be cached locally.

---

# Outputs

Running the project generates a training summary printed to the console, including:

accuracy for each optimizer

optimizer comparison results

best optimizer after training

# Notes

Because the training subset contains only 50 images, the model will not achieve high accuracy or learn meaningful patterns. This lightweight setup is intended for experimentation and demonstration purposes.

To obtain more realistic results, you can increase:

the dataset size

the number of training epochs

the training batch size
