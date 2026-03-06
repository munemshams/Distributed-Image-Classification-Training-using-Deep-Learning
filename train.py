import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader

import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import AutoModelForImageClassification, AutoImageProcessor
import evaluate

from optimizer_configs import optimizer_configs, optimizer_summary
from evaluate_model import evaluate_model

accelerator = Accelerator()

# Dataset preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

subset = Subset(dataset, range(50))
dataloader = DataLoader(subset, batch_size=16, shuffle=True)

model_name = "microsoft/swin-tiny-patch4-window7-224"

processor = AutoImageProcessor.from_pretrained(model_name)

base_model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=10,
    ignore_mismatched_sizes=True
)

initial_state_dict = base_model.state_dict().copy()

criterion = nn.CrossEntropyLoss()

training_report = {}

best_optimizer = None
best_accuracy = 0.0
num_epochs = 2


for name, opt_fn in optimizer_configs.items():

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=10,
        ignore_mismatched_sizes=True
    )

    model.load_state_dict(initial_state_dict)

    optimizer = opt_fn(model.parameters())

    model, optimizer, data = accelerator.prepare(
        model,
        optimizer,
        dataloader
    )

    model.train()

    for _ in range(num_epochs):

        for inputs, labels in data:

            outputs = model(pixel_values=inputs).logits

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            accelerator.backward(loss)

            optimizer.step()

    accuracy = evaluate_model(model, data)

    training_report[name] = {
        "accuracy": accuracy,
        "training_time": 1.0
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_optimizer = name


training_report["comparison"] = {
    "best_optimizer": best_optimizer
}

print("\nTraining Report Summary")

for name in optimizer_configs:
    r = training_report[name]
    print(f"\n{name} Optimizer:\n- Accuracy: {r['accuracy']*100:.2f}")

print(
    f"\nBest Optimizer after {num_epochs} epochs:",
    training_report["comparison"]["best_optimizer"]
)
