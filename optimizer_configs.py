import torch.optim as optim

optimizer_configs = {
    "SGD": lambda p: optim.SGD(p, lr=0.01, momentum=0.9, weight_decay=0.0),
    "Adam": lambda p: optim.Adam(p, lr=0.001, weight_decay=1e-4),
    "AdamW": lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4),
}

optimizer_summary = {
    "SGD": {"learning_rate": 0.01, "weight_decay": 0.0},
    "Adam": {"learning_rate": 0.001, "weight_decay": 1e-4},
    "AdamW": {"learning_rate": 0.001, "weight_decay": 1e-4},
}
