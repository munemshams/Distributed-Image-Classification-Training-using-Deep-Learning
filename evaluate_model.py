import torch
import evaluate
from accelerate import Accelerator

accelerator = Accelerator()

def evaluate_model(model, dataloader):
    metric = evaluate.load("accuracy")
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(pixel_values=inputs).logits
            preds = outputs.argmax(dim=1)
            preds, labels = accelerator.gather_for_metrics((preds, labels))
            metric.add_batch(predictions=preds, references=labels)

    return metric.compute()["accuracy"]
