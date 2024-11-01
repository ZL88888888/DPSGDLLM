import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from datasets import load_dataset

def calculate_noise_scale(sensitivity, epsilon, delta):
    """ Calculate the noise """
    return (sensitivity * torch.sqrt(2 * torch.log(torch.tensor(1.25 / delta)))) / epsilon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(device)

# Load and preprocess the SST-2 dataset
dataset = load_dataset('glue', 'sst2')
def tokenize_and_format(example):
    """ Tokenize sentences and adjust batch dimensions. """
    tokens = tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=512)
    return {k: torch.tensor(v) for k, v in tokens.items()}  # Ensure output is tensor

dataset = dataset.map(tokenize_and_format, batched=True)
dataset = dataset.map(lambda example: {'labels': example['label']}, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=8, shuffle=False)

# Define hyperparameters and privacy parameters
learning_rate = 2e-5
epochs = 3
epsilon = 8
delta = 1e-5
C = 10  # Clipping threshold
beta1, beta2 = 0.9, 0.999
alpha = 0.9

# Calculate noise scale
sensitivity = C
noise_scale = calculate_noise_scale(sensitivity, epsilon, delta)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop using ANADP algorithm
for epoch in range(epochs):
    model.train()
    St_prev = torch.tensor(0.)
    Ut_prev = torch.tensor(0.)
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Compute gradients and their sensitivities
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if gradients:
            St = torch.norm(torch.stack([torch.norm(grad.detach(), 1) for grad in gradients]), 1)
            St_bar = beta1 * St_prev + (1 - beta1) * St
            Ut = beta2 * Ut_prev + (1 - beta2) * (St_bar - St).abs()

            # Calculate importance
            It = St_bar * Ut
            median_It = torch.median(It)
            q75, q25 = np.percentile(It.cpu().detach().numpy(), [75 ,25])
            IQR = torch.tensor(q75 - q25)
            mu = torch.mean((It - median_It) / IQR)
            It_hat = (1 - alpha) * ((It - median_It) / IQR) + alpha * mu

            # Adjust gradients based on importance
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad)
                    clip_coef = min(C / (grad_norm + 1e-6), 1.0)
                    p.grad.data = p.grad.data * clip_coef + (noise_scale / torch.sqrt(It_hat + 1e-6)) * torch.randn_like(p.grad.data)

        optimizer.step()
        optimizer.zero_grad()
        St_prev = St_bar.detach()
        Ut_prev = Ut.detach()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Validation phase
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    accuracy = correct / total
    print(f'Validation Accuracy after Epoch {epoch+1}: {accuracy:.4f}')



