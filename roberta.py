from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# Load the dataset and the tokenizer
dataset = load_dataset("imdb")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    # Tokenize the texts and convert outputs directly to tensors
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Apply the tokenization to the dataset
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoaders with uniform sampling for privacy guarantees
train_dataset = dataset['train']
valid_dataset = dataset['test']  # Using the test set as validation for simplicity
sample_rate = 0.01  # Sample rate for privacy calculations

train_loader = DataLoader(
    train_dataset,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(train_dataset),
        sample_rate=sample_rate,
        generator=torch.Generator().manual_seed(42)
    )
)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Load RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.train()
# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-6)

# Attach the privacy engine to the optimizer
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0
)

# Training loop with safe loss initialization

num_epochs = 3
for epoch in range(num_epochs):
    loss = torch.tensor(0.0).to(model.device)  # Initialize loss for safety
    for batch in train_loader:
        batch = {k: v.to(model.device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(model.device)
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Report progress with privacy metrics
    epsilon, best_alpha = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Epsilon: {epsilon}, Alpha: {best_alpha}")

# Evaluate the model
model.eval()
total_eval_loss = 0
for batch in valid_loader:
    batch = {k: v.to(model.device) for k, v in batch.items() if k != 'label'}
    labels = batch['label'].to(model.device)
    with torch.no_grad():
        outputs = model(**batch, labels=labels)
    total_eval_loss += outputs.loss.item()

# Final validation loss
print(f"Validation Loss: {total_eval_loss / len(valid_loader)}")

# Save the model
# model.save_pretrained('./roberta_imdb_dp_model')



