import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from datasets import load_dataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

# Add a padding token to the tokenizer if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_dataset = dataset['train']
test_dataset = dataset['test']

# Preprocess the dataset
def encode(examples):
    tokens = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    input_ids = torch.tensor(tokens['input_ids'])
    attention_mask = torch.tensor(tokens['attention_mask'])
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_dataset = train_dataset.map(encode, batched=True, remove_columns=["text"])
test_dataset = test_dataset.map(encode, batched=True, remove_columns=["text"])

# Custom collate function to ensure batch consistency
def custom_collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# DataLoader with custom collate function
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=custom_collate_fn)

# Validate the model for Opacus
model = ModuleValidator.fix(model)

# Set model to training mode
model.train()

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Initialize PrivacyEngine
privacy_engine = PrivacyEngine()

# Attach PrivacyEngine to the model and optimizer
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    for i, batch in enumerate(train_loader):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        print(f"Epoch {epoch}, Batch {i}")
        print(f"inputs shape: {inputs.shape}")
        print(f"labels shape: {labels.shape}")
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        print(f"loss: {loss.item()}")
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs, labels=labels)
        test_loss += outputs.loss.item()
test_loss /= len(test_loader)
print(f"Test loss: {test_loss}")

# Calculate epsilon
epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
print(f"DP-SGD with ε = {epsilon} and δ = 1e-5")








import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import opacus
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score

# Load dataset and tokenizer
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Function to encode the texts
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Map function to encode the dataset
train_dataset = dataset['train'].map(encode, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids'])

# Validation set
val_dataset = dataset['validation'].map(encode, batched=True)
val_dataset.set_format(type='torch', columns=['input_ids'])

# Define data loader with reduced batch size
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings

# Ensure the model is in training mode
model.train()

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Attach Opacus' PrivacyEngine to the model
privacy_engine = opacus.PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids']
            labels = batch['input_ids']
            with autocast():
                outputs = model(inputs, labels=labels)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
    model.train()
    return accuracy_score(all_labels, all_preds)

# Training loop using BatchMemoryManager
num_epochs = 3

for epoch in range(num_epochs):
    with BatchMemoryManager(data_loader=train_loader, optimizer=optimizer, max_physical_batch_size=2) as memory_safe_loader:
        for i, batch in enumerate(memory_safe_loader):
            inputs = batch['input_ids']
            labels = batch['input_ids']

            optimizer.zero_grad()

            with autocast():
                # Run forward pass
                outputs = model(inputs, labels=labels)
                loss = outputs.loss

            # Run backward pass with scaled loss
            scaler.scale(loss).backward()

            # Step optimizer
            scaler.step(optimizer)
            scaler.update()

            # Clear CUDA cache
            torch.cuda.empty_cache()

            print(f"Epoch {epoch}, Batch {i}")
            print(f"inputs shape: {inputs.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"loss: {loss.item()}")

            if i % 10 == 0:  # Evaluate every 10 batches
                accuracy = evaluate(model, val_loader)
                print(f"Validation Accuracy after {i} batches: {accuracy}")

            if i >= 50:  # limit to 50 batches for demonstration
                break






























