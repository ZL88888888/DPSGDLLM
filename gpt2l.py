import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from datasets import load_dataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from transformers import default_data_collator

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

# Add a padding token to the tokenizer (if it doesn't exist)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_dataset = dataset['train']
test_dataset = dataset['test']

# Preprocess the dataset
def encode(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

train_dataset = train_dataset.map(encode, batched=True, remove_columns=["text"])
test_dataset = test_dataset.map(encode, batched=True, remove_columns=["text"])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=default_data_collator)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=default_data_collator)

# Validate the model for Opacus
model.train()
model = ModuleValidator.fix(model)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Initialize PrivacyEngine
privacy_engine = PrivacyEngine(
    accountant="rdp",
    secure_mode=False
)

privacy_engine = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(3):
    for batch in train_loader:
        inputs = batch['input_ids'].to(device)
        labels = inputs.clone()
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed")

# Evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input_ids'].to(device)
        labels = inputs.clone()
        outputs = model(inputs, labels=labels)
        test_loss += outputs.loss.item()
test_loss /= len(test_loader)
print(f"Test loss: {test_loss}")





