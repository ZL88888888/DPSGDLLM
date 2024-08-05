from datasets import load_dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

# Load dataset
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_format(examples):
    # Tokenize text and ensure output is tensor
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    # Remove the unnecessary batch dimension
    tokenized_inputs = {k: v.squeeze(0) for k, v in tokenized_inputs.items()}
    # Ensure labels are tensors
    tokenized_inputs['labels'] = torch.tensor(examples['label'], dtype=torch.long)
    return tokenized_inputs

# Prepare and set format for dataset
train_dataset = dataset['train'].map(tokenize_and_format, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader with uniform sampling
sample_rate = 0.01
batch_size = 16
data_loader = DataLoader(
    train_dataset,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(train_dataset),
        sample_rate=sample_rate,
        generator=torch.Generator().manual_seed(42)
    ),
    collate_fn=lambda batch: {k: torch.stack([x[k] for x in batch]) for k in batch[0]}
)

# Model initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to('cuda')
model.train()

# Optimizer and differential privacy engine
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0
)

# Training loop
for epoch in range(3):
    for batch in data_loader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(model.device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    epsilon, best_alpha = privacy_engine.get_epsilon(delta=1e-5)
    print(f'Epoch: {epoch+1}, Epsilon: {epsilon:.2f}, Alpha: {best_alpha}')

# Save the trained model
model.save_pretrained('./bert_dp_model')



# Save the model
# model.save_pretrained('./bert_dp_model')



