import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

# Using tensorflow.keras everywhere here instead of standalone Keras
from tensorflow import keras

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = TFGPT2LMHeadModel.from_pretrained("gpt2-medium")

# Load dataset
dataset = load_dataset("stas/openwebtext-10k")

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Using DataCollatorForLanguageModeling from transformers to handle padding dynamically
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors="tf", mlm=False)
train_dataset = tokenized_datasets["train"].to_tf_dataset(columns=["input_ids"], shuffle=True, batch_size=8, collate_fn=data_collator)

# Differential privacy settings
l2_norm_clip = 1.0
noise_multiplier = 0.1
num_microbatches = 8

# Compile the model with the differential privacy optimizer
optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=0.001
)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# Training
model.fit(train_dataset, epochs=3)

# Evaluation
eval_loss = model.evaluate(train_dataset)
print(f"Evaluation loss: {eval_loss}")

