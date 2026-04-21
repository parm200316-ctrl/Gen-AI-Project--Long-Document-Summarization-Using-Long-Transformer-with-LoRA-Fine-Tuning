import os, json, torch
from transformers import Trainer, TrainingArguments
from model import load_model
from data import load_data, preprocess, add_global_attention

SAVE_DIR = "./outputs"

model, tokenizer = load_model()

train, val, test = load_data()

train = train.map(preprocess, batched=True, remove_columns=train.column_names)
val = val.map(preprocess, batched=True, remove_columns=val.column_names)

train = train.map(add_global_attention, batched=True)
val = val.map(add_global_attention, batched=True)

train.set_format("torch")
val.set_format("torch")

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=val,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained(SAVE_DIR + "/final_model")
tokenizer.save_pretrained(SAVE_DIR + "/final_model")
