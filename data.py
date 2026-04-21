from datasets import load_dataset
from transformers import LEDTokenizer

tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")

def load_data(train_size=16000, val_size=2000, test_size=2000):
    dataset = load_dataset("ccdv/arxiv-summarization")["train"]
    dataset = dataset.shuffle(seed=42).select(range(train_size + val_size + test_size))

    train = dataset.select(range(train_size))
    val = dataset.select(range(train_size, train_size + val_size))
    test = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

    return train, val, test


def preprocess(example):
    inputs = tokenizer(
        example["article"],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["abstract"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs


def add_global_attention(batch):
    batch["global_attention_mask"] = [
        [1] + [0]*(len(ids)-1) for ids in batch["input_ids"]
    ]
    return batch
