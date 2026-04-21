import torch, json
import evaluate
from model import load_model
from data import load_data, preprocess, add_global_attention

model, tokenizer = load_model()
model.eval()

rouge = evaluate.load("rouge")

_, _, test = load_data()

test = test.map(preprocess, batched=True, remove_columns=test.column_names)
test = test.map(add_global_attention, batched=True)
test.set_format("torch")

preds, refs = [], []

for i, sample in enumerate(test.select(range(100))):

    input_ids = sample["input_ids"].unsqueeze(0).to("cuda")
    global_attention_mask = sample["global_attention_mask"].unsqueeze(0).to("cuda")

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            max_length=128,
            num_beams=4
        )

    preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    refs.append(tokenizer.decode(sample["labels"], skip_special_tokens=True))

scores = rouge.compute(predictions=preds, references=refs)
print(scores)

with open("outputs/rouge_scores.json", "w") as f:
    json.dump(scores, f, indent=2)
