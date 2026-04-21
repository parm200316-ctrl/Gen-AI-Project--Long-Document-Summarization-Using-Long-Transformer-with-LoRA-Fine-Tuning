from model import load_model
from data import load_data

model, tokenizer = load_model()
model.eval()

_, _, test = load_data()

samples = test.select(range(5))

for i, sample in enumerate(samples):

    inputs = tokenizer(sample["article"], return_tensors="pt", truncation=True, max_length=1024).to("cuda")

    global_attention = torch.zeros_like(inputs["input_ids"])
    global_attention[:, 0] = 1

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            global_attention_mask=global_attention,
            max_length=128
        )

    print("\n--- SAMPLE", i+1, "---")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
