from transformers import LEDForConditionalGeneration, LEDTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def load_model():

    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer
