from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "models/YandexGPT-5-Lite-8B-instruct"

# Настройка устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация токенайзера и модели
print(f"Loading tokenizer and model '{MODEL_ID}' on device {device} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Оптимизированная инициализация модели
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=None,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

model.eval()

def build_prompt(code: str, system_prompt: Optional[str] = None) -> str:
    """
    Создаёт промпт для данного куска C-кода.
    Вы можете менять шаблон под свои стандарты.
    """
    base_system = (
        "Ты — инженер по требованиям и документации. Твоя задача: прочитать переданный C-код и "
        "сформулировать чёткое, тестируемое требование (или набор требований) к функциональности "
    )
    if system_prompt:
        base_system = system_prompt

    prompt = (
        f"{base_system}\n\n"
        "Инструкция: на основе следующего кода на языке C сформулируй:\n"
        "Точные требования (пункты), подробно описывающие действия функии.\n"
        "Код:\n```\n" + code.strip() + "\n```\n\n"
        "Ответь развернуто, иногда можешь использовать псевдокод."
    )
    return prompt

# Функция генерации требования к коду на С
def generate_requirement(code: str, system_prompt: Optional[str]=None, max_new_tokens: int = 2024, temperature: float = 0.0) -> str:
    """
    Сгенерировать требование/спецификацию для переданного C-кода.
    """
    prompt = build_prompt(code, system_prompt=system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=False if temperature == 0.0 else True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )

    out = tokenizer.decode(generated[0], skip_special_tokens=True)
    # Вычистим: оставить только текст после исходного промпта, если модель вернула и вход.
    if out.startswith(prompt):
        result = out[len(prompt):].strip()
    else:
        # иногда модель возвращает полный диалог иначе — просто возвращаем всё, но аккуратно
        result = out.strip()
    return result

# Пример использования
if __name__ == "__main__":

    sample_c_code = """..."""
    spec = generate_requirement(sample_c_code, max_new_tokens=2024, temperature=0.0)
    print("=== GENERATED SPEC ===\n")
    print(spec)