import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========= CONFIG =========
BASE_MODEL_PATH = "models/foundational_20260224_164448_044071"
LORA_PATH = "models/lora_r2/instruction_lora_20260302_195243_830407"
DEVICE = "cuda"

NUM_SAMPLES = 400
PROMPT = "5 + 7 - (10 - 3)"
MAX_NEW_TOKENS = 32
BATCH_SIZE = 8  # 可调

# ========= LOAD MODEL =========
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
).to(DEVICE)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# ========= PREP INPUT =========
inputs = tokenizer([PROMPT] * BATCH_SIZE, return_tensors="pt", padding=True).to(DEVICE)

# ========= WARMUP =========
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

# ========= BENCHMARK =========
start_time = time.time()

total_tokens = 0
num_batches = NUM_SAMPLES // BATCH_SIZE

with torch.no_grad():
    for _ in range(num_batches):
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )
        total_tokens += outputs.shape[1] * outputs.shape[0]

end_time = time.time()

# ========= METRICS =========
total_time = end_time - start_time
samples_per_sec = NUM_SAMPLES / total_time
tokens_per_sec = total_tokens / total_time

print("===== LoRA Speed Benchmark =====")
print(f"Total time (s): {total_time:.4f}")
print(f"Samples/sec: {samples_per_sec:.4f}")
print(f"Tokens/sec: {tokens_per_sec:.4f}")