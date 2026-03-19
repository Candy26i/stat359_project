# Arithmetic LLM Pipeline

This project trains and evaluates a small arithmetic language model in five stages:

1. Generate data
2. Train tokenizer
3. Train foundational model
4. Fine-tune instruction model
5. Optionally fine-tune with LoRA

## Notes
CHECK OUT venv_setup.md for environment setup
- Replace `YYYYMMDD_HHMMSS` with an actual run directory under `models/`.
- Example: `models/foundational_20260201_012912_173614/best_model.pt`
- All commands below use the `code` package path because the project files are stored in the `code/` folder.

# Foundational corpus
poetry run python -m code.generate_foundational_plaintext \
  --num-samples 100000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.05 \
  --output-txt data/foundational_corpus.txt

# Instruction corpus
poetry run python -m code.generate_instruction_corpus_mixed \
  --num-samples 20000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0 \
  --output-mixed data/instruction_corpus.txt

# Test corpus
poetry run python -m code.generate_corpus \
  --instruction-only \
  --num-samples 1000 \
  --max-depth 4 \
  --output-instruction data/instruction_corpus_test.txt \
  --num-range 1 20 \
  --invalid-rate 0
  
# Train tokenizer
poetry run python -m code.train_tokenizer \
  --corpus-path data/foundational_corpus.txt \
  --output-dir data/tokenizer \
  --vocab-size 1000
  
# Train Fundational Model
poetry run python -m code.run_foundational_training \
  --corpus-path data/foundational_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --num-epochs 10 \
  --max-seq-length 512 \
# Train Lora Model

Change the lora rank paramters around, try 2, 16, 32, 64. At the same time, lora alpha should be twice as large as the lora rank.


poetry run python -m code.run_instruction_training_lora \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 10 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules attention \
  --save-merged-model
  --batch-size 16

# Evaluate Lora
poetry run python -m code.run_evaluation \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

# Inference speed examination

1. use Measure_Command{poetry run python -m code.run_evaluation \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000}
check for time usage.

2. use code/inference_time_400.py
- loads the foundational base model and LoRA adapter,
- runs batched generation on a fixed prompt,
- performs one warmup pass,
- benchmarks generation over many repeated samples,
- reports total runtime, samples per second, and tokens per second.

This is useful for comparing efficiency across different LoRA ranks and understanding the speed cost of adaptation during inference.
  
# Use Code/test_tocken_length.py to check token length

## Tokenizer Inspection

This script loads the trained `ArithmeticBPETokenizer` and analyzes how different types of arithmetic inputs are tokenized.

It encodes:
- Complex nested expressions
- Simple arithmetic expressions
- Long reasoning-style sequences (with `<think>` traces)
- Plain numbers (e.g., `1986`, `2222`)

For each input, it:
- Computes tokenized length
- Prints the first and last tokens (to inspect structure)
- Optionally decodes tokens back to text for verification

This helps diagnose:
- Sequence length growth with expression complexity
- Tokenization behavior for numbers vs expressions
- Whether the tokenizer preserves structure correctly

Visualize how current tokenizer treat the vocabulary in console.

# Use Code/prompt+length_accuracy+plot.py to generate plot

## Prompt-Length Analysis

This script visualizes how evaluation accuracy changes with prompt token length. It loads saved sample outputs, bins examples by `prompt_len`, computes per-bin accuracy, plots sample counts, and saves the figure to `evaluation_results/`.


# Use Code/Error_type_analysis+plot.py to check for error type and generate plot

## Error type Anlysis

- loads saved evaluation samples from JSON or JSONL,
- bins examples by `prompt_len`,
- classifies each sample as **unparsable**, **wrong reasoning**, or **correct**,
- computes the percentage of each category within every bin,
- prints a per-bin summary table,
- saves a stacked bar chart to `evaluation_results/`.
