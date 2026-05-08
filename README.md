*This project has been created as part of the 42 curriculum by ldzuba.*

# call me maybe — Introduction to Function Calling in LLMs

## Description

**call me maybe** is a function-calling tool that translates natural language prompts into structured, schema-compliant JSON function calls. Given a prompt like `"What is the sum of 40 and 2?"`, the program does not answer the question — it identifies the right function to call and extracts the correct arguments:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The core challenge this project addresses is reliability: small language models (like `Qwen/Qwen3-0.6B`) are inherently unreliable at producing structured output when left unconstrained. This project solves that with **constrained decoding** — a technique that controls token generation step-by-step to guarantee 100% valid, schema-compliant JSON output, regardless of the model's natural tendencies.

---

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- Numpy
- Json

### Installation

```bash
# to direct the path to the Sgoinfre if you are on a 42 computer
source nimp.sh
make install
# or
make
```

The `llm_sdk/` directory must be present at the project root (alongside `src/`). It is not installed as a package — it is copied directly and imported locally.

### Running the program

```bash
make run
```

With custom paths:

```bash
make run ARGS="--functions_definition data/input/functions_definition.json  --input data/input/function_calling_tests.json --output data/output/function_calls.json"
```

**Default paths** (used when no flags are provided):
- Functions definition: `data/input/functions_definition.json`
- Input prompts: `data/input/function_calling_tests.json`
- Output: `data/output/function_calls.json`

### Other Makefile targets

```bash
make lint         # Run flake8 and mypy with standard flags
make debug        # Run with Python's pdb debugger
make clean        # Remove __pycache__, .mypy_cache, etc.
```

---

## Algorithm Explanation

### Overview: Constrained Decoding

Standard LLM generation picks the most probable next token at each step with no regard for output structure. Constrained decoding intercepts this process and restricts which tokens are legal at each step, based on the current generation state.

This project implements a **finite-state machine (FSM)** approach to constrained decoding with the following stages:

| Stage | Description |
|---|---|
| `NEED_FONCTION` | Only function name tokens are allowed |
| `NEED_SPACE` | Only a space token is allowed (separator) |
| `NEED_PARAM_NAME` | Only parameter name tokens for the chosen function are allowed |
| `NEED_COLON` | Only `:` is allowed (key-value separator) |
| `NEED_QUOT` | Only `"` is allowed (opening quote for string values) |
| `NEED_PARAM_VALUE` | Free generation with logit boosting (biased toward prompt content) |
| `NEED_BRACKET` | Only `{` is allowed (opening bracket for object values) |
| `DONE` | Generation is complete; loop exits |

### Step-by-step generation

1. A structured prompt is built (`build_dynamic_prompt`) describing available functions and the user request.
2. The prompt is tokenized via `encode()`.
3. At each step, `get_stage()` inspects the tokens generated so far and updates the FSM state.
4. `_get_allowed_tokens()` returns the set of valid next token IDs based on the current stage.
5. `_apply_mask()` sets all other logits to `-inf`, making invalid tokens impossible.
6. For value generation (`NEED_PARAM_VALUE`), logit **boosting** is applied: numeric tokens that appear in the original prompt get a `+0.5` boost; string tokens get a `+3.0` boost. This guides the model toward extracting values directly from the prompt rather than hallucinating.
7. `argmax` selects the next token from the masked logit distribution.
8. The loop ends when `_all_params_complete()` confirms all parameters have valid, complete values.

### Output cleaning

After generation, `clean_output()` strips the prompt prefix from the generated text, then parses parameter key-value pairs using regex-based extraction. Type coercion is applied (string → int or float where possible). The result is validated against the function schema using Pydantic before being written to disk.

---

## Design Decisions

**FSM-based state tracking over grammar-based approaches:** Rather than implementing a full JSON grammar parser (e.g., with a PDA), the project uses a lighter finite-state machine tuned specifically to the expected output format (`fn_name param1:val1, param2:val2`). This is simpler to reason about, easier to debug, and sufficient for the complexity level required.

**Pydantic for all validation:** Input files (`function_calling_tests.json`, `functions_definition.json`) and generated outputs are all validated through Pydantic models (`PromptItem`, `Function`, `OutputItem`). This guarantees schema compliance at every boundary and produces clear error messages on malformed input.

**Logit boosting for value extraction:** Instead of relying solely on the model to produce correct values, the generation is biased toward tokens that appear in the original prompt. This is a lightweight form of grounding that significantly improves accuracy for argument extraction without requiring a larger model.

**Custom prompt format for the LLM:** The LLM is instructed to produce output in a compact, space-separated format (`fn_name param1:val1, param2:val2`) rather than raw JSON. This makes it easier to constrain token-by-token, since JSON's nested structure and quoting rules are more complex to enforce incrementally.

**Vocabulary pre-loading:** The full token vocabulary is loaded once at startup (`_load_vocabulary()`) and cached in both directions (`_id_to_token`, `_token_to_id`), avoiding repeated file I/O during generation.

---

## Performance Analysis

**Accuracy:** The constrained decoding approach guarantees that the output is always structurally valid (correct function name, correct parameter names). Value extraction accuracy depends on the logit boosting and the model's ability to ground values to the prompt; for well-formed prompts, this achieves high reliability.

**Speed:** Is highly dependent on the hardwear. On a good cumputer with `Qwen3-0.6B`, this is approximately 2–5 seconds per prompt depending on the number of parameters. A full test set of ~10 prompts runs well within the 5-minute target. But on a 42 computer it's approximately 30-60 seconds per prompt making it close to the 5-minutes target.

**Reliability:** Because the tokens generated are cast in a Json, invalid JSON structure is structurally impossible. All outputs pass Pydantic validation before being written to disk; any failure raises a clear `ValueError` rather than producing a corrupted file.

---

## Challenges Faced

**State detection from generated tokens:** Determining the current FSM stage purely by inspecting the suffix of generated token IDs is fragile. The solution was to decode generated tokens incrementally and match against known function names and parameter names via string search and regex.

**Partial vs. complete values:** Numbers can be generated across multiple tokens (e.g., `"3"`, `"4"`, `"5"` for `345`). Detecting when a number is complete required checking that the *next* token is not a digit or decimal point, not just that the current substring is numeric.

**String value boundaries:** String values may contain spaces or commas, which are also used as separators. The `_extract_param_value()` method handles this by tracking quote depth to correctly identify the end of a string value but make it impossible to generate a value that include a quote.


---

## Testing Strategy

Testing was performed by:

1. **Running the full pipeline** on the provided example input files and automatically inspecting the output JSON for correctness.
2. **Edge case prompts:** empty-ish strings, large numbers, multi-parameter functions, string values containing spaces.
3. **Schema validation:** All outputs are validated by Pydantic's `OutputItem` model against the loaded function definitions before being written — any mismatch (wrong type, missing parameter, unknown function) is caught and reported.
4. **Malformed input handling:** Tested with missing files and invalid JSON inputs to confirm graceful error reporting via `try/except` blocks throughout the pipeline.


---

## Example Usage

**Minimal (using defaults):**

```bash
uv run python -m src
```

**Full explicit paths:**

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

**Example input** (`data/input/function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

**Example output** (`data/input/functions_definition.json`):

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "a": {
        "type": "string"
      },
      "b": {
        "type": "string"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
]
```

---

## Resources

### References

- [Qwen3 Model Card — Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)


### AI Usage

Deepseek was used during this project for the following tasks:

- **README writing:** Drafting and structuring this README based on the subject requirements and the actual codebase.
- **Debugging assistance:** Helping identify edge cases in the FSM state transitions (e.g., partial number detection, string boundary handling).


All AI-generated content was reviewed, tested, and validated before inclusion.
