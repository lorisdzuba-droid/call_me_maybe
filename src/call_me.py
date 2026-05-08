from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError, TypeAdapter
import json
import re
import numpy as np
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from .check_files import validate_functions, validate_prompts, OutputItem
import os


class ConstrainedFunctionCaller(BaseModel):
    """A class to handle constrained decoding for function calling
    that will generate a valide Json.

    Args:
        BaseModel (BaseModel): _Pydantic Base class

    Raises:
        ValueError: Invalide Json
    """
    func_path: str
    input_path: str
    output_path: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize the model and load vocabulary.

        Args:
            __context (Any): privatise
        """
        self._model = Small_LLM_Model()
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}
        self._functions: list[dict[str, Any]] = []
        self._load_vocabulary()
        self._prompt: str = ""
        self._function_names: List[str] = []
        self._function_params: Dict[str, list[Any]] = {}
        self._stage: str = "NEED_FONCTION"
        self._current_function: str = ""
        self._param_added: List[str] = []
        self._current_param: str = ""
        self._completed_param: List[str] = []
        self._param_type: str = ""
        self._func_dict: Dict[str, Any] = {}
        self._function_map: Dict[str, Any] = {}
        self._small_prompt: str = ""
        self._params: Dict[str, str] = {}
        self._next_token: str = ""
        self._all_token: List[int] = []
        self._flag: int = 1

    def _extract_param_value(self, text: str,
                             param_name: str,
                             param_type: str) -> Optional[str]:
        """ Given the generated addition (after the prompt),
        extract the value substring
        for a specific parameter.

        Assumes text format: fn_name param1:val1, param2:val2, ...

        Args:
            text (str): tokens generated
            param_name (str): parameter being searched
            param_type (str): type of the param

        Returns:
            Optional[str]: the value generated if any
        """

        pattern = re.escape(param_name) + r':'
        match = re.search(pattern, text)
        if not match:
            return None
        start = match.end()

        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text) and param_type == "string":
            self._stage = "NEED_QUOT"
        elif start == len(text) and param_type == "object":
            self._stage = "NEED_BRACKET"
        i = start
        n = len(text)
        in_quotes = False
        brace_depth = 0

        while i < n:
            ch = text[i]
            if ch == '"' and (i == 0 or text[i-1] != '\\'):
                in_quotes = not in_quotes
            elif ch == '{' and not in_quotes:
                brace_depth += 1
            elif ch == '}' and not in_quotes:
                brace_depth -= 1
            elif (ch == ',' and not in_quotes
                  and brace_depth == 0 or ch == ' '
                  and not in_quotes and brace_depth == 0):
                break
            i += 1
        value = text[start:i].strip()
        return value if value else None

    def _is_value_complete(self, value: Optional[str],
                           param_type: str,
                           param_name: str) -> bool:
        """Check that a value is complete.

        Args:
            value (Optional[str]): value
            param_type (str): type of value
            param_name (str): name of the parameter

        Returns:
            bool: is it complete or not
        """
        if value is None or value == '':
            return False
        if param_type == "number" or param_type == "integer":
            if not bool(re.match(r'^(?:[+-]|\d+(?:\.\d*)?|\.\d+|\.)$',
                                 self._next_token)):
                if param_type == "number" and "." not in value:
                    self._stage = "NEED_FLOAT"
                    float_token = self._get_next_token(
                        self._all_token, [self.encode(".")])
                    self._all_token.append(float_token)
                    float_token = self._get_next_token(
                        self._all_token, [self.encode("0")])
                    self._all_token.append(float_token)
                    value = self._model.decode(self._all_token)
                    self._stage = "NEED_PARAM_VALUE"
                result = bool(re.match(r'^-?\d+(\.\d+)?$', value.strip('"')))
                if result:
                    if param_name not in self._completed_param:
                        self._completed_param.append(param_name)
                        if any(param in self._next_token and
                               param != self._current_param
                                for param in self._function_map[
                                    self._current_function][
                                        "parameters"].keys()):
                            self._stage = "NEED_COLON"
                        else:
                            self._stage = "NEED_SPACE"
                    return result
        elif param_type == "string":
            result = (len(value) >= 2 and value.startswith('"')
                      and value.endswith('"'))
            if result:
                if param_name not in self._completed_param:
                    self._completed_param.append(param_name)
                    self._stage = "NEED_SPACE"
                return result
        elif param_type == "object":
            # Must start with '{', end with '}', and braces balanced
            if not (value.startswith('{') and value.endswith('}')):
                return False
            depth = 0
            for ch in value:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                if depth < 0:
                    return False
            result = depth == 0
            if result:
                if param_name not in self._completed_param:
                    self._completed_param.append(param_name)
                    self._stage = "NEED_SPACE"
                return result
        return False

    def _all_params_complete(self, added_text: str) -> bool:
        """Dteremine the end of the fonction calling.

        By checking that all parameters have a complete value

        Args:
            added_text (str): Tokens generated

        Returns:
            bool: end or not
        """
        func = self._function_map.get(self._current_function)
        if not func:
            return False
        self._current_function_params_names = list(func["parameters"].keys())
        for param_name, param_info in func["parameters"].items():
            self._param_type = param_info["type"]
            value = self._extract_param_value(added_text, param_name,
                                              param_info["type"])
            if not self._is_value_complete(value, param_info["type"],
                                           param_name):
                return False
        # Also ensure that the last parameter's value is actually "closed"
        # For numbers, this is automatically true if valid.
        return True

    def _load_vocabulary(self) -> None:
        """Load the vocabulary from the model's vocab file."""
        vocab_path = self._model.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self._token_to_id = json.load(file)
            self._id_to_token = {int(v): k for k, v in
                                 self._token_to_id.items()}

    def encode(self, text: str) -> Any:
        """Encode a string into token IDs.

        Args:
            text (str): string

        Returns:
            Any: Token IDS
        """
        tokens = self._model.encode(text)
        return tokens.tolist()[0] if hasattr(tokens, 'tolist') else tokens

    def get_stage(self, tokens_id: List[int]) -> Optional[None]:
        """Set the stages of generation.

        Used to apply differents mask for different
        stages.

        Args:
            tokens_id (List[int]): Tokens generated

        Returns:
            Optional[None]: None
        """
        current_text = self._model.decode(tokens_id)
        added_text = current_text[len(self._prompt):]
        print(added_text)
        if not added_text.strip():
            self._stage = "NEED_FONCTION"
            return
        if self._stage == "NEED_FONCTION":
            for name in self._function_names:
                if name in added_text:
                    self._current_function = name
                    self._stage = "NEED_SPACE"
        elif self._stage == "NEED_SPACE":
            self._stage = "NEED_PARAM_NAME"
        elif self._stage == "NEED_PARAM_NAME":
            func_dict = self._function_map[self._current_function]
            added_text = added_text[len(self._current_function):].strip()
            for param in func_dict["parameters"].keys():
                if param in added_text:
                    if param not in self._param_added:
                        self._param_added.append(param)
                        self._current_param = param
                    self._stage = "NEED_COLON"
        elif self._stage == "NEED_COLON":
            self._stage = "NEED_PARAM_VALUE"
        elif self._stage == "NEED_QUOT":
            self._stage = "NEED_PARAM_VALUE"

    def _apply_mask(self, logits: List[float],
                    allowed_ids: List[int]) -> np.ndarray | list[float]:
        """Apply a mask to logits, allowing only specified token IDs.

        Args:
            logits (List[float]): list of logits generated
            allowed_ids (List[int]): mask

        Returns:
            np.ndarray | list[float]: logits with authorised ids
        """
        mask = np.full_like(logits, -np.inf)
        mask[allowed_ids] = 0
        if self._stage == "NEED_PARAM_VALUE":
            return logits
        return logits + mask

    def load_functions(self, functions_path: str) -> None:
        """Open and validate Json.

        Args:
            functions_path (str): path to the Json

        Raises:
            ValueError: invalide Json
        """
        with open(functions_path, "r") as file:
            raw_text = file.read()
        try:
            validated = validate_functions.validate_json(raw_text)
            self._functions = [f.model_dump() for f in validated]
        except ValidationError as e:
            raise ValueError(f"Invalid function definitions file: {e}") from e
        self._func_dict = {f["name"]: f for f in self._functions}
        self._parse_functions()

    def _parse_functions(self) -> None:
        """Extract function names and parameter names from loaded functions."""
        for func in self._functions:
            name = func["name"]
            self._function_names.append(name)
            params = list(func["parameters"].keys())
            self._function_params[name] = params
        self._function_map = {f["name"]: f for f in self._functions}

    def get_function_names(self) -> List[str]:
        """Extract function names from loaded functions.

        Returns:
            List[str]: functions
        """
        return [func["name"] for func in self._functions]

    def get_functions_param(self) -> List[str]:
        """Extract function parameters from loaded function

        Returns:
            List[str]: parameters
        """
        result = []
        for func in self._functions:
            self._function_names.append(func["name"])
            result.extend(func["parameters"].keys())
        return result

    def _get_allowed_tokens(self) -> List[int]:
        """Calculate allowed token IDs for a given prompt.

        Returns:
            List[int]: allowed token IDs
        """
        allowed_ids = []
        if self._stage == "NEED_FONCTION":
            for func_name in self.get_function_names():
                allowed_ids.extend(self.encode(func_name))
        if self._stage == "NEED_SPACE":
            allowed_ids = self.encode(" ")
        if self._stage == "NEED_PARAM_NAME":
            for param_name in (self._function_map[self._current_function]
                               ["parameters"].keys()):
                allowed_ids.extend(self.encode(param_name))
        if self._stage == "NEED_COLON":
            allowed_ids.extend(self.encode(":"))
        if self._stage == "NEED_QUOT":
            allowed_ids.extend(self.encode('"'))
        return allowed_ids

    def _waying_token(self) -> None:
        """Way token IDs to get a control on value generations.
        """
        self._prompt_logit_num = []
        self._prompt_logit_str = []
        prompt = self._small_prompt.split()
        for word in prompt:
            if word.isalpha() and len(word) > 2:
                self._prompt_logit_str.extend(self.encode(word))
            if word.isnumeric():
                self._prompt_logit_num.extend(self.encode(word))

    def _get_next_token(self, token_ids: List[int],
                        allowed_ids: List[int]) -> int:
        """Get the next token ID based on current context and constraints.

        Args:
            token_ids (List[int]): generated token
            allowed_ids (List[int]): mask

        Returns:
            int: Token Id
        """
        logits = self._model.get_logits_from_input_ids(token_ids)
        masked_logits = self._apply_mask(logits, allowed_ids)
        if self._stage == "NEED_PARAM_VALUE":
            self._waying_token()
            if self._param_type == "number":
                for id in self._prompt_logit_num:
                    masked_logits[id] += 0.5
            elif self._current_param.lower() != "regex":
                printed_boost = []
                for string in self._prompt_logit_str:
                    printed_boost.append(self._model.decode(string))
                for id in self._prompt_logit_str:
                    masked_logits[id] += 3
        if self._stage != "NEED_FLOAT":
            self._next_token = self._model.decode([int(np.argmax(
                masked_logits))])
        return int(np.argmax(masked_logits))

    def build_dynamic_prompt(self, prompt: str) -> str:
        """Prompt given to the LLM.

        Args:
            prompt (str): prompt from the Json

        Returns:
            str: full prompt
        """
        intro = (
         "You are a precise function-calling system. "
         "Your task is to translate "
         "the user's request into a single function call.\n\n"
         "CRITICAL RULES:\n"
         "1. Output ONLY the function call in this EXACT format: "
         'fn_name param1:"string", param2:number\n'
         "2. Use the EXACT function names and parameter names shown below\n"
         "3. Extract parameter values directly from the user's request\n"
         "EXAMPLES:\n"
         'User: replace all numbers with "Hey" in "1 my friend" '
         '→ fn_substitute_string_with_regex source_string:"1 my friend" '
         'regex:"[(0-9)+]" replacement:"Hey"'
        )

        available = "Available functions:\n"
        for func in self._functions:
            name = func["name"]
            desc = func["description"]
            params = func["parameters"]
            self._params = params
            param_strs = []
            for pname, pinfo in params.items():
                param_strs.append(f"{pname}: {pinfo['type']}")
            param_desc = ", ".join(param_strs)
            available += f"- {name}: {desc} (parameters: {param_desc})\n"

        request_line = f"\nUser request: {prompt}\nFunction call:"
        return intro + available + request_line

    def generate_function_call(self, prompt: str, max_steps: int = 50) -> str:
        """Generate a function call for a single prompt.

        Args:
            prompt (str): prompt
            max_steps (int, optional): limit of steps. Defaults to 50.

        Returns:
            str: function call
        """
        print(f"\n=== Processing: {prompt} ===")
        self._small_prompt = prompt
        self._prompt = self.build_dynamic_prompt(prompt)
        token_ids = self.encode(self._prompt)
        self._all_token = token_ids
        self._stage = "NEED_FONCTION"
        for step in range(max_steps):
            token_ids = self._all_token
            allowed_ids = self._get_allowed_tokens()
            next_token = self._get_next_token(token_ids, allowed_ids)
            added_text = self._model.decode(token_ids)[len(self._prompt):]
            added_text = added_text[len(self._current_function):].strip()
            if self._all_params_complete(added_text):
                self._stage = "DONE"
            if self._stage == "DONE":
                break
            token_ids.append(next_token)
            self.get_stage(token_ids)
            added_text = self._model.decode(token_ids)[len(self._prompt):]
            added_text = added_text[len(self._current_function):].strip()
            if self._all_params_complete(added_text):
                self._stage = "DONE"
            self._all_token = token_ids
        return self._model.decode(token_ids)

    def clean_output(self, output: Any,
                     prompt: str) -> Optional[Dict[str, Any]]:
        """Clean up the outputed tokens

        Args:
            output (Any): Tokens
            prompt (str): prompt used

        Returns:
            Optional[Dict[str, Any]]: the generated output if valide
        """
        final_output = output[len(self._prompt):]
        parameters = final_output[len(self._current_function):]
        normalised = parameters
        param_dict = {}
        for param_name in self._current_function_params_names[::-1]:
            pattern = re.escape(param_name) + r':'
            match = re.search(pattern, normalised)
            if not match:
                return None
            start = match.start()
            key, val = normalised[start:].split(':', 1)
            key = key.strip()
            val = val.strip()
            val = val.strip(",")
            val = val.strip('"')
            if len(val) > 2:
                val = val.strip()
            try:
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass
            param_dict[key] = val
            normalised = normalised[:start]
        param_dict = dict(reversed(list(param_dict.items())))
        results = ({"prompt": prompt,
                    "name": self._current_function,
                    "parameters": param_dict})
        self._completed_param = []
        return results

    def process_prompts(self,
                        prompts: List[str]) -> List[Dict[str, Any] | None]:
        """Process multiple prompts and return results.

        Args:
            prompts (List[str]): prompt

        Returns:
            List[Dict[str, Any] | None]: results
        """
        results = []
        for prompt in tqdm(prompts):
            output = self.generate_function_call(prompt)
            clean_output = self.clean_output(output, prompt)
            results.append(clean_output)

        return results

    def run(self) -> None:
        """Run the fonction call.

        Raises:
            ValueError: Invalide output
        """
        self.load_functions(self.func_path)
        definitions = {f["name"]: f for f in self._functions}

        with open(self.input_path) as f:
            prompt_items = validate_prompts.validate_json(f.read())
        prompts = [item.prompt for item in prompt_items]

        raw_results = self.process_prompts(prompts)

        adapter = TypeAdapter(list[OutputItem])
        try:
            validated = adapter.validate_python(
                raw_results,
                context={"function_definitions": definitions}
            )
        except ValidationError as e:
            raise ValueError(f"Generated output is invalid: {e}") from e

        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump([item.model_dump() for item in validated],
                          f,
                          indent=2)
        del self._model
