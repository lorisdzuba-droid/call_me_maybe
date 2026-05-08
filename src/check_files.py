from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic import model_validator, ValidationInfo
from typing import Any
from typing_extensions import Self


class PromptItem(BaseModel):
    """Check the Json architecture of the prompt calling.

    Args:
        BaseModel (Class): Pydantic Class
    """
    model_config = ConfigDict(extra='forbid')
    prompt: str

    @model_validator(mode='after')
    def check_empty(self) -> Self:
        """Check if the prompt is empty."""
        if not self.prompt or self.prompt.strip() == "":
            raise ValueError("The prompt can't be an empty string")
        return self


validate_prompts = TypeAdapter(list[PromptItem])


class TypeOnly(BaseModel):
    """Check the Types in function definition Json.

    Args:
        BaseModel (Class): Pydantic Class
    """
    model_config = ConfigDict(extra='forbid')
    type: str


class Function(BaseModel):
    """Chack the architecture of the fonction definition Json.

    Args:
        BaseModel (Class): Pydantic Class
    """
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: dict[str, TypeOnly]
    returns: TypeOnly


validate_functions = TypeAdapter(list[Function])


class OutputItem(BaseModel):
    """Check The output Json.

    validate the name of the function, it's parameters name
    and type. In case of error print a message.

    Args:
        BaseModel (Class): Pydantic Class

    Raises:
        ValueError: invalide Json

    Returns:
        Self: self
    """
    model_config = {"extra": "forbid"}

    prompt: str
    name: str
    parameters: dict[str, Any]

    @model_validator(mode='after')
    def check_against_definition(self, info: ValidationInfo) -> Self:
        """Validate the Json.

        Args:
            info (ValidationInfo): definition of functions

        Raises:
            ValueError: invalide Json

        Returns:
            Self: self
        """
        definitions = (info.context.get("function_definitions")
                       if info.context else None)
        if definitions is None:
            raise ValueError("Missing function"
                             " definitions in validation context")

        func_def = definitions.get(self.name)
        if func_def is None:
            print(f"Function '{self.name}' is not in the loaded definitions")

        expected_params = func_def["parameters"]
        provided_params = self.parameters

        extra = set(provided_params) - set(expected_params)
        if extra:
            print(f"Unexpected parameters: {extra}")

        missing = set(expected_params) - set(provided_params)
        if missing:
            print(f"Missing required parameters: {missing}")

        for pname, value in provided_params.items():
            expected_type = expected_params[pname]["type"]
            if not _value_matches_type(value, expected_type):
                print(
                    f"Parameter '{pname}' expected type '{expected_type}', "
                    f"but got {type(value).__name__}: {value!r}"
                )
        return self


def _value_matches_type(value: Any, expected_type: str) -> bool:
    """Check the types of values generated.

    Args:
        value (Any): value
        expected_type (str): type

    Returns:
        bool: valide value or not
    """
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    return True
