import argparse
from src.call_me import ConstrainedFunctionCaller


def main() -> None:
    """Parse the arguments and sets defaults one if needed.
    Launch a function callers that will read the prompts and generate
    a valide json with the functions needed and the arguments.
    """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--functions_definition',
                            default='data/input/functions_definition.json')
        parser.add_argument('--input',
                            default='data/input/function_calling_tests.json')
        parser.add_argument('--output',
                            default='data/output/function_calls.json')
        args = parser.parse_args()

        runner = ConstrainedFunctionCaller(func_path=args.functions_definition,
                                           input_path=args.input,
                                           output_path=args.output)
        runner.run()
    except BaseException as e:
        print(e)


if __name__ == "__main__":
    main()
