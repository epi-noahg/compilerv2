from core.parse import parse
import traceback
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: python main.py <token_file>")
        sys.exit(1)
    input_file: str = sys.argv[1]
    try:
        with open(input_file, 'r') as file:
            buffer = file.read()
        result = parse(buffer)
        print(result)
    except FileNotFoundError:
        print(f"error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as err:
        print(f"error: {err}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
