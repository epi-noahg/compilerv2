from parser import parse
import sys

def main():
    """
    Main entry point for the compiler.

    This function:
    1. Checks command-line arguments for the input file
    2. Reads the source code from the file
    3. Passes the source code to the parser
    4. Prints the result (AST or error message)
    5. Handles any exceptions that occur during processing
    """
    # Check if an input file was provided
    if len(sys.argv) < 2:
        print("usage: python main.py <token_file>")
        sys.exit(1)

    input_file: str = sys.argv[1]

    try:
        # Read the source code from the input file
        with open(input_file, 'r') as file:
            buffer = file.read()

        # Parse the source code and get the result
        result = parse(buffer)

        # Print the result (AST or error message)
        print(result)

    except FileNotFoundError:
        # Handle file not found error
        print(f"error: File '{input_file}' not found.")
        sys.exit(1)

    except Exception as err:
        # Handle any other exceptions
        print(f"error: {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
