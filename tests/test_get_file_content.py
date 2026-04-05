from functions.get_file_content import get_file_content as gfc


def run_tests():

    print(f"Result for .main.py:\n - {gfc('calculator', 'main.py')}\n")

    print(f"Result for 'pkg/calculator.py':\n - {gfc('calculator', 'pkg/calculator.py')}\n")

    print(f"Result for '/bin/cat':\n - {gfc('calculator', '/bin/cat')}\n")

    print(f"Result for 'pkg/does_not_exist.py':\n - {gfc('calculator', 'pkg/does_not_exist.py')}\n")

    
if __name__ == "__main__":
    run_tests()