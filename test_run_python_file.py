from functions.run_python_file import run_python_file as rpf


def run_tests():

    print(f'Result for "calc main.py":\n - {rpf("calculator", "main.py")}\n')

    print(f'Result for "cal main.py ["3 + 5"]":\n - {rpf("calculator", "main.py", ["3 + 5"])}\n')

    print(f'Result for "calc tests.py":\n - {rpf("calculator", "tests.py")}\n')

    print(f'Result for "calc ../main.py":\n - {rpf("calculator", "../main.py")}\n')

    print(f'Result for "calc nonexistent.py":\n - {rpf("calculator", "nonexistent.py")}\n')

    print(f'Result for "calc lorem.txt":\n - {rpf("calculator", "lorem.txt")}\n')

    
if __name__ == "__main__":
    run_tests()