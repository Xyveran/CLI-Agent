from functions.get_files_info import get_files_info as gfi


def run_tests():

    print(f"Result for current directory:\n - {gfi('calculator', '.')}\n")

    print(f"Result for 'pkg' directory:\n - {gfi('calculator', 'pkg')}\n")

    print(f"Result for '/bin' directory:\n - {gfi('calculator', '/bin')}\n")

    print(f"Result for '../' directory:\n - {gfi('calculator', '../')}\n")

    
if __name__ == "__main__":
    run_tests()