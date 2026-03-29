from functions.write_file import write_file as wf


def run_tests():

    print(f"Result for 'lorem.txt':\n - {wf('calculator', 'lorem.txt', "wait, this isn't lorem ipsum")}\n")

    print(f'Result for "pkg/morelorem.txt":\n - {wf("calculator", "pkg/morelorem.txt", "lorem ipsum dolor sit amet")}\n')

    print(f"Result for '/tmp/temp.txt':\n - {wf('calculator', '/tmp/temp.txt', 'this should not be allowed')}\n")

    
if __name__ == "__main__":
    run_tests()