import os

if __name__ == "__main__":
    cdir_path = os.getcwd()
    arr = os.listdir(cdir_path + '/tests/test_instances')
    print(f"Found {len(arr)} test instances.")

    for file in arr:

        print(file)

        os.system("python3 src/table.py tests/test_instances/" + file)  

        print()