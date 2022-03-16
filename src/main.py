import os

from cli import run_cli

# TODO this is temporary, find a longterm solution for finding dataset path
DATASET_DIR = os.path.join(os.getcwd(), "datasets")

def main():
    run_cli()

if __name__ == "__main__":
    main()

