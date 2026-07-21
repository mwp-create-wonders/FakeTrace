import sys

from src.faketrace_app.cli import main


if __name__ == "__main__":
    main(["audio", *sys.argv[1:]])
