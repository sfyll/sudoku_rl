"""
Usage:
    poetry run python scripts/print_profile.py /tmp/sudoku.prof

Prints top cumulative and tottime stats for a cProfile file.
"""

import sys
import pstats


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sudoku.prof"
    p = pstats.Stats(path)
    p.strip_dirs()
    print("Top 30 cumulative:")
    p.sort_stats("cumulative").print_stats(30)
    print("\nTop 30 tottime:")
    p.sort_stats("tottime").print_stats(30)


if __name__ == "__main__":
    main()
