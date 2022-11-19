import cProfile
import pstats

from block_parser import build_ledger_history


def main():
    with cProfile.Profile() as pr:
        build_ledger_history(r'D:\Bitcoin\blocks', end=20000)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='needs_profiling.prof')


if __name__ == '__main__':
    main()
