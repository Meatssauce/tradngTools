import cProfile
import pstats
import os

from blockchain_datamining.block_parser import build_index, build_ledger_history


def main():
    with cProfile.Profile() as pr:
        blocks_dir = 'datasets/blocks'
        index_filepath = 'datasets/blocks/index.csv'

        if not os.path.isfile(index_filepath):
            build_index(blocks_dir, index_filepath)
        utxo, balances = build_ledger_history(index_filepath, read_from_index=True)
        print(f'{utxo=}')
        print(f'{balances=}')

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='needs_profiling.prof')


if __name__ == '__main__':
    main()
