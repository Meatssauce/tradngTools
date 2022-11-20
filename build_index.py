from block_parser import get_sorted_index
import pandas as pd
import os


def main():
    blocks_dir = r'D:\Bitcoin\blocks'
    index_path = os.path.join(blocks_dir, 'index.csv')
    index = get_sorted_index(blocks_dir)
    pd.DataFrame(index, columns=['time', 'path', 'position']).to_csv(index_path, index=False)


if __name__ == '__main__':
    main()
