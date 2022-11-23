import fileinput
import glob
import hashlib
import os
import re
import warnings
from collections import defaultdict
from itertools import pairwise
from typing import IO

import pandas as pd
from tqdm import tqdm

from blockchain import Block, Output, Input
from tools import ReadableFileInput, Opcode


def make_merkle_root(lst):  # https://gist.github.com/anonymous/7eb080a67398f648c1709e41890f8c44
    if len(lst) == 1:
        return lst[0]
    if len(lst) % 2 == 1:
        lst.append(lst[-1])

    sha256d = lambda x: hashlib.sha256(x).digest()
    hash_pair = lambda x, y: sha256d(sha256d(x[::-1] + y[::-1]))[::-1]
    # todo: check if its addition or arr of x y

    return make_merkle_root([hash_pair(x, y) for x, y in pairwise(lst)])


def read_dat_from_index(index_filepath: str):
    with open(index_filepath, 'r') as f:
        i = 0
        f.readline()
        while True:
            try:
                _, path, position = f.readline().split(',')
                position = int(position)
            except (EOFError, ValueError):
                break

            with open(path, 'rb') as block_file:
                block_file.seek(position)
                yield Block.from_file(block_file, height=i)
            i += 1


def read_dat(filepaths: [str], return_index=False):
    try:
        with ReadableFileInput(filepaths, 'rb', verbose=True) as f:
            i = 0
            while True:
                try:
                    if return_index:
                        yield f.openedFilepath(), f.positionInFile(), Block.from_file(f, height=i)
                    else:
                        yield Block.from_file(f, height=i)
                except EOFError:
                    break
                i += 1
    except StopIteration:
        return


def update_ledger(block: Block, utxo: defaultdict[any, set[tuple[str, int]]], balances: defaultdict[str, float]):
    for tx in block.transactions:
        for i, input_ in enumerate(tx.inputs):
            if input_.coinbase:
                continue

            txo_being_spent = input_.original_output

            for sender in txo_being_spent.recipients:
                if not sender:
                    continue

                # if sender in utxo:
                #     utxo[sender].remove((input_.tx_id, input_.vout))
                # else:
                #     utxo[sender] = set()

                if sender in balances:
                    balances[sender] -= txo_being_spent.value
                else:
                    balances[sender] = -txo_being_spent.value

        for vout, output in enumerate(tx.outputs):
            for recipient in output.recipients:
                if not recipient:
                    continue

                # if recipient in utxo:
                #     utxo[recipient].add((tx.id, vout))
                # else:
                #     utxo[recipient] = {(tx.id, vout)}

                if recipient in balances:
                    balances[recipient] += output.value
                else:
                    balances[recipient] = output.value

    return utxo, balances


def build_ledger_history(location: str, read_from_index: bool = False, end: int = None):
    """Build history of account balances from Bitcoin blocks"""

    utxo = defaultdict(set)
    balances = defaultdict(float)

    if read_from_index:
        block_iter = read_dat_from_index(index_filepath=location)
    else:
        filepaths = glob.glob(os.path.join(location, 'blk*.dat'))
        block_iter = read_dat(filepaths)

    for block_height, block in enumerate(block_iter):
        if block.height == 30000:
                pass
        if end is not None and block_height >= end - 1:
            return utxo, balances
        # utxo, balances = update_ledger(block, utxo, balances)

    return utxo, balances


def check_order(blocks_dir: str, end: int = None):
    """Checks if all blocks are in ascending order"""

    max_time = 0
    filepaths = glob.glob(os.path.join(blocks_dir, 'blk*.dat'))

    for i, block in enumerate(read_dat(filepaths)):
        if end is not None and i >= end - 1:
            return True
        if block.time_ < max_time:
            return False
        max_time = block.time_

    return True


def get_sorted_index(blocks_dir: str, end: int = None):
    """Build an index of blocks (filename, position)"""

    index = []
    filepaths = glob.glob(os.path.join(blocks_dir, 'blk*.dat'))

    for i, (filepath, position, block) in enumerate(read_dat(filepaths, return_index=True)):
        if end is not None and i >= end:
            break
        index.append((block.time_, filepath, position))

    return sorted(index, key=lambda x: x[0])


def build_index(blocks_dir: str, filename: str):
    index = get_sorted_index(blocks_dir)
    pd.DataFrame(index, columns=['time', 'path', 'position']).to_csv(filename, index=False)


def main():
    blocks_dir = 'datasets/blocks'
    index_filepath = 'datasets/blocks/index.csv'

    if not os.path.isfile(index_filepath):
        build_index(blocks_dir, index_filepath)
    utxo, balances = build_ledger_history(index_filepath, read_from_index=True)
    print(f'{utxo=}')
    print(f'{balances=}')
    # index = get_sorted_index('datasets/blocks', end=600)
    # print(index[585], index[586])


if __name__ == '__main__':
    main()
