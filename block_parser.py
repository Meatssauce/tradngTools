import datetime
import fileinput
import glob
import hashlib
import os
import re
import sys
import warnings
from collections import defaultdict
from itertools import pairwise, count
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
                try:
                    yield Block.from_file(block_file, height=i)
                except ValueError as exc:
                    raise EOFError from exc
            i += 1


def read_dat(filepaths: [str], return_index=False):
    try:
        with ReadableFileInput(filepaths, 'rb', verbose=True) as f:
            i = 0
            while True:
                try:
                    if return_index:
                        yield f.openedFilepath(), f.positionInFile(), Block.from_file(f, height=-1)
                    else:
                        yield Block.from_file(f, height=-1)
                except EOFError:
                    break
                i += 1
    except StopIteration:
        return


def update_ledger(block: Block,
                  # utxo: defaultdict[any, set[tuple[str, int]]],
                  balances: defaultdict[str, float]):
    for tx in block.transactions:
        for i, input_ in enumerate(tx.inputs):
            if input_.coinbase:
                continue

            try:
                txo_being_spent = input_.original_output
            except KeyError:
                # basically, give up and just skip deducting inputs if block is out of order
                continue

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
            try:
                recipients = output.recipients
            except ValueError:
                continue

            for recipient in recipients:
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

    return balances


def build_ledger_history(location: str, result_dir: str, read_from_index: bool = False, end: int = None):
    """Build history of account balances from Bitcoin blocks"""
    os.makedirs(result_dir, exist_ok=True)

    prev_frame_time = None
    frames = []
    balances = defaultdict(float)
    i = 0

    if read_from_index:
        block_iter = read_dat_from_index(index_filepath=location)
    else:
        filepaths = glob.glob(os.path.join(location, 'blk*.dat'))
        block_iter = read_dat(filepaths)

    for block_height in count(0, 1):
        try:
            block = next(block_iter)
        except EOFError:
            break

        if end is not None and block_height >= end - 1:
            break

        balances = update_ledger(block, balances)

        if not prev_frame_time:
            prev_frame_time = block.time_
        curr_time = block.time_

        # record a snapshot of balances
        if datetime.timedelta(seconds=curr_time - prev_frame_time) >= datetime.timedelta(weeks=1):
            prev_frame_time = curr_time
            time_label = pd.to_datetime(curr_time, unit='s')
            df = pd.DataFrame(balances.values(), index=list(balances.keys()), columns=[time_label])
            frames.append(df)

        # save to disk
        if sys.getsizeof(frames) > 256 * 1024:
            pd.concat(frames).to_csv(os.path.join(result_dir, f'ledger{i:03}.csv'))
            frames = []
            i += 1

    pd.concat(frames).to_csv(os.path.join(result_dir, f'ledger{i:03}.csv'))


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


# def get_sorted_index(blocks_dir: str, end: int = None):
#     """Build an index of blocks (filename, position)"""
#
#     block_info_dict = {}
#     filepaths = glob.glob(os.path.join(blocks_dir, 'blk*.dat'))
#
#     for i, (filepath, position, block) in enumerate(read_dat(filepaths, return_index=True)):
#         if end is not None and i >= end:
#             break
#         block_info_dict[block.prev_block_hash] = (block, filepath, position)
#
#     curr = min([i for i in block_info_dict.values()], key=lambda x: x[0].time_)
#     chain = [curr]
#
#     while len(chain) < len(block_info_dict):
#         next_ = block_info_dict[curr[0].id]
#         chain.append(next_)
#         curr = next_
#
#     return chain


def build_index(blocks_dir: str, filename: str):
    index = get_sorted_index(blocks_dir)
    pd.DataFrame(index, columns=['time', 'path', 'position']).to_csv(filename, index=False)


def main():
    blocks_dir = 'datasets/blocks'
    index_filepath = 'datasets/blocks/index.csv'
    results_dir = 'results'

    if not os.path.isfile(index_filepath):
        build_index(blocks_dir, index_filepath)
    build_ledger_history(index_filepath, result_dir=results_dir, read_from_index=True, end=20000)
    # print(f'{balances=}')
    # index = get_sorted_index('datasets/blocks', end=600)
    # print(index[585], index[586])


if __name__ == '__main__':
    main()
