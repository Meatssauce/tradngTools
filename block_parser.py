# -*- coding: utf-8 -*-
#
# Blockchain parser
# Copyright (c) 2015-2021 Denis Leonov <466611@gmail.com>
#

import os
import datetime
import hashlib


def reverse(arr):
    if len(arr) % 2 != 0:
        return None

    result = ''
    for i in range(len(arr) // 2):
        result = arr[i * 2] + arr[i * 2 + 1] + result
    return result


def merkle_root(lst):  # https://gist.github.com/anonymous/7eb080a67398f648c1709e41890f8c44
    if len(lst) == 1:
        return lst[0]

    sha256d = lambda x: hashlib.sha256(hashlib.sha256(x).digest()).digest()
    hash_pair = lambda x, y: sha256d(x[::-1] + y[::-1])[::-1]

    if len(lst) % 2 == 1:
        lst.append(lst[-1])
    return merkle_root([hash_pair(x, y) for x, y in zip(*[iter(lst)] * 2)])


def read_bytes(file, n, little_endian=True):
    data = file.read(n)
    if little_endian:
        data = data[::-1]
    return data.hex().upper()


def read_varint(file):
    var_int = file.read(1).hex()
    bInt = int(var_int, 16)

    if bInt < 253:
        return var_int.upper()
    if bInt == 253:
        c = 3
    elif bInt == 254:
        c = 5
    elif bInt == 255:
        c = 9
    else:
        return ''
    return ''.join([file.read(1).hex().upper() for _ in range(1, c)][::-1])




dirA = 'D:/Bitcoin/blocks/'  # Directory where blk*.dat files are stored
# dirA = sys.argv[1]
dirB = './result/'  # Directory where to save parsing results
# dirA = sys.argv[2]

# fList = [x for x in os.listdir(dirA) if x.endswith('.dat') and x.startswith('blk')]
# fList.sort()
# fList = [r"blk00000.dat"]

for filename in sorted(x for x in os.listdir(dirA) if x.endswith('.dat') and x.startswith('blk')):
    filepath = dirA + filename

    msg = f'Start {filepath} in {str(datetime.datetime.now())}\n'
    print(msg)

    with open(filepath, 'rb') as f:
        while f.tell() < os.path.getsize(filepath):
            msg += f'Magic number = {read_bytes(f, 4)}\n'
            msg += f'Block size = {read_bytes(f, 4)}\n'

            tmpPos3 = f.tell()
            whatthis = bytes.fromhex(read_bytes(f, 80, False))
            tx_hash = hashlib.new('sha256', whatthis).digest()
            tx_hash = hashlib.new('sha256', tx_hash).digest()
            tx_hash = tx_hash[::-1].hex().upper()
            msg += f'SHA256 hash of the current block hash = {tx_hash}\n'

            f.seek(tmpPos3, 0)
            msg += f'Version number = {read_bytes(f, 4)}\n'
            msg += f'SHA256 hash of the previous block hash = {read_bytes(f, 32)}\n'
            msg += f'MerkleRoot hash = {read_bytes(f, 32)}\n'

            # MerkleRoot = tmpHex
            msg += f'Time stamp = {read_bytes(f, 4)}\n'
            msg += f'Difficulty = {read_bytes(f, 4)}\n'
            msg += f'Random number = {read_bytes(f, 4)}\n'

            tx_count = int(read_varint(f), 16)
            msg += f'Transactions count = {tx_count}\n'

            tx_hashes = []
            for k in range(tx_count):
                msg += f'TX version number = {read_bytes(f, 4)}\n'

                RawTX = reverse(tmpHex)
                Witness = False
                b = f.read(1)
                tmpB = b.hex().upper()

                bInt = int(b.hex(), 16)
                if bInt == 0:
                    tmpB = ''
                    f.seek(1, 1)
                    c = f.read(1)
                    bInt = int(c.hex(), 16)
                    tmpB = c.hex().upper()
                    Witness = True

                c = 0
                if bInt < 253:
                    c = 1
                    tmpHex = hex(bInt)[2:].upper().zfill(2)
                    tmpB = ''
                if bInt == 253:
                    c = 3
                if bInt == 254:
                    c = 5
                if bInt == 255:
                    c = 9
                for j in range(1, c):
                    b = f.read(1).hex().upper()
                    tmpHex = b + tmpHex

                inCount = int(tmpHex, 16)
                msg += 'Inputs count = ' + tmpHex
                tmpHex += tmpB
                RawTX += reverse(tmpHex)
                for m in range(inCount):
                    msg += f'TX from hash = {read_bytes(f, 32)}\n'
                    RawTX += reverse(tmpHex)
                    msg += f'N output = {read_bytes(f, 4)}\n'
                    RawTX += reverse(tmpHex)
                    tmpHex = ''

                    b = f.read(1)
                    tmpB = b.hex().upper()

                    bInt = int(b.hex(), 16)

                    c = 0
                    if bInt < 253:
                        c = 1
                        tmpHex = b.hex().upper()
                        tmpB = ''
                    if bInt == 253:
                        c = 3
                    if bInt == 254:
                        c = 5
                    if bInt == 255:
                        c = 9
                    for j in range(1, c):
                        b = f.read(1).hex().upper()
                        tmpHex = b + tmpHex

                    scriptLength = int(tmpHex, 16)
                    tmpHex += tmpB
                    RawTX += reverse(tmpHex)
                    msg += f'Input script = {read_bytes(f, scriptLength, False)}\n'
                    RawTX += tmpHex
                    msg += f'Sequence number = {read_bytes(f, 4, False)}\n'
                    RawTX += tmpHex
                    tmpHex = ''

                b = f.read(1)
                tmpB = b.hex().upper()
                bInt = int(b.hex(), 16)
                c = 0
                if bInt < 253:
                    c = 1
                    tmpHex = b.hex().upper()
                    tmpB = ''
                if bInt == 253:
                    c = 3
                if bInt == 254:
                    c = 5
                if bInt == 255:
                    c = 9
                for j in range(1, c):
                    b = f.read(1).hex().upper()
                    tmpHex = b + tmpHex

                outputCount = int(tmpHex, 16)
                tmpHex += tmpB
                msg += f'Outputs count = {str(outputCount)}\n'
                RawTX += reverse(tmpHex)

                for m in range(outputCount):
                    Value = read_bytes(f, 8)
                    RawTX += reverse(Value)

                    b = f.read(1)
                    tmpB = b.hex().upper()
                    bInt = int(b.hex(), 16)
                    c = 0
                    if bInt < 253:
                        c = 1
                        tmpHex = b.hex().upper()
                        tmpB = ''
                    if bInt == 253:
                        c = 3
                    if bInt == 254:
                        c = 5
                    if bInt == 255:
                        c = 9
                    for j in range(1, c):
                        b = f.read(1).hex().upper()
                        tmpHex = b + tmpHex

                    scriptLength = int(tmpHex, 16)
                    tmpHex += tmpB

                    RawTX += reverse(tmpHex)
                    msg += f'Value = {Value}\n'
                    msg += f'Output script = {read_bytes(f, scriptLength, False)}\n'
                    RawTX += tmpHex

                if Witness:
                    for m in range(inCount):
                        for j in range(int(read_varint(f), 16)):
                            WitnessItemLength = int(read_varint(f), 16)
                            tmpHex = read_bytes(f, WitnessItemLength)
                            msg += f'Witness {str(m)} {str(j)} {str(WitnessItemLength)} {tmpHex}\n'

                Witness = False
                temp = read_bytes(f, 4)
                msg += f'Lock time = {temp} \n'

                RawTX += reverse(temp)
                whatthis = bytes.fromhex(RawTX)
                tx_hash = hashlib.new('sha256', whatthis).digest()
                tx_hash = hashlib.new('sha256', tx_hash).digest()
                tx_hash = tx_hash[::-1].hex().upper()
                msg += f'TX hash = {tx_hash} \n'

            tx_hashes = [bytes.fromhex(h) for h in tx_hashes]
            tmpHex = merkle_root(tx_hashes).hex().upper()
            if tmpHex != MerkleRoot:
                print('Merkle roots does not match! >', MerkleRoot, tmpHex)

    os.makedirs(dirB, exist_ok=True)
    with open(dirB + filename.replace('.dat', '.txt'), 'w') as f:
        f.write(msg)
