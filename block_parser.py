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
    data = data.hex().upper()
    return data


def read_varint(file):
    bytes_ = file.read(1)
    bInt = int(bytes_.hex(), 16)
    c = 0
    data = ''

    if bInt < 253:
        c = 1
        data = bytes_.hex().upper()
    elif bInt == 253:
        c = 3
    elif bInt == 254:
        c = 5
    elif bInt == 255:
        c = 9

    for _ in range(1, c):
        data = file.read(1).hex().upper() + data

    return data


dirA = 'D:/Bitcoin/blocks/'  # Directory where blk*.dat files are stored
# dirA = sys.argv[1]
dirB = './result/'  # Directory where to save parsing results
# dirA = sys.argv[2]

# fList = [x for x in os.listdir(dirA) if x.endswith('.dat') and x.startswith('blk')]
# fList.sort()
# fList = [r"blk00000.dat"]

for filename in sorted(x for x in os.listdir(dirA) if x.endswith('.dat') and x.startswith('blk')):
    resList = []
    filepath = dirA + filename

    msg = 'Start ' + filepath + ' in ' + str(datetime.datetime.now())
    print(msg)

    with open(filepath, 'rb') as f:
        tmpHex = ''

        while f.tell() < os.path.getsize(filepath):
            msg += 'Magic number = ' + read_bytes(f, 4) + '\n'
            msg += 'Block size = ' + read_bytes(f, 4) + '\n'

            tmpPos3 = f.tell()
            tmpHex = read_bytes(f, 80, False)
            tmpHex = bytes.fromhex(tmpHex)
            tmpHex = hashlib.new('sha256', tmpHex).digest()
            tmpHex = hashlib.new('sha256', tmpHex).digest()
            tmpHex = tmpHex[::-1].hex().upper()
            msg += 'SHA256 hash of the current block hash = ' + tmpHex + '\n'

            f.seek(tmpPos3, 0)
            msg += 'Version number = ' + read_bytes(f, 4) + '\n'
            msg += 'SHA256 hash of the previous block hash = ' + read_bytes(f, 32) + '\n'
            msg += 'MerkleRoot hash = ' + read_bytes(f, 32) + '\n'

            # MerkleRoot = tmpHex
            msg += 'Time stamp = ' + read_bytes(f, 4) + '\n'
            msg += 'Difficulty = ' + read_bytes(f, 4) + '\n'
            msg += 'Random number = ' + read_bytes(f, 4) + '\n'

            txCount = int(read_varint(f), 16)
            msg += 'Transactions count = ' + str(txCount) + '\n'

            resList.append('')
            tmpHex = ''
            RawTX = ''
            tx_hashes = []
            for k in range(txCount):
                msg += 'TX version number = ' + read_bytes(f, 4) + '\n'

                RawTX = reverse(tmpHex)
                tmpHex = ''
                Witness = False
                b = f.read(1)
                tmpB = b.hex().upper()
                bInt = int(b.hex(), 16)
                if bInt == 0:
                    tmpB = ''
                    f.seek(1, 1)
                    c = 0
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
                resList.append('Inputs count = ' + tmpHex)
                tmpHex += tmpB
                RawTX = RawTX + reverse(tmpHex)
                for m in range(inCount):
                    tmpHex = read_bytes(f, 32)
                    resList.append('TX from hash = ' + tmpHex)
                    RawTX = RawTX + reverse(tmpHex)
                    tmpHex = read_bytes(f, 4)
                    resList.append('N output = ' + tmpHex)
                    RawTX = RawTX + reverse(tmpHex)
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
                        b = f.read(1)
                        b = b.hex().upper()
                        tmpHex = b + tmpHex
                    scriptLength = int(tmpHex, 16)
                    tmpHex = tmpHex + tmpB
                    RawTX = RawTX + reverse(tmpHex)
                    tmpHex = read_bytes(f, scriptLength, False)
                    resList.append('Input script = ' + tmpHex)
                    RawTX = RawTX + tmpHex
                    tmpHex = read_bytes(f, 4, False)
                    resList.append('Sequence number = ' + tmpHex)
                    RawTX = RawTX + tmpHex
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
                    b = f.read(1)
                    b = b.hex().upper()
                    tmpHex = b + tmpHex

                outputCount = int(tmpHex, 16)
                tmpHex = tmpHex + tmpB
                resList.append('Outputs count = ' + str(outputCount))
                RawTX = RawTX + reverse(tmpHex)

                for m in range(outputCount):
                    tmpHex = read_bytes(f, 8)
                    Value = tmpHex
                    RawTX = RawTX + reverse(tmpHex)
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
                        b = f.read(1)
                        b = b.hex().upper()
                        tmpHex = b + tmpHex
                    scriptLength = int(tmpHex, 16)
                    tmpHex = tmpHex + tmpB
                    RawTX = RawTX + reverse(tmpHex)
                    tmpHex = read_bytes(f, scriptLength, False)
                    resList.append('Value = ' + Value)
                    resList.append('Output script = ' + tmpHex)
                    RawTX = RawTX + tmpHex
                    tmpHex = ''

                if Witness:
                    for m in range(inCount):
                        tmpHex = read_varint(f)
                        WitnessLength = int(tmpHex, 16)
                        for j in range(WitnessLength):
                            tmpHex = read_varint(f)
                            WitnessItemLength = int(tmpHex, 16)
                            tmpHex = read_bytes(f, WitnessItemLength)
                            resList.append(
                                'Witness ' + str(m) + ' ' + str(j) + ' ' + str(WitnessItemLength) + ' ' + tmpHex)
                            tmpHex = ''

                Witness = False
                tmpHex = read_bytes(f, 4)
                msg += 'Lock time = ' + tmpHex
                RawTX = RawTX + reverse(tmpHex)
                tmpHex = RawTX
                tmpHex = bytes.fromhex(tmpHex)
                tmpHex = hashlib.new('sha256', tmpHex).digest()
                tmpHex = hashlib.new('sha256', tmpHex).digest()
                tmpHex = tmpHex[::-1]
                tmpHex = tmpHex.hex().upper()
                resList.append('TX hash = ' + tmpHex)
                tx_hashes.append(tmpHex)
                resList.append('')
                tmpHex = ''
                RawTX = ''

            tx_hashes = [bytes.fromhex(h) for h in tx_hashes]
            tmpHex = merkle_root(tx_hashes).hex().upper()
            if tmpHex != MerkleRoot:
                print('Merkle roots does not match! >', MerkleRoot, tmpHex)

    output = '\n'.join(resList)
    os.makedirs(dirB, exist_ok=True)
    with open(dirB + filename.replace('.dat', '.txt'), 'w') as f:
        f.write(output)