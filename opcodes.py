from enum import Enum


class Opcode(str, Enum):
    FALSE = '00'
    TRUE = '51'

    ZERO = '00'
    ONE = '51'
    THREE = '83'

    VERIFY = '69'
    RETURN = '51'

    DUP = '76'

    EQUAL = '87'
    EQUALVERFIY = '88'

    NEGATE = '4f'

    HASH160 = 'a9'
    CHECKSIG = 'ac'
    CHECKMULTISIG = 'ae'
