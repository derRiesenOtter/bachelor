import unittest

import numpy as np


def block_decomposition(
    sequence: str, balance: int, min_block_size: int, mapping: dict
):
    mapped_sequence = map_sequence(sequence, mapping)
    pass


def map_sequence(sequence: str, mapping: dict) -> np.ndarray:
    """
    Returns the mapped sequence.

    :param sequence: Amino Acid sequence.
    :type sequence: str
    :param mapping: Mapping for the Amin Acid sequence.
    :type mapping: dict
    :return: Array containing the mapped sequence.
    :rtype: np.ndarray
    """
    return np.array([mapping.get(char) for char in sequence])


def check_block(sequence: str, balance: int, min_block_size: int):
    pass


def create_dict_list(sequence: np.ndarray, block_length: int) -> list:
    block_dict_list = []
    block_dict = {}
    for char in sequence:
        block_dict[int(char)] = block_dict.get(int(char), 0)
    for char in sequence[:block_length]:
        block_dict[int(char)] = block_dict.get(int(char), 0) + 1
    block_dict_list.append(block_dict.copy())
    for i in range(1, len(sequence) - block_length):
        prev_char = int(sequence[i - 1])
        next_char = int(sequence[i + block_length])
        block_dict[prev_char] = block_dict.get(prev_char, 0) - 1
        block_dict[next_char] = block_dict.get(next_char, 0) + 1
        block_dict_list.append(block_dict.copy())
    return block_dict_list


def get_balance(factor_left: dict, factor_right: dict) -> int:
    """
    Returns the balance of two factors.

    :param factor_left: Occurences of each letter of the left factor.
    :type factor_left: dict
    :param factor_right: Occurences of each letter of the right factor.
    :type factor_right: dict
    :return: Balance of two factors.
    :rtype: int
    """
    max_balance = 0
    for key, value in factor_left.items():
        balance = abs(value - factor_right.get(key, 0))
        if balance > max_balance:
            max_balance = balance
    return max_balance


class TestBlockDecomposition(unittest.TestCase):
    def test_one(self):
        result = map_sequence("ABCD", {"A": 0, "B": 1, "C": 2, "D": 0})
        excepted = np.array([0, 1, 2, 0])
        np.testing.assert_array_equal(result, excepted)


class TestMapSequence(unittest.TestCase):
    pass


class TestCheckBlock(unittest.TestCase):
    pass


class TestCreateDictList(unittest.TestCase):
    def test_one(self):
        result = create_dict_list(np.array([0, 0, 1, 1, 2]), 3)
        expected = [{0: 2, 1: 1, 2: 0}, {0: 1, 1: 2, 2: 0}, {0: 0, 1: 2, 2: 1}]
        self.assertEqual(result, expected)


class TestGetBalance(unittest.TestCase):
    def test_one(self):
        result = get_balance({"0": 2, "1": 1}, {"0": 1, "1": 2})
        expected = 1
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
