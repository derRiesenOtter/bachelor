import unittest


def block_decomposition(
    sequence: str, balance_threshold: int, min_block_size: int, mapping: dict
) -> list:
    mapped_sequence = map_sequence(sequence, mapping)
    block_decomposition = [] * len(mapped_sequence)
    block_decomposition_rec(
        mapped_sequence,
        block_decomposition,
        0,
        len(mapped_sequence),
        balance_threshold,
        min_block_size,
    )
    return block_decomposition


def block_decomposition_rec(
    sequence: list,
    block_decomposition: list,
    start: int,
    end: int,
    balance_threshold: int,
    min_block_size: int,
):
    seq_length = end - start
    # block_dict = {}
    # for char in sequence[start:end]:
    #     block_dict[char] = block_dict.get(char, 0) + 1
    # max_diff = get_balance(block_dict, block_dict)
    # max_block_size = seq_length
    # if (max_diff - balance_threshold)
    if seq_length < min_block_size:
        return
    for block_size in reversed(range(min_block_size, seq_length + 1)):
        for block_position in range(start, start + seq_length - block_size):
            current_seq = sequence[block_position : block_position + block_size]
            check_block_res = check_block(
                sequence,
                block_position,
                block_position + block_size,
                balance_threshold,
                min_block_size,
            )
            if check_block_res:
                label = get_label(current_seq)
                for pos in range(
                    start + block_position, start + block_position + block_size
                ):
                    block_decomposition[pos] = label
                print(block_decomposition)
                block_decomposition_rec(
                    sequence,
                    block_decomposition,
                    start,
                    block_position,
                    balance_threshold,
                    min_block_size,
                )
                block_decomposition_rec(
                    sequence,
                    block_decomposition,
                    block_size + block_position,
                    end,
                    balance_threshold,
                    min_block_size,
                )


def map_sequence(sequence: str, mapping: dict) -> list:
    """
    Returns the mapped sequence.

    :param sequence: Amino Acid sequence.
    :type sequence: str
    :param mapping: Mapping for the Amin Acid sequence.
    :type mapping: dict
    :return: Array containing the mapped sequence.
    :rtype: np.ndarray
    """
    return [mapping.get(char) for char in sequence]


def check_block(
    sequence: list, start: int, end: int, balance_threshold: int, min_block_size: int
) -> bool:
    """
    Checks each block if it is at least the minimal block size and has a balance
    less than or equal to the balance threshold.

    :param sequence: Block of the sequence being tested.
    :type sequence: np.ndarray
    :param balance_threshold: Balance Threshold.
    :type balance_threshold: int
    :param min_block_size: Minimal block size.
    :type min_block_size: int
    :return: True if block meets criteria. False if not.
    :rtype: bool
    """
    seq_length = end - start
    max_block_size = seq_length - balance_threshold
    if seq_length < min_block_size:
        return False
    for block_size in range(balance_threshold, max_block_size):
        dict_list = create_dict_list(sequence, start, end, block_size)
        for left_block in range(len(dict_list)):
            for right_block in range(left_block + 1, len(dict_list)):
                balance = get_balance(dict_list[left_block], dict_list[right_block])
                if balance > balance_threshold:
                    return False
    return True


def create_dict_list(sequence: list, start: int, end: int, block_length: int) -> list:
    """
    Creates a list of dictionaries that house the number of each mapping
    category per factor of the current block.

    :param sequence: A block of the sequence that is being tested for his
    balance.
    :type sequence: np.ndarray
    :param block_length: The current length that is tested.
    :type block_length: int
    :return: List of dictionaries with counts for each mapping category.
    :rtype: list
    """
    seq_length = end - start
    block_dict_list = []
    block_dict = {}
    for char in sequence[start:end]:
        block_dict[char] = block_dict.get(char, 0)
    for char in sequence[start : start + block_length]:
        block_dict[char] = block_dict.get(char, 0) + 1
    block_dict_list.append(block_dict.copy())
    for i in range(start + 1, seq_length - block_length + 1):

        prev_char = sequence[i - 1]
        next_char = sequence[i + block_length - 1]
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


def get_label(sequence: list) -> int:
    block_dict = {}
    for char in sequence:
        block_dict[char] = block_dict.get(char, 0) + 1
    max_key = 0
    max_value = 0
    for key, value in block_dict.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key


def precalculate_differences(mapped_sequence: list, balance_threshold: int):
    the_list = []
    for block_size in reversed(range(balance_threshold, len(mapped_sequence))):
        block_size_dict_list = create_dict_list(
            mapped_sequence, 0, len(mapped_sequence), block_size
        )
        diff_list = []
        for block_size_dict_left in range(len(block_size_dict_list) - 1):
            diff_list_left_block = []
            for block_size_dict_right in range(
                block_size_dict_left + 1, len(block_size_dict_list)
            ):
                diff = get_diff(
                    block_size_dict_list[block_size_dict_left],
                    block_size_dict_list[block_size_dict_right],
                )
                diff_list_left_block.append(diff)
            diff_list.append(diff_list_left_block)
        the_list.append(diff_list)
    return the_list


def get_diff(left_dict: dict, rigt_dict: dict):
    max_diff = 0
    for key, value in left_dict.items():
        diff = abs(value - rigt_dict.get(key, 0))
        if diff > max_diff:
            max_diff = diff
    return max_diff


# def find_largest_block(the_list):


# class TestBlockDecomposition(unittest.TestCase):
#     def test_fus(self):
#         fus = "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGGRGRGGSGGGGGGGGGGYNRSSGGYEPRGRGGGRGGRGGMGGSDRGGFNKFGGPRDQGSRHDSEQDNSDNNT"
#         apna_mapping = {
#             "A": 4,
#             "C": 4,
#             "D": 1,
#             "E": 1,
#             "F": 3,
#             "G": 4,
#             "H": 4,
#             "I": 4,
#             "K": 2,
#             "L": 4,
#             "M": 4,
#             "N": 4,
#             "P": 4,
#             "Q": 4,
#             "R": 2,
#             "S": 4,
#             "T": 4,
#             "V": 4,
#             "W": 3,
#             "Y": 3,
#             "Ali": 4,
#             "Neg": 1,
#             "Pos": 2,
#             "Aro": 3,
#         }
#         result = block_decomposition(fus, 4, 20, apna_mapping)
#         expected = []
#         self.assertEqual(result, expected)
#


class TestPreCalculateDifferences(unittest.TestCase):
    def test_one(self):
        result = precalculate_differences([0, 0, 1, 1, 2, 2], 4)
        # expected = [[{0: 2, 1: 2, 2: 0}, {0: 1, 1: 2, 2: 1}]]
        expected = [[[1]], [[1, 2], [1]]]
        self.assertEqual(result, expected)

    def test_two(self):
        fus = "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGGRGRGGSGGGGGGGGGGYNRSSGGYEPRGRGGGRGGRGGMGGSDRGGFNKFGGPRDQGSRHDSEQDNSDNNT"
        apna_mapping = {
            "A": 4,
            "C": 4,
            "D": 1,
            "E": 1,
            "F": 3,
            "G": 4,
            "H": 4,
            "I": 4,
            "K": 2,
            "L": 4,
            "M": 4,
            "N": 4,
            "P": 4,
            "Q": 4,
            "R": 2,
            "S": 4,
            "T": 4,
            "V": 4,
            "W": 3,
            "Y": 3,
            "Ali": 4,
            "Neg": 1,
            "Pos": 2,
            "Aro": 3,
        }
        mapped_sequence = map_sequence(fus, apna_mapping)
        result = precalculate_differences(mapped_sequence, 4)
        # expected = [[{0: 2, 1: 2, 2: 0}, {0: 1, 1: 2, 2: 1}]]
        expected = [[[1]], [[1, 2], [1]]]
        self.assertEqual(result, expected)


# class TestBlockDecompositionRec(unittest.TestCase):
#     def test_one(self):
#         result = [0] * 9
#         block_decomposition_rec(
#             [3, 1, 1, 1, 3, 3, 2, 2, 2],
#             result,
#             0,
#             9,
#             0,
#             3,
#         )
#         expected = [0, 1, 1, 1, 0, 0, 2, 2, 2]
#         self.assertEqual(result, expected)
#


class TestMapSequence(unittest.TestCase):
    def test_one(self):
        result = map_sequence("ABCD", {"A": 0, "B": 1, "C": 2, "D": 0})
        excepted = [0, 1, 2, 0]
        self.assertEqual(result, excepted)


class TestCheckBlock(unittest.TestCase):
    def test_one(self):
        result = check_block([0, 0, 1, 1], 0, 4, 2, 2)
        expected = True
        self.assertEqual(result, expected)

    def test_two(self):
        result = check_block([0, 0, 1, 1], 0, 4, 1, 2)
        expected = False
        self.assertEqual(result, expected)


class TestCreateDictList(unittest.TestCase):
    def test_one(self):
        result = create_dict_list([0, 0, 1, 1, 2], 0, 5, 3)
        expected = [{0: 2, 1: 1, 2: 0}, {0: 1, 1: 2, 2: 0}, {0: 0, 1: 2, 2: 1}]
        self.assertEqual(result, expected)

    def test_two(self):
        result = create_dict_list([0, 0, 1, 1, 2], 0, 5, 2)
        expected = [
            {0: 2, 1: 0, 2: 0},
            {0: 1, 1: 1, 2: 0},
            {0: 0, 1: 2, 2: 0},
            {0: 0, 1: 1, 2: 1},
        ]
        self.assertEqual(result, expected)


class TestGetBalance(unittest.TestCase):
    def test_one(self):
        result = get_balance({"0": 2, "1": 1}, {"0": 1, "1": 2})
        expected = 1
        self.assertEqual(result, expected)


class TestGetLabel(unittest.TestCase):
    def test_one(self):
        result = get_label([0, 0, 3, 3, 3, 3, 2, 1, 2])
        expected = 3
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
