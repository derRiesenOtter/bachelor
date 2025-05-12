class Mapping(object):
    Forward = dict()
    Backward = dict()


class APNAMapping(Mapping):
    Forward = {  # other: 0, neg: 1, pos: 2, aro:3
        "A": 0,
        "C": 0,
        "D": 1,
        "E": 1,
        "F": 3,
        "G": 0,
        "H": 0,
        "I": 0,
        "K": 2,
        "L": 0,
        "M": 0,
        "N": 0,
        "P": 0,
        "Q": 0,
        "R": 2,
        "S": 0,
        "T": 0,
        "V": 0,
        "W": 3,
        "Y": 3,
        "Ali": 0,
        "Neg": 1,
        "Pos": 2,
        "Aro": 3,
    }
    Backward = {0: "Ali", 1: "Neg", 2: "Pos", 3: "Aro"}


class ACDFWMapping(Mapping):
    Forward = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 2,
        "F": 3,
        "G": 0,
        "H": 2,
        "I": 3,
        "K": 2,
        "L": 3,
        "M": 3,
        "N": 2,
        "P": 0,
        "Q": 2,
        "R": 2,
        "S": 0,
        "T": 0,
        "V": 3,
        "W": 4,
        "Y": 3,
        "GA": 0,
        "GC": 1,
        "GD": 2,
        "GF": 3,
        "GW": 4,
    }
    Backward = {0: "GA", 1: "GC", 2: "GD", 3: "GF", 4: "GW"}


class ProteinMapping(Mapping):
    Forward = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 0,
    }
    Backward = {
        1: "A",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "K",
        10: "L",
        11: "M",
        12: "N",
        13: "P",
        14: "Q",
        15: "R",
        16: "S",
        17: "T",
        18: "V",
        19: "W",
        0: "Y",
    }
