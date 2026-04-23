"""Frozen token vocabulary for the included data."""


VOCAB = {
    "charlie": 0,
    "heidi": 1,
    "sophia": 2,
    "boy": 3,
    "girl": 4,
    "someone": 5,
    "chess": 6,
    "hide-and-seek": 7,
    "soccer": 8,
    "football": 9,
    "game": 10,
    "puzzle": 11,
    "ball": 12,
    "doll": 13,
    "jigsaw": 14,
    "toy": 15,
    "bathroom": 16,
    "bedroom": 17,
    "playground": 18,
    "shower": 19,
    "street": 20,
    "inside": 21,
    "outside": 22,
    "wins": 23,
    "loses": 24,
    "beats": 25,
    "plays": 26,
    "won": 27,
    "lost": 28,
    "played": 29,
    "is": 30,
    "well": 31,
    "badly": 32,
    "ease": 33,
    "difficulty": 34,
    "with": 35,
    "to": 36,
    "at": 37,
    "in": 38,
    "by": 39,
}

def token_to_index(token: str) -> int:
    """Map one token to its fixed integer index."""

    if token not in VOCAB:
        raise ValueError(f"Unknown token: {token}")
    return VOCAB[token]
