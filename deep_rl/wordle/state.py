"""
Keep the state in a 1D int array

index[0] = remaining steps
Rest of data is laid out as binary array

[1..27] = whether char has been guessed or not

[[status, status, status, status, status]
 for _ in "ABCD..."]
where status has codes
 [1, 0, 0] - char is definitely not in this spot
 [0, 1, 0] - char is maybe in this spot
 [0, 0, 1] - char is definitely in this spot
"""
import numpy as np

from wordle.const import WORDLE_CHARS, WORDLE_N


WordleState = np.ndarray


def get_nvec(max_turns: int):
    return [max_turns] + [2] * len(WORDLE_CHARS) + [2] * 3 * WORDLE_N * len(WORDLE_CHARS)


def new(max_turns: int) -> WordleState:
    return np.array(
        [max_turns] + [0] * len(WORDLE_CHARS) + [0, 1, 0] * WORDLE_N * len(WORDLE_CHARS),
        dtype=np.int32)


def remaining_steps(state: WordleState) -> int:
    return state[0]


def update(state: WordleState, word: str, goal_word: str) -> WordleState:
    """
    return a copy of state that has been updated to new state

    :param state:
    :param word:
    :param goal_word:
    :return:
    """
    state = state.copy()

    state[0] -= 1
    for i, c in enumerate(word):
        cint = ord(c) - ord(WORDLE_CHARS[0])
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        state[1 + cint] = 1
        if goal_word[i] == c:
            # char at position i = yes, all other chars at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for ocint in range(len(WORDLE_CHARS)):
                if ocint != cint:
                    oc_offset = 1 + len(WORDLE_CHARS) + ocint * WORDLE_N * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]
        elif c in goal_word:
            # Char at position i = no, other chars stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        else:
            # Char at all positions = no
            state[offset:offset + 3 * WORDLE_N] = [1, 0, 0] * WORDLE_N

    return state
