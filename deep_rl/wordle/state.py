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
import collections
from typing import List
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


NO = 0
SOMEWHERE = 1
YES = 2


def update_from_mask(state: WordleState, word: str, mask: List[int]) -> WordleState:
    """
    return a copy of state that has been updated to new state

    From a mask we need slighty different logic since we don't know the
    goal word.

    :param state:
    :param word:
    :param goal_word:
    :return:
    """
    state = state.copy()

    prior_yes = []
    prior_maybe = []
    # We need two passes because first pass sets definitely yesses
    # second pass sets the no's for those who aren't already yes
    state[0] -= 1
    for i, c in enumerate(word):
        cint = ord(c) - ord(WORDLE_CHARS[0])
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        state[1 + cint] = 1
        if mask[i] == YES:
            prior_yes.append(c)
            # char at position i = yes, all other chars at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for ocint in range(len(WORDLE_CHARS)):
                if ocint != cint:
                    oc_offset = 1 + len(WORDLE_CHARS) + ocint * WORDLE_N * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]

    for i, c in enumerate(word):
        cint = ord(c) - ord(WORDLE_CHARS[0])
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        if mask[i] == SOMEWHERE:
            prior_maybe.append(c)
            # Char at position i = no, other chars stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        elif mask[i] == NO:
            # Need to check this first in case there's prior maybe + yes
            if c in prior_maybe:
                # Then the maybe could be anywhere except here
                state[offset+3*i:offset+3*i+3] = [1, 0, 0]
            elif c in prior_yes:
                # No maybe, definitely a yes, so it's zero everywhere except the yesses
                for j in range(WORDLE_N):
                    # Only flip no if previously was maybe
                    if state[offset + 3 * j:offset + 3 * j + 3][1] == 1:
                        state[offset + 3 * j:offset + 3 * j + 3] = [1, 0, 0]
            else:
                # Just straight up no
                state[offset:offset+3*WORDLE_N] = [1, 0, 0]*WORDLE_N

    return state


def get_mask(word: str, goal_word: str) -> List[int]:
    # Definite yesses first
    mask = [0, 0, 0, 0, 0]
    counts = collections.Counter(goal_word)
    for i, c in enumerate(word):
        if goal_word[i] == c:
            mask[i] = 2
            counts[c] -= 1

    for i, c in enumerate(word):
        if mask[i] == 2:
            continue
        elif c in counts:
            if counts[c] > 0:
                mask[i] = 1
                counts[c] -= 1
            else:
                for j in range(i+1, len(mask)):
                    if mask[j] == 2:
                        continue
                    mask[j] = 0

    return mask

def update_mask(state: WordleState, word: str, goal_word: str) -> WordleState:
    """
    return a copy of state that has been updated to new state

    :param state:
    :param word:
    :param goal_word:
    :return:
    """
    mask = get_mask(word, goal_word)
    return update_from_mask(state, word, mask)


def update(state: WordleState, word: str, goal_word: str) -> WordleState:
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

