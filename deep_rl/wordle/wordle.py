import math
from dataclasses import dataclass
from typing import Optional, List

import gym
from gym import spaces
import numpy as np

VALID_WORDS_PATH = '../data/wordle_words.txt'
WORDLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
WORDLE_N = 5
REWARD = 10


def _load_words(limit: Optional[int]=None) -> List[str]:
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines
        else:
            return lines[:limit]


@dataclass
class WordleState:
    """
    Keep the state in a 1D vec to keep it performant.
    
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
    vec: np.ndarray

    @classmethod
    def get_nvec(cls, max_turns: int):
        return [max_turns] + [2] * len(WORDLE_CHARS) + [2] * 3 * WORDLE_N * len(WORDLE_CHARS)

    @classmethod
    def new(cls, max_turns: int):
        return WordleState(
            np.array([max_turns] + [0] * len(WORDLE_CHARS) + [0, 1, 0] * WORDLE_N * len(WORDLE_CHARS),
                     dtype=np.int32)
        )

    def copy(self):
        return WordleState(self.vec.copy())

    def remaining_steps(self) -> int:
        return self.vec[0]

    def update(self, word: str, goal_word: str):
        self.vec[0] -= 1
        for i, c in enumerate(word):
            cint = ord(c) - ord(WORDLE_CHARS[0])
            offset = 1 + len(WORDLE_CHARS) + cint*WORDLE_N*3
            self.vec[1+cint] = 1
            if goal_word[i] == c:
                # char at position i = yes, all other chars at position i == no
                self.vec[offset+3*i:offset+3*i+3] = [0, 0, 1]
                for ocint in range(len(WORDLE_CHARS)):
                    if ocint != cint:
                        oc_offset = 1 + len(WORDLE_CHARS) + ocint*WORDLE_N*3
                        self.vec[oc_offset+3*i:oc_offset+3*i+3] = [1, 0, 0]
            elif c in goal_word:
                # Char at position i = no, other chars stay as they are
                self.vec[offset+3*i:offset+3*i+3] = [1, 0, 0]
            else:
                # Char at all positions = no
                self.vec[offset:offset+3*WORDLE_N] = [1, 0, 0]*WORDLE_N


class WordleEnvBase(gym.Env):
    """
    Actions:
        Can play any 5 letter word in vocabulary
        * 13k for full vocab
    State space is defined as:
        * 6 possibilities for turns (WORDLE_TURNS)
        * For each in VALID_CHARS [A-Z] can be in one of 3^WORDLE_N states: (No, Maybe, Yes)
        for full game, this is (3^5)^26
        Each state has 1 + 5*26 possibilities
    Reward:
        Reward is 10 for guessing the right word, -10 for not guessing the right word after 6 guesses.
    Starting State:
        Random goal word
        Initial state with turn 0, all chars MAYBE
    Episode Termination:

        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    #metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, words: List[str], max_turns: int, frequencies: Optional[List[float]]=None):
        assert all(len(w) == WORDLE_N for w in words), f'Not all words of length {WORDLE_N}, {words}'
        self.words = words
        self.max_turns = max_turns

        self.frequencies = None
        if frequencies:
            assert len(words) == len(frequencies), f'{len(words), len(frequencies)}'
            self.frequencies = np.array(frequencies, dtype=np.float32) / sum(frequencies)

        self.action_space = spaces.Discrete(len(self.words))
        self.observation_space = spaces.MultiDiscrete(WordleState.get_nvec(self.max_turns))
        self._initial_state = WordleState.new(self.max_turns)

        self.done = True
        self.goal_word: int = -1

        # self.viewer = None
        self.state: WordleState = None

    def step(self, action):
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        self.state.update(self.words[action],
                          self.words[self.goal_word])

        reward = 0
        if action == self.goal_word:
            self.done = True
            #reward = REWARD
            if self.state.remaining_steps() == self.max_turns-1:
                reward = 0#-10*REWARD  # No reward for guessing off the bat
            else:
                #reward = REWARD*(self.state.remaining_steps() + 1) / self.max_turns
                reward = REWARD
        elif self.state.remaining_steps() == 0:
            self.done = True
            reward = -REWARD

        return self.state.copy(), reward, self.done, {"goal_id": self.goal_word}

    def reset(self, seed: Optional[int] = None):
        self.state = WordleState.new(self.max_turns)
        self.done = False
        self.goal_word = int(np.random.random()*len(self.words))
        #self.goal_word = np.random.choice(len(self.words), p=self.frequencies)
        #self.goal_word = np.random.choice(10)

        return self.state.copy()


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=6)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6)


class WordleEnv1000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6)


class WordleEnv(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), max_turns=6)
