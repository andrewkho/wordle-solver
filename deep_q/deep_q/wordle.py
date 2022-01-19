import random
from typing import Optional, List

import gym
from gym import spaces, logger
import numpy as np

VALID_WORDS_PATH = '../data/wordle_words.txt'
WORDLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
WORDLE_N = 5
WORDLE_TURNS = 6
REWARD = 10


def _load_words(limit: Optional[int]=None) -> List[str]:
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines
        else:
            return lines[:limit]


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

    def __init__(self, words: List[str], frequencies: Optional[List[float]]=None):
        assert all(len(w) == WORDLE_N for w in words), f'Not all words of length {WORDLE_N}, {words}'
        self.words = words

        self.frequencies = None
        if frequencies:
            assert len(words) == len(frequencies), f'{len(words), len(frequencies)}'
            self.frequencies = np.array(frequencies, dtype=np.float32) / sum(frequencies)

        self.action_space = spaces.Discrete(len(self.words))
        self.observation_space = spaces.MultiDiscrete([WORDLE_TURNS] + [2]*3*WORDLE_N*len(WORDLE_CHARS))
        self._initial_state = np.array([0] + [0, 1, 0]*WORDLE_N*len(WORDLE_CHARS))

        self.done = True
        self.goal_word: int = -1
        #
        # self.viewer = None
        # self.state = None
        #
        # self.steps_beyond_done = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )

        word = self.words[action]
        goal_word = self.words[self.goal_word]

        self.state[0] += 1
        for i, c in enumerate(word):
            cint = ord(c) - ord(WORDLE_CHARS[0])
            offset = 1 + cint*WORDLE_N*3
            if goal_word[i] == c:
                self.state[offset+3*i:offset+3*i+3] = [0, 0, 1]
                for oc in WORDLE_CHARS:
                    if oc != c:
                        ocint = ord(oc) - ord(WORDLE_CHARS[0])
                        oc_offset = 1+ocint*WORDLE_N*3
                        self.state[oc_offset+3*i:oc_offset+3*i+3] = [1, 0, 0]
            elif c in goal_word[i]:
                self.state[offset:offset+3] = [0, 1, 0]
            else:
                self.state[offset:offset+3*WORDLE_N] = [1, 0, 0]*WORDLE_N

        reward = 0.
        if action == self.goal_word:
            self.done = True
            if self.state[0] == 1:
                reward = 0
            else:
                reward = REWARD#*(WORDLE_TURNS-self.state[0]+1)/WORDLE_TURNS
        elif self.state[0] >= WORDLE_TURNS:
            self.done = True
            reward = -REWARD

        return np.array(self.state, dtype=np.float32).copy(), reward, self.done, {}

    def reset(self, seed: Optional[int] = None):
        #super().reset()
        self.state = self._initial_state.copy()
        self.done = False
        # np.random.seed(seed)
        self.goal_word = np.random.choice(len(self.words), p=self.frequencies)
        #self.goal_word = 7
        #self.goal_word = int(random.uniform(0, len(self.words)))

        return np.array(self.state, dtype=np.float32).copy()

    def getCharsFromAction(self, action: int) -> np.ndarray:
        return self.word_array[action, :]

    # def render(self, mode="human"):
        # screen_width = 600
        # screen_height = 400
        #
        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0
        #
        # if self.viewer is None:
        #     from gym.utils import pyglet_rendering
        #
        #     self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        #     axleoffset = cartheight / 4.0
        #     cart = pyglet_rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = pyglet_rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     l, r, t, b = (
        #         -polewidth / 2,
        #         polewidth / 2,
        #         polelen - polewidth / 2,
        #         -polewidth / 2,
        #     )
        #     pole = pyglet_rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     pole.set_color(0.8, 0.6, 0.4)
        #     self.poletrans = pyglet_rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.viewer.add_geom(pole)
        #     self.axle = pyglet_rendering.make_circle(polewidth / 2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
        #     self.axle.set_color(0.5, 0.5, 0.8)
        #     self.viewer.add_geom(self.axle)
        #     self.track = pyglet_rendering.Line((0, carty), (screen_width, carty))
        #     self.track.set_color(0, 0, 0)
        #     self.viewer.add_geom(self.track)
        #
        #     self._pole_geom = pole
        #
        # if self.state is None:
        #     return None
        #
        # # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = (
        #     -polewidth / 2,
        #     polewidth / 2,
        #     polelen - polewidth / 2,
        #     -polewidth / 2,
        # )
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])
        #
        # return self.viewer.render(return_rgb_array=mode == "rgb_array")

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10))


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100))


class WordleEnv1000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000))


class WordleEnv(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words())
