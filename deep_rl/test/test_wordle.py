import pytest

import wordle.wordle
import wordle.state

TESTWORDS = [
    "APPAA",
    "APPAB",
    "APPAC",
    "APPAD",

    "BPPAB",
    "BPPAC",
    "BPPAD",

    "CPPAB",
    "CPPAC",
    "CPPAD",
]

@pytest.fixture
def wordleEnv():
    env = wordle.wordle.WordleEnvBase(
        words=TESTWORDS,
        max_turns=6,
    )
    return env


def test_reset(wordleEnv):
    wordleEnv.reset(seed=13)


def test_guess_win(wordleEnv):
    wordleEnv.reset(seed=13)
    goal = wordleEnv.goal_word
    new_state, reward, done, _ = wordleEnv.step(goal)
    assert done
    assert wordleEnv.done
    assert reward == 0

    try:
        wordleEnv.step(goal)
        raise ValueError("Shouldn't reach here!")
    except ValueError:
        pass


def test_win_reward(wordleEnv):
    wordleEnv.reset(seed=13)
    goal = wordleEnv.goal_word
    new_state, reward, done, _ = wordleEnv.step((goal+1)%len(wordleEnv.words))
    assert wordle.state.remaining_steps(new_state) == wordleEnv.max_turns-1
    assert not done
    assert not wordleEnv.done
    assert reward == 0

    new_state, reward, done, _ = wordleEnv.step(goal)
    assert wordle.state.remaining_steps(new_state) == wordleEnv.max_turns-2
    assert done
    assert wordleEnv.done
    assert reward == wordle.wordle.REWARD

    try:
        wordleEnv.step(goal)
        raise ValueError("Shouldn't reach here!")
    except ValueError:
        pass


def test_win_reward_6(wordleEnv):
    wordleEnv.reset(seed=13)
    goal = wordleEnv.goal_word

    for i in range(5):
        new_state, reward, done, _ = wordleEnv.step((goal+1)%len(wordleEnv.words))

    new_state, reward, done, _ = wordleEnv.step(goal)

    assert wordleEnv.max_turns - wordle.state.remaining_steps(new_state) == 6
    assert done
    assert wordleEnv.done
    assert reward == wordle.wordle.REWARD


def test_lose_reward(wordleEnv):
    wordleEnv.reset(seed=13)
    goal = wordleEnv.goal_word
    for i in range(1, wordleEnv.max_turns):
        new_state, reward, done, _ = wordleEnv.step((goal + i) % len(wordleEnv.words))
        assert wordle.state.remaining_steps(new_state) == wordleEnv.max_turns-i
        assert not done
        assert not wordleEnv.done
        assert reward == 0

    new_state, reward, done, _ = wordleEnv.step((goal + wordleEnv.max_turns) % len(wordleEnv.words))
    assert wordle.state.remaining_steps(new_state) == 0
    assert done
    assert wordleEnv.done
    assert reward == -wordle.wordle.REWARD

    try:
        wordleEnv.step(goal)
        raise ValueError("Shouldn't reach here!")
    except ValueError:
        pass


def test_step(wordleEnv):
    wordleEnv.reset(seed=13)
    wordleEnv.goal_word = 0

    cur_state = wordleEnv.state
    new_state, reward, done, _ = wordleEnv.step(1)
    assert wordle.state.remaining_steps(cur_state) == wordleEnv.max_turns
    assert wordle.state.remaining_steps(new_state) == wordleEnv.max_turns-1
    # Expect B to be all 1,0,0
    offset = 1+26+3*5*(ord('B')-ord('A'))
    assert tuple(new_state[offset:offset+15]) == tuple([1, 0, 0]*5)

    # Expect A to be right in position 0 4 and maybe otherwise
    offset = 1+26
    assert tuple(new_state[offset:offset+15]) == (0,0,1,
                                                  1,0,0,
                                                  1,0,0,
                                                  0,0,1,
                                                  0,1,0)

    # Expect P to be right in position 2 3 and maybe otherwise
    offset = 1 +26+ 3*5*(ord('P') - ord('A'))
    assert tuple(new_state[offset:offset+15]) == (1,0,0,
                                                  0,0,1,
                                                  0,0,1,
                                                  1,0,0,
                                                  0,1,0)

    # Expect C to be maybes
    offset = 1 +26+ 3*5*(ord('C') - ord('A'))
    assert tuple(new_state[offset:offset+15]) == (1,0,0,
                                                  1,0,0,
                                                  1,0,0,
                                                  1,0,0,
                                                  0,1,0)
    cur_state = wordleEnv.state
    new_state, reward, done, _ = wordleEnv.step(1)
    assert wordle.state.remaining_steps(cur_state) == wordleEnv.max_turns-1
    assert wordle.state.remaining_steps(new_state) == wordleEnv.max_turns-2
    # Expect B to be all 1,0,0
    offset = 1+26+3*5*(ord('B')-ord('A'))
    assert tuple(new_state[offset:offset+15]) == tuple([1, 0, 0]*5)

    # Expect A to be right in position 0 4 and maybe otherwise
    offset = 1+26
    assert tuple(new_state[offset:offset+15]) == (0,0,1,
                                                  1,0,0,
                                                  1,0,0,
                                                  0,0,1,
                                                  0,1,0)

    # Expect P to be right in position 2 3 and maybe otherwise
    offset = 1+26 + 3*5*(ord('P') - ord('A'))
    assert tuple(new_state[offset:offset+15]) == (1,0,0,
                                                  0,0,1,
                                                  0,0,1,
                                                  1,0,0,
                                                  0,1,0)

    new_state, reward, done, _ = wordleEnv.step(2)
    assert wordle.state.remaining_steps(new_state) == wordleEnv.max_turns-3
    # Expect B to be all 1,0,0
    offset = 1+26+3*5*(ord('B')-ord('A'))
    assert tuple(new_state[offset:offset+15]) == tuple([1, 0, 0]*5)

    # Expect C to be all 1,0,0
    offset = 1+26+3*5*(ord('C')-ord('A'))
    assert tuple(new_state[offset:offset+15]) == tuple([1, 0, 0]*5)

    # Expect A to be right in position 0 4 and maybe otherwise
    offset = 1+26
    assert tuple(new_state[offset:offset+15]) == (0,0,1,
                                                  1,0,0,
                                                  1,0,0,
                                                  0,0,1,
                                                  0,1,0)

    # Expect P to be right in position 2 3 and maybe otherwise
    offset = 1+26 + 3*5*(ord('P') - ord('A'))
    assert tuple(new_state[offset:offset+15]) == (1,0,0,
                                                  0,0,1,
                                                  0,0,1,
                                                  1,0,0,
                                                  0,1,0)

    new_state, reward, done, _ = wordleEnv.step(0)
    assert wordle.state.remaining_steps(new_state) == wordleEnv.max_turns-4
    assert done
    assert wordleEnv.done
    assert reward == wordle.wordle.REWARD
