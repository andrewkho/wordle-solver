import fire

from pytorch_lightning import Trainer

import wordle.state
from a2c.module import AdvantageActorCritic
from a2c.agent import GreedyActorCriticAgent


def main(
        checkpoint: str,
        mode: str = 'goal',
):
    print("Loading from checkpoint", checkpoint, "...")
    model = AdvantageActorCritic.load_from_checkpoint(checkpoint)

    agent = GreedyActorCriticAgent(model.net)
    env = model.env
    print("Got env with", len(env.words), "words!")

    if mode == 'goal':
        goal(agent, env)
    elif mode == 'suggest':
        suggest(agent, env)
    elif mode == 'wordle-site':
        #play_wordle_site()
        pass


def suggest(agent, env):
    print("Interactive mode")
    print("When I ask for <word> <mask>, give me the word you entered\n"
          "and the result, example: stare 21021\n"
          "  where 0 = not in word, 1 = somewhere in this word, 2 = in this spot")
    while True:
        print("Alright, a new game!")
        state = env.reset()
        while True:
            guess = agent(state, "cpu")[0]
            print(f"I suggest", env.words[guess])
            word_mask = input("<word> <mask> or done: ")
            if word_mask.lower() == 'done':
                break
            try:
                word, mask = word_mask.strip().split(' ')
                word = word.upper()
                assert word in env.words
                mask_arr = [int(i) for i in mask]
                assert all(i in (0, 1, 2) for i in mask_arr)
                assert len(mask_arr) == 5

                state = wordle.state.update_from_mask(state, word, mask_arr)
                offset_p = ord('P') - ord('A')
                offset = 27 + offset_p*15
                print(word, mask_arr, state[offset:offset+15])
            except:
                print(f"Failed to parse {word_mask}!")
                continue


def goal(agent, env):
    print("Goal word mode")
    while True:
        state = env.reset()
        goal_word = input("Give me a goal word: ")
        try:
            env.set_goal_word(goal_word.upper())
        except:
            print("Goal word", goal_word, "not found in env words!")
            continue

        for i in range(env.max_turns):
            action = agent(state, "cpu")[0]
            state, reward, done, _ = env.step(action)
            print(f"Turn {i+1}: {env.words[action]} ({action}), reward ({reward})")
            offset_p = ord('P') - ord('A')
            offset = 27 + offset_p * 15
            print(state[offset:offset + 15])
            if done:
                if reward >= 0:
                    print(f"Done! took {i+1} guesses!")
                else:
                    print(f"LOSE! took {i+1} guesses!")
                break


if __name__ == '__main__':
    fire.Fire(main)