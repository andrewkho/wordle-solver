import fire

from pytorch_lightning import Trainer

from a2c.module import AdvantageActorCritic
from a2c.agent import GreedyActorCriticAgent


def main(
        checkpoint: str,
):
    print("Loading from checkpoint", checkpoint, "...")
    model = AdvantageActorCritic.load_from_checkpoint(checkpoint)

    agent = GreedyActorCriticAgent(model.net)
    env = model.env
    print("Got env with", len(env.words), "words!")
    state = env.reset()

    while True:
        state = env.reset()
        goal_word = input("Give me a goal word: ")
        try:
            env.set_goal_word(goal_word.upper())
        except:
            print("Goal word", goal_word, "not found in env words!")
            continue

        done = False
        for i in range(6):
            action = agent(state, "cpu")[0]
            state, reward, done, _ = env.step(action)
            print(f"Turn {i+1}: {env.words[action]} ({action}), reward ({reward})")
            if done:
                if reward >= 0:
                    print(f"Done! took {i+1} guesses!")
                else:
                    print(f"LOSE! took {i + 1} guesses!")
                break


if __name__ == '__main__':
    fire.Fire(main)