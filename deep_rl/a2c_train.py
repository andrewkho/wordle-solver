"""Advantage Actor Critic (A2C)"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from a2c.module import AdvantageActorCritic


def cli_main() -> None:
    parser = ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = AdvantageActorCritic.add_model_specific_args(parser)
    args = parser.parse_args()

    with wandb.init(project='wordle-solver'):
        wandb.config.update(args)

        model = AdvantageActorCritic(**args.__dict__)
        # save checkpoints based on avg_reward
        checkpoint_callback = ModelCheckpoint(every_n_train_steps=100)

        seed_everything(123)

        trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=checkpoint_callback)
        trainer.fit(model)


if __name__ == '__main__':
    cli_main()
