import fire
import torch
from pytorch_lightning import LightningModule, Trainer

from deep_q.dqn import DQNLightning

AVAIL_GPUS = min(1, torch.cuda.device_count())


def main(env: str = "WordleEnv100-v0"):
    model = DQNLightning(
        env=env
    )
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=10000,
        val_check_interval=100,
    )

    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(main)
