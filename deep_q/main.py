import fire
import torch
from pytorch_lightning import LightningModule, Trainer

from deep_q.dqn import DQNLightning

AVAIL_GPUS = min(1, torch.cuda.device_count())


def main(
        env: str = "WordleEnv100-v0",
        max_epochs: int = 500,
        val_check_interval: int = 1000,
        batch_size: int = 64,
        hidden_size: int = 256,
):
    model = DQNLightning(
        env=env,
        batch_size=batch_size,
        hidden_size=hidden_size,
        eps_last_frame=int(max_epochs*0.95),
    )
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
    )

    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(main)
