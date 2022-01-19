import fire
import torch
from pytorch_lightning import LightningModule, Trainer

from deep_q.dqn import DQNLightning

AVAIL_GPUS = min(1, torch.cuda.device_count())


def main(
        env: str = "WordleEnv100-v0",
        max_epochs: int = 500,
        num_workers: int = 0,
        val_check_interval: int = 1000,
        batch_size: int = 512,
        hidden_size: int = 256,
        lr: float = 1.e-3,
        weight_decay: float = 1.e-5,
        last_frame_cutoff: float=0.8
):
    model = DQNLightning(
        env=env,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_workers=num_workers,
        eps_last_frame=int(max_epochs*last_frame_cutoff),
    )
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
    )

    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(main)
