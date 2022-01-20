from dataclasses import dataclass

import fire
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from deep_q.dqn import DQNLightning
from deep_q.experience import SequenceReplay

AVAIL_GPUS = min(1, torch.cuda.device_count())


def main(
        resume_from_checkpoint: str = None,
        initialize_winning_replays: str = None,
        env: str = "WordleEnv100-v0",
        deep_q_network: str = 'SumChars',
        max_epochs: int = 500,
        checkpoint_every_n_epochs: int = 1000,
        num_workers: int = 0,
        hidden_size: int = 256,
        lr: float = 1.e-3,
        weight_decay: float = 1.e-5,
        last_frame_cutoff: float=0.8,
        max_eps: float=1.,
        min_eps: float=0.01,
        episode_length: int = 512,
        batch_size: int = 512,
):
    model = DQNLightning(
        initialize_winning_replays=initialize_winning_replays,
        deep_q_network=deep_q_network,
        env=env,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        episode_length=episode_length,
        hidden_size=hidden_size,
        num_workers=num_workers,
        eps_start=max_eps,
        eps_end=min_eps,
        eps_last_frame=int(max_epochs*last_frame_cutoff),
    )

    @dataclass
    class SaveBufferCallback(Callback):
        buffer: SequenceReplay

        def on_train_end(self, trainer, pl_module):
            path = f'{trainer.log_dir}/checkpoints'
            fname = 'sequence_buffer.pkl'
            self.buffer.save_winners(f'{path}/{fname}')

    save_buffer_callback = SaveBufferCallback(buffer=model.buffer)
    model_checkpoint = ModelCheckpoint(every_n_epochs=checkpoint_every_n_epochs)
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=max_epochs,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, save_buffer_callback],
        resume_from_checkpoint=resume_from_checkpoint,
    )

    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(main)
