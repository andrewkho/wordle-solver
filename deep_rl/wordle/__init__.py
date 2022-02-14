from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)


# Classic
# ----------------------------------------

register(
    id="WordleEnv10-v0",
    entry_point="wordle.wordle:WordleEnv10",
    max_episode_steps=200,
)

register(
    id="WordleEnv100-v0",
    entry_point="wordle.wordle:WordleEnv100",
    max_episode_steps=500,
)

register(
    id="WordleEnv100OneAction-v0",
    entry_point="wordle.wordle:WordleEnv100OneAction",
    max_episode_steps=500,
)

register(
    id="WordleEnv100TwoAction-v0",
    entry_point="wordle.wordle:WordleEnv100TwoAction",
    max_episode_steps=500,
)

register(
    id="WordleEnv100FullAction-v0",
    entry_point="wordle.wordle:WordleEnv100FullAction",
    max_episode_steps=500,
)

register(
    id="WordleEnv100WithMask-v0",
    entry_point="wordle.wordle:WordleEnv100WithMask",
    max_episode_steps=500,
)

register(
    id="WordleEnv1000-v0",
    entry_point="wordle.wordle:WordleEnv1000",
    max_episode_steps=500,
)

register(
    id="WordleEnv1000WithMask-v0",
    entry_point="wordle.wordle:WordleEnv1000WithMask",
    max_episode_steps=500,
)

register(
    id="WordleEnv1000FullAction-v0",
    entry_point="wordle.wordle:WordleEnv1000FullAction",
    max_episode_steps=500,
)

register(
    id="WordleEnvFull-v0",
    entry_point="wordle.wordle:WordleEnvFull",
    max_episode_steps=500,
)

register(
    id="WordleEnvReal-v0",
    entry_point="wordle.wordle:WordleEnvReal",
    max_episode_steps=500,
)

register(
    id="WordleEnvRealWithMask-v0",
    entry_point="wordle.wordle:WordleEnvRealWithMask",
    max_episode_steps=500,
)
