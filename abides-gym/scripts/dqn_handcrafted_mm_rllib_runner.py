import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb
from abides_gym.envs.markets_mm_custom_metrics import MyBaseCallbacks

api_key = wandb.api.api_key

# Import custom model
from my_dqn_model import CustomDQNTorchModel
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_dqn_model", CustomDQNTorchModel)

# Import environment
import abides_gym
from abides_gym.envs.markets_mm_basic_environment_v0 import SubGymMarketsMmBasicEnv_v0

from ray.tune.registry import register_env

register_env(
    "markets-mm-basic-v0",
    lambda config: SubGymMarketsMmBasicEnv_v0(**config),
)

ray.shutdown()
ray.init(num_cpus=1)

name_xp = "dqn_mm_basic_v0_with_norm"
tune.run(
    "DQN",
    name=name_xp,
    resume=False,
    stop={"training_iteration": 100}, 
    checkpoint_at_end=True, 
    checkpoint_freq=20, 
    config={
        "callbacks": MyBaseCallbacks,
        "env": "markets-mm-basic-v0",
        "env_config": {
            "mkt_close": "16:00:00",
            "timestep_duration": "60s",
            "order_fixed_size": 500,
        },
        "seed": 1,
        "num_gpus": 0,
        "num_workers": 0,
        "gamma": 0.999,
        "lr": 0.001,
        "framework": "torch",
        "hiddens": [50, 20],#[128, 64, 32],
        "model": {
            "custom_model": "my_dqn_model",  # Nome del modello personalizzato registrato
            "fcnet_hiddens": [50,20],#[128, 64, 32],     # Dimensione dei layer nascosti
        },
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.05,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 50000,
         },
        "train_batch_size": 64,
        "prioritized_replay": True,
    },
    callbacks=[
        WandbLoggerCallback(
            project="abides_market_maker_00_abm",
            group=name_xp,
            api_key=wandb.api.api_key,
            log_config=False,
        )
    ],
)
