from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv

from typing import Dict, Tuple
from collections import defaultdict
import numpy as np
#import wandb

class MyBaseCallbacks(DefaultCallbacks):
    """
    Class that defines callbacks for the market maker environments
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RELEVANT_KEYS = [
            "spread",
            "holdings",
            "action",
            "book imbalance",
            "value"
        ]
        self.keys_to_calibrate = [
            "spread",
            "holdings",
            "bid-vol",
            "ask-vol",
            "Mid Price",
        ]
        # for Min_Max, z-score calibration
        self.global_hist_data = defaultdict(list)
        self.calibration_set = defaultdict(list)
        for key in self.keys_to_calibrate:
            self.global_hist_data[key] = []
            #

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        # Ensure this episode has just been started (only initial obs logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # Create lists to store info (in user_data and hist_data)
        episode.user_data = defaultdict(list)
        for key in self.RELEVANT_KEYS:
            episode.user_data[key] = []
        episode.user_data["rewards"] = []
        episode.hist_data["daily_net_cash_performance"] = []
        
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        # Ensure this episode is ongoing
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        agent0_info = episode._agent_to_last_info["agent0"]
        print(agent0_info)
        for k, v in agent0_info.items():
            if k in self.RELEVANT_KEYS:
                episode.user_data[k].append(v)
            if k in self.keys_to_calibrate:
                self.global_hist_data[k].append(v)
        episode.user_data["rewards"].append(agent0_info.get("reward", 0))

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        # Add to hist data and add averages to custom metrics
        for key in ["spread", "holdings"]:
            episode.custom_metrics[f"{key}_avg"] = np.mean(episode.user_data[key])

        episode.custom_metrics["Mean Absolute Position"] = np.mean(np.abs(episode.user_data["holdings"]))
        episode.custom_metrics["Daily Net Cash Performance"] = episode.user_data["value"][-1] - 10_000_000
        
        # Data computation for`episode.hist_data`
        daily_net_cash_performance = episode.user_data["value"][-1] - 10_000_000
        episode.hist_data["daily_net_cash_performance"].append(daily_net_cash_performance)
        
        for key in self.keys_to_calibrate:
            self.calibration_set[f"{key} Max"].append(np.max(self.global_hist_data[key]))
            self.calibration_set[f"{key} Min"].append(np.min(self.global_hist_data[key]))
            self.calibration_set[f"{key} Avg"].append(np.mean(self.global_hist_data[key]))
            self.calibration_set[f"{key} Std"].append(np.std(self.global_hist_data[key]))

        for k, v in self.calibration_set.items():
            print(k, self.calibration_set[k][-1])


    def on_sample_end(
        self,
        *,
        worker: RolloutWorker,
        samples: SampleBatch,
        **kwargs
    ):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """
        Called at the end of Trainable.train().
        """
        pass

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        """Called at the beginning of Policy.learn_on_batch().

        Note: This is called before 0-padding via
        `pad_batch_to_sequences_of_same_size`.

        Args:
            policy (Policy): Reference to the current Policy object.
            train_batch (SampleBatch): SampleBatch to be trained on. You can
                                       mutate this object to modify the samples generated.
            result (dict): A results dict to add custom metrics to.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: MultiAgentEpisode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch],
        **kwargs,
    ):
        """Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            episode (MultiAgentEpisode): Episode object.
            agent_id (str): Id of the current agent.
            policy_id (str): Id of the current policy for the agent.
            policies (dict): Mapping of policy id to policy objects. In single
                             agent mode there will only be a single "default_policy".
            postprocessed_batch (SampleBatch): The postprocessed sample batch
                                               for this agent. You can mutate this
                                               object to apply your own trajectory
                                               postprocessing.
            original_batches (Dict[str, SampleBatch]): Unpostprocessed trajectory data.
            kwargs: Forward compatibility placeholder.
        """
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0

        episode.custom_metrics["num_batches"] += 1
