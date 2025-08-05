from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, \
    StateDependentNoiseDistribution, CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution, \
    SquashedDiagGaussianDistribution
from torch import nn
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.policies import ActorCriticPolicy

from .AdeDiagGaussianDistribution import AdeGaussianDistribution
from .PortfolioMasterMlpExtractor import PortfolioMasterMlpExtractor
import torch
import torch.nn as nn

class PortfolioMasterActorCritic(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            stock_num: int,
            net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
            activation_fn: type[nn.Module] = nn.Tanh,
            *args,
            **kwargs
    ):
        self.stock_num = stock_num
        self.model_kwargs = kwargs.pop('model_kwargs', {})
        self.exploration_intensity = self.model_kwargs.pop('exploration_intensity', 0.7)
        super(PortfolioMasterActorCritic, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor=PortfolioMasterMlpExtractor(self.features_dim,self.stock_num, **self.model_kwargs)
    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        indicator_features,assets_features= super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(indicator_features,assets_features)
        return self._get_action_dist_from_latent(latent_pi)
    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        indicator_features,assets_features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(indicator_features,assets_features)
        return self.value_net(latent_vf)
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions,log_std = self.action_net(latent_pi)
        self.log_std=log_std

        if isinstance(self.action_dist, AdeGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            raise ValueError("Invalid action distribution")

    def _build(self, lr_schedule: Schedule) -> None:
        # 重新赋值一下原先在init里面初始化的self.action_dist
        self.action_dim=int(np.prod(self.action_space.shape))
        self.action_dist=AdeGaussianDistribution(self.action_dim, self.exploration_intensity)


        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, AdeGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                # self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
