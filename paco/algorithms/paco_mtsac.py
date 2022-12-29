# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parameter Compositional Multi task Soft Actor Critic Algorithm with online Resetting."""

from absl import logging
import numpy as np
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork, QRNNNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops

ActionType = Enum('ActionType', ('Discrete', 'Continuous', 'Mixed'))

SacActionState = namedtuple(
    "SacActionState", ["actor_network", "critic"], default_value=())

SacCriticState = namedtuple("SacCriticState", ["critics", "target_critics"])

SacState = namedtuple(
    "SacState", ["action", "actor", "critic"], default_value=())

SacCriticInfo = namedtuple("SacCriticInfo", ["critics", "target_critic"])

SacActorInfo = namedtuple(
    "SacActorInfo", ["actor_loss", "neg_entropy"], default_value=())

SacInfo = namedtuple(
    "SacInfo", [
        "observation", "reward", "step_type", "discount", "action",
        "action_distribution", "actor", "critic", "alpha", "log_pi"
    ],
    default_value=())

SacLossInfo = namedtuple(
    'SacLossInfo', ('actor', 'critic', 'alpha', 'task_reg', 'task_loss'),
    default_value=())


def _set_target_entropy(name, target_entropy, flat_action_spec):
    """A helper function for computing the target entropy under different
    scenarios of ``target_entropy``.

    Args:
        name (str): the name of the algorithm that calls this function.
        target_entropy (float|Callable|None): If a floating value, it will return
            as it is. If a callable function, then it will be called on the action
            spec to calculate a target entropy. If ``None``, a default entropy will
            be calculated.
        flat_action_spec (list[TensorSpec]): a flattened list of action specs.
    """
    if target_entropy is None or callable(target_entropy):
        if target_entropy is None:
            target_entropy = dist_utils.calc_default_target_entropy
        target_entropy = np.sum(list(map(target_entropy, flat_action_spec)))
        logging.info("Target entropy is calculated for {}: {}.".format(
            name, target_entropy))
    else:
        logging.info("User-supplied target entropy for {}: {}".format(
            name, target_entropy))
    return target_entropy


@alf.configurable
class PaCoMTSacAlgorithm(OffPolicyAlgorithm):
    r"""Multi task SAC used in PaCo with w-resetting, modified based on SAC implementation
    in ALF.
    
    There are 3 points different with ``tf_agents.agents.sac.sac_agent``:

    
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 task_encoding_network_cls=None,
                 q_network_cls=QNetwork,
                 reward_weights=None,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 env=None,
                 num_tasks=1,
                 reg_weight=0,
                 fixed_alpha=None,
                 temp_reweight=False,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 min_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 task_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 separate_task_loss=False,
                 mask_extreme_task=False,
                 reset_task_while_extreme=False,
                 task_loss_threshold=3e3,
                 reset_task_method="interpolate",
                 name="SacAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called
                to sample continuous actions. All of its output specs must be
                continuous. Note that we don't need a discrete actor network
                because a discrete action can simply be sampled from the Q values.
            critic_network_cls (Callable): is used to construct critic network.
                for estimating ``Q(s,a)`` given that the action is continuous.
            q_network (Callable): is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``alf.get_config_value(TrainerConfig.epsilon_greedy)``
            use_entropy_reward (bool): whether to include entropy as reward
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            initial_log_alpha (float): initial value for variable ``log_alpha``.
            max_log_alpha (float|None): if not None, ``log_alpha`` will be
                capped at this value.
            target_entropy (float|Callable|None): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated. For the mixed action type, discrete action and
                continuous action will have separate alphas and target entropies,
                so this argument can be a 2-element list/tuple, where the first
                is for discrete action and the second for continuous action.
            prior_actor_ctor (Callable): If provided, it will be called using
                ``prior_actor_ctor(observation_spec, action_spec, debug_summaries=debug_summaries)``
                to constructor a prior actor. The output of the prior actor is
                the distribution of the next action. Two prior actors are implemented:
                ``alf.algorithms.prior_actor.SameActionPriorActor`` and
                ``alf.algorithms.prior_actor.UniformPriorActor``.
            target_kld_per_dim (float): ``alpha`` is dynamically adjusted so that
                the KLD is about ``target_kld_per_dim * dim``.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between
                ``[-dqda_clipping, dqda_clipping]``. Will not perform clipping if
                ``dqda_clipping == 0``.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        self._num_critic_replicas = num_critic_replicas
        self._calculate_priority = calculate_priority
        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy
        self._reg_weight = reg_weight
        self._separate_task_loss = separate_task_loss
        self._mask_extreme_task = mask_extreme_task
        self._reset_task_while_extreme = reset_task_while_extreme
        self._extreme_task_mask = None
        self._extreme_task_count = torch.zeros(num_tasks)
        self._task_loss_threshold = task_loss_threshold

        assert (reset_task_method in ["interpolate", "copy"])
        self._reset_task_method = reset_task_method

        critic_networks, actor_network, self._act_type, task_encoding_network = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            critic_network_cls, q_network_cls, task_encoding_network_cls)

        self._use_entropy_reward = use_entropy_reward

        if reward_spec.numel > 1:
            assert not use_entropy_reward, (
                "use_entropy_reward=True is not supported for multidimensional reward"
            )
            assert self._act_type == ActionType.Continuous, (
                "Only continuous action is supported for multidimensional reward"
            )

        self._num_of_tasks = num_tasks
        self._fixed_alpha = fixed_alpha
        self._use_separate_alpha = fixed_alpha is None
        self._temp_reweight = temp_reweight

        def _init_log_alpha():
            if self._use_separate_alpha:
                return nn.Parameter(
                    torch.tensor(
                        [float(initial_log_alpha)] * self._num_of_tasks))
            else:
                return torch.tensor(
                    [float(self._fixed_alpha)] * self._num_of_tasks).log()

        if self._act_type == ActionType.Mixed:
            # separate alphas for discrete and continuous actions
            log_alpha = type(action_spec)((_init_log_alpha(),
                                           _init_log_alpha()))
        else:
            log_alpha = _init_log_alpha()

        action_state_spec = SacActionState(
            actor_network=(() if self._act_type == ActionType.Discrete else
                           actor_network.state_spec),
            critic=(() if self._act_type == ActionType.Continuous else
                    critic_networks.state_spec))
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=SacState(
                action=action_state_spec,
                actor=(() if self._act_type != ActionType.Continuous else
                       critic_networks.state_spec),
                critic=SacCriticState(
                    critics=critic_networks.state_spec,
                    target_critics=critic_networks.state_spec)),
            predict_state_spec=SacState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None and actor_network is not None:
            if actor_network_cls.func == alf.networks.actor_distribution_networks.ActorDistributionCompositionalNetwork:
                actor_optimizer.add_param_group({
                    'params': list(actor_network._reduced_modules.parameters())
                })
                self.add_optimizer(actor_optimizer, [])
            else:
                self.add_optimizer(actor_optimizer, [actor_network])

        if critic_optimizer is not None:
            if critic_network_cls.func == alf.networks.critic_networks.CriticCompositionalNetwork:
                critic_optimizer.add_param_group({
                    'params':
                        sum([
                            list(critic_networks._networks[i]._reduced_module.
                                 parameters())
                            for i in range(self._num_critic_replicas)
                        ], [])
                })
                self.add_optimizer(critic_optimizer, [])
            else:
                self.add_optimizer(critic_optimizer, [critic_networks])

        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, nest.flatten(log_alpha))

        if task_optimizer is not None:
            self.add_optimizer(task_optimizer, [task_encoding_network])

        self._log_alpha = log_alpha
        if self._act_type == ActionType.Mixed:
            self._log_alpha_paralist = nn.ParameterList(
                nest.flatten(log_alpha))

        if max_log_alpha is not None:
            self._max_log_alpha = torch.tensor(
                [float(max_log_alpha)] * self._num_of_tasks)
        else:
            self._max_log_alpha = None
        if min_log_alpha is not None:
            self._min_log_alpha = min_log_alpha
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._task_encoding_network = task_encoding_network
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._prior_actor = None
        if prior_actor_ctor is not None:
            assert self._act_type == ActionType.Continuous, (
                "Only continuous action is supported when using prior_actor")
            self._prior_actor = prior_actor_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries)
            total_action_dims = sum(
                [spec.numel for spec in alf.nest.flatten(action_spec)])
            self._target_entropy = -target_kld_per_dim * total_action_dims
        else:
            if self._act_type == ActionType.Mixed:
                if not isinstance(target_entropy, (tuple, list)):
                    target_entropy = nest.map_structure(
                        lambda _: target_entropy, self._action_spec)
                # separate target entropies for discrete and continuous actions
                self._target_entropy = nest.map_structure(
                    lambda spec, t: _set_target_entropy(self.name, t, [spec]),
                    self._action_spec, target_entropy)
            else:
                self._target_entropy = _set_target_entropy(
                    self.name, target_entropy, nest.flatten(self._action_spec))

        self._dqda_clipping = dqda_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _get_log_alpha(self, observation):
        log_alpha = self._log_alpha  # n
        if isinstance(observation, dict):
            one_hots = observation['task_id']
        elif isinstance(observation, torch.Tensor):
            one_hots = observation[..., -self._num_of_tasks:]
        assert log_alpha.shape[0] == one_hots.shape[-1] and log_alpha.shape[
            0] == self._num_of_tasks, (
                "Number of tasks in the environment doesn't match")
        current_log_alpha = torch.matmul(
            one_hots,
            log_alpha.unsqueeze(0).t()).squeeze(-1)
        current_log_alpha = torch.clamp_min(current_log_alpha,
                                            self._min_log_alpha)
        return current_log_alpha

    def _get_temp_reweight(self, observation):
        log_alpha = self._log_alpha.detach()
        softmax_temp = torch.nn.functional.softmax(-log_alpha)
        if isinstance(observation, dict):
            one_hots = observation['task_id']
        elif isinstance(observation, torch.Tensor):
            one_hots = observation[..., -self._num_of_tasks:]
        assert softmax_temp.shape[0] == one_hots.shape[
            -1] and softmax_temp.shape[0] == self._num_of_tasks, (
                "Number of tasks in the environment doesn't match")
        temp_reweight = torch.matmul(one_hots,
                                     softmax_temp.unsqueeze(0).t()).squeeze(
                                         -1)  # (num_task, ) -> (batch,)
        return temp_reweight

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       continuous_actor_network_cls, critic_network_cls,
                       q_network_cls, task_encoding_network_cls):
        def _make_parallel(net):
            return net.make_parallel(self._num_critic_replicas)

        def _check_spec_equal(spec1, spec2):
            assert nest.flatten(spec1) == nest.flatten(spec2), (
                "Unmatched action specs: {} vs. {}".format(spec1, spec2))

        discrete_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_discrete
        ]
        continuous_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_continuous
        ]
        if task_encoding_network_cls is not None:
            task_encoding_network = task_encoding_network_cls(
                input_tensor_spec=observation_spec, name="TaskEncodingNetwork")
        else:
            task_encoding_network = None

        if discrete_action_spec and continuous_action_spec:
            # When there are both continuous and discrete actions, we require
            # that acition_spec is a tuple/list ``(discrete, continuous)``.
            assert (isinstance(
                action_spec, (tuple, list)) and len(action_spec) == 2), (
                    "In the mixed case, the action spec must be a tuple/list"
                    " (discrete_action_spec, continuous_action_spec)!")
            _check_spec_equal(action_spec[0], discrete_action_spec)
            _check_spec_equal(action_spec[1], continuous_action_spec)
            discrete_action_spec = action_spec[0]
            continuous_action_spec = action_spec[1]
        elif discrete_action_spec:
            discrete_action_spec = action_spec
        elif continuous_action_spec:
            continuous_action_spec = action_spec

        actor_network = None
        if continuous_action_spec:
            assert continuous_actor_network_cls is not None, (
                "If there are continuous actions, then a ActorDistributionNetwork "
                "must be provided for sampling continuous actions!")
            # Hard Code on type classification to see whether task encoding net is a input of constructor
            if continuous_actor_network_cls.func == alf.networks.actor_distribution_networks.ActorDistributionCompositionalNetwork:
                actor_network = continuous_actor_network_cls(
                    input_tensor_spec=observation_spec,
                    action_spec=continuous_action_spec,
                    task_encoding_net=task_encoding_network)
            else:
                actor_network = continuous_actor_network_cls(
                    input_tensor_spec=observation_spec,
                    action_spec=continuous_action_spec)
            if not discrete_action_spec:
                act_type = ActionType.Continuous
                assert critic_network_cls is not None, (
                    "If only continuous actions exist, then a CriticNetwork must"
                    " be provided!")
                # Hard Code on type classification to see whether task encoding net is a input of constructor
                if critic_network_cls.func == alf.networks.critic_networks.CriticCompositionalNetwork:
                    critic_network = critic_network_cls(
                        input_tensor_spec=(observation_spec,
                                           continuous_action_spec),
                        output_tensor_spec=reward_spec,
                        task_encoding_net=task_encoding_network)
                else:
                    critic_network = critic_network_cls(
                        input_tensor_spec=(observation_spec,
                                           continuous_action_spec),
                        output_tensor_spec=reward_spec)
                critic_networks = _make_parallel(critic_network)

        if discrete_action_spec:
            assert reward_spec.numel == 1, (
                "Discrete action is not supported for multidimensional reward")
            act_type = ActionType.Discrete
            assert len(alf.nest.flatten(discrete_action_spec)) == 1, (
                "Only support at most one discrete action currently! "
                "Discrete action spec: {}".format(discrete_action_spec))
            assert q_network_cls is not None, (
                "If there exists a discrete action, then QNetwork must "
                "be provided!")
            if continuous_action_spec:
                act_type = ActionType.Mixed
                q_network = q_network_cls(
                    input_tensor_spec=(observation_spec,
                                       continuous_action_spec),
                    action_spec=discrete_action_spec)
            else:
                q_network = q_network_cls(
                    input_tensor_spec=observation_spec,
                    action_spec=action_spec)
            critic_networks = _make_parallel(q_network)

        return critic_networks, actor_network, act_type, task_encoding_network

    def _predict_action(self,
                        observation,
                        state: SacActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False):
        """The reason why we want to do action sampling inside this function
        instead of outside is that for the mixed case, once a continuous action
        is sampled here, we should pair it with the discrete action sampled from
        the Q value. If we just return two distributions and sample outside, then
        the actions will not match.
        """
        new_state = SacActionState()
        if self._act_type != ActionType.Discrete:
            continuous_action_dist, actor_network_state = self._actor_network(
                observation, state=state.actor_network)
            new_state = new_state._replace(actor_network=actor_network_state)
            if eps_greedy_sampling:
                continuous_action = dist_utils.epsilon_greedy_sample(
                    continuous_action_dist, epsilon_greedy)
            else:
                continuous_action = dist_utils.rsample_action_distribution(
                    continuous_action_dist)
                # continuous_action = continuous_action * 0.0

        critic_network_inputs = observation
        if self._act_type == ActionType.Mixed:
            critic_network_inputs = (observation, continuous_action)

        q_values = None
        if self._act_type != ActionType.Continuous:
            q_values, critic_state = self._critic_networks(
                critic_network_inputs, state=state.critic)
            new_state = new_state._replace(critic=critic_state)
            if self._act_type == ActionType.Discrete:
                alpha = torch.exp(self._log_alpha).detach()
            else:
                alpha = torch.exp(self._log_alpha[0]).detach()
            # p(a|s) = exp(Q(s,a)/alpha) / Z;
            q_values = q_values.min(dim=1)[0]
            logits = q_values / alpha
            discrete_action_dist = td.Categorical(logits=logits)
            if eps_greedy_sampling:
                discrete_action = dist_utils.epsilon_greedy_sample(
                    discrete_action_dist, epsilon_greedy)
            else:
                discrete_action = dist_utils.sample_action_distribution(
                    discrete_action_dist)

        if self._act_type == ActionType.Mixed:
            # Note that in this case ``action_dist`` is not the valid joint
            # action distribution because ``discrete_action_dist`` is conditioned
            # on a particular continuous action sampled above. So DO NOT use this
            # ``action_dist`` to directly sample an action pair with an arbitrary
            # continuous action anywhere else!
            # However, for computing the log probability of *this* sampled
            # ``action``, it's still valid. It can also be used for summary
            # purpose because of the expectation taken over the continuous action
            # when summarizing.
            action_dist = type(self._action_spec)((discrete_action_dist,
                                                   continuous_action_dist))
            action = type(self._action_spec)((discrete_action,
                                              continuous_action))
        elif self._act_type == ActionType.Discrete:
            action_dist = discrete_action_dist
            action = discrete_action
        else:
            action_dist = continuous_action_dist
            action = continuous_action

        return action_dist, action, q_values, new_state

    def predict_step(self, inputs: TimeStep, state: SacState):
        # test phase output:
        if alf.summary.render.is_rendering_enabled():

            if inputs.is_last():
                task_one_hot = inputs.observation['task_id'].float().cpu()
                curr_task = np.where(np.array(task_one_hot))[1][0]
                exist_succ = inputs.env_info['exist_success'].float().cpu()
                logging.info(
                    "Task id=%s, episode_success=%s" % (curr_task, exist_succ))
                param_weight_matrix = self._actor_network._task_encoding_net._fc_layers[
                    0]._weight.detach().cpu()
                curr_task_param_weight = task_one_hot @ param_weight_matrix.T
                if self._actor_network._task_encoding_net._last_activation is not None:
                    logging.info(
                        "Current task param weights before activation: %s" %
                        curr_task_param_weight)
                    logging.info(self._actor_network._task_encoding_net.
                                 _last_activation)
                    curr_task_param_weight = self._actor_network._task_encoding_net._last_activation(
                        curr_task_param_weight).numpy()
                logging.info(
                    "Current task param weights: %s" % curr_task_param_weight)

        action_dist, action, _, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)

        return AlgStep(
            output=action,
            state=SacState(action=action_state),
            info=SacInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: SacState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        action_dist, action, _, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(
                self._critic_networks, inputs.observation, action,
                state.critic.critics)
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, inputs.observation, action,
                state.critic.target_critics)
            critic_state = SacCriticState(
                critics=critics_state, target_critics=target_critics_state)
            if self._act_type == ActionType.Continuous:
                # During unroll, the computations of ``critics_state`` and
                # ``actor_state`` are the same.
                actor_state = critics_state
            else:
                actor_state = ()
        else:
            actor_state = state.actor
            critic_state = state.critic

        new_state = SacState(
            action=action_state, actor=actor_state, critic=critic_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=SacInfo(action=action, action_distribution=action_dist))

    def _compute_critics(self, critic_net, observation, action, critics_state):
        if self._act_type == ActionType.Continuous:
            observation = (observation, action)
        elif self._act_type == ActionType.Mixed:
            observation = (observation, action[1])  # continuous action
        # discrete/mixed: critics shape [B, replicas, num_actions]
        # continuous: critics shape [B, replicas]
        critics, critics_state = critic_net(observation, state=critics_state)
        return critics, critics_state

    def _actor_train_step(self, inputs: TimeStep, state, action, critics,
                          log_pi, action_distribution):
        neg_entropy = sum(nest.flatten(log_pi))

        if self._act_type == ActionType.Discrete:
            # Pure discrete case doesn't need to learn an actor network
            return (), LossInfo(extra=SacActorInfo(neg_entropy=neg_entropy))

        if self._act_type == ActionType.Continuous:
            critics, critics_state = self._compute_critics(
                self._critic_networks, inputs.observation, action, state)

            if self.has_multidim_reward():
                # Multidimensional reward: [B, replicas, reward_dim]
                critics = critics * self.reward_weights
            # min over replicas
            q_value = critics.min(dim=1)[0]

            continuous_log_pi = log_pi
            cont_alpha = torch.exp(self._get_log_alpha(
                inputs.observation)).detach()
        else:
            # use the critics computed during action prediction for Mixed type
            # ``critics``` is already after min over replicas
            critics_state = ()
            discrete_act_dist = action_distribution[0]
            q_value = discrete_act_dist.probs.detach() * critics
            action, continuous_log_pi = action[1], log_pi[1]
            cont_alpha = torch.exp(self._log_alpha[1]).detach()

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))

        actor_info = LossInfo(
            loss=actor_loss + cont_alpha * continuous_log_pi,
            extra=SacActorInfo(actor_loss=actor_loss, neg_entropy=neg_entropy))
        return critics_state, actor_info

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action (Tensor): discrete actions with shape ``[batch_size]``.
            q_values (Tensor): Q values with shape ``[batch_size, replicas, num_actions]``.
        Returns:
            Tensor: selected Q values with shape ``[batch_size, replicas]``.
        """
        # action shape: [batch_size] -> [batch_size, n, 1]
        action = action.view(q_values.shape[0], 1, -1).expand(
            -1, q_values.shape[1], -1).long()
        return q_values.gather(-1, action).squeeze(-1)

    def _critic_train_step(self, inputs: TimeStep, state: SacCriticState,
                           rollout_info: SacInfo, action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks, inputs.observation, rollout_info.action,
            state.critics)

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks, inputs.observation, action,
            state.target_critics)

        if self.has_multidim_reward():
            sign = self.reward_weights.sign()
            target_critics = (target_critics * sign).min(dim=1)[0] * sign
        else:
            target_critics = target_critics.min(dim=1)[0]

        if self._act_type == ActionType.Discrete:
            critics = self._select_q_value(rollout_info.action, critics)
            target_critics = self._select_q_value(
                action, target_critics.unsqueeze(dim=1))
        elif self._act_type == ActionType.Mixed:
            critics = self._select_q_value(rollout_info.action[0], critics)
            discrete_act_dist = action_distribution[0]
            target_critics = torch.sum(
                discrete_act_dist.probs * target_critics, dim=-1)

        target_critic = target_critics.reshape(inputs.reward.shape)

        target_critic = target_critic.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info

    def _alpha_train_step(self, log_pi, observation):
        alpha_loss = nest.map_structure(
            lambda la, lp, t: la * (-lp - t).detach(),
            self._get_log_alpha(observation), log_pi, self._target_entropy)
        return sum(nest.flatten(alpha_loss))

    def train_step(self, inputs: TimeStep, state: SacState,
                   rollout_info: SacInfo):
        (action_distribution, action, critics,
         action_state) = self._predict_action(
             inputs.observation, state=state.action)

        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                    action_distribution, action)

        if self._act_type == ActionType.Mixed:
            # For mixed type, add log_pi separately
            log_pi = type(self._action_spec)((sum(nest.flatten(log_pi[0])),
                                              sum(nest.flatten(log_pi[1]))))
        else:
            log_pi = sum(nest.flatten(log_pi))

        if self._prior_actor is not None:
            prior_step = self._prior_actor.train_step(inputs, ())
            log_prior = dist_utils.compute_log_probability(
                prior_step.output, action)
            log_pi = log_pi - log_prior

        actor_state, actor_loss = self._actor_train_step(
            inputs, state.actor, action, critics, log_pi, action_distribution)
        critic_state, critic_info = self._critic_train_step(
            inputs, state.critic, rollout_info, action, action_distribution)
        alpha_loss = self._alpha_train_step(log_pi, inputs.observation)

        state = SacState(
            action=action_state, actor=actor_state, critic=critic_state)
        info = SacInfo(
            observation=inputs.observation,
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_distribution,
            actor=actor_loss,
            critic=critic_info,
            alpha=alpha_loss,
            log_pi=log_pi)
        return AlgStep(action, state, info)

    def after_update(self, root_inputs, info: SacInfo):
        self._update_target()
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)
        
        ## We reset task weight in extreme task sets
        if self._reset_task_while_extreme and self._extreme_task_mask is not None:
            extreme_task_set = torch.arange(
                self._num_of_tasks)[self._extreme_task_mask]
            normal_task_mask = ~self._extreme_task_mask
            normal_task_set = torch.arange(
                self._num_of_tasks)[normal_task_mask]
            assert (extreme_task_set.shape[0] > 0)
            for fc in self._task_encoding_network._fc_layers:
                new_weight = fc._weight.data.clone()
                normal_task_weight = new_weight[:,
                                                normal_task_mask]  # K x num_normal
                interpolate_weight = torch.rand(
                    (extreme_task_set.shape[0],
                     normal_task_set.shape[0]))  # num_extreme x num_of_normal
                if self._reset_task_method == "interpolate":
                    interpolate_weight = torch.nn.functional.normalize(
                        interpolate_weight, p=1, dim=-1)
                else:
                    interpolate_weight = torch.nn.functional.one_hot(
                        torch.argmax(interpolate_weight, dim=-1),
                        normal_task_set.shape[0]).float()

                new_weight_extreme = normal_task_weight @ interpolate_weight.T
                new_weight[:, self._extreme_task_mask] = new_weight_extreme
                with torch.no_grad():
                    fc._weight.copy_(new_weight)
            for task_id in extreme_task_set:
                self._extreme_task_count[task_id] += 1
                if self._debug_summaries and alf.summary.should_record_summaries(
                ):
                    with alf.summary.scope(self._name):
                        alf.summary.scalar(
                            "extreme_task_count/" + str(task_id),
                            self._extreme_task_count[task_id])
            self._extreme_task_mask = None
        elif self._extreme_task_mask is not None:
            extreme_task_set = torch.arange(
                self._num_of_tasks)[self._extreme_task_mask]
            for task_id in extreme_task_set:
                self._extreme_task_count[task_id] += 1
                if self._debug_summaries and alf.summary.should_record_summaries(
                ):
                    with alf.summary.scope(self._name):
                        alf.summary.scalar(
                            "extreme_task_count/" + str(task_id),
                            self._extreme_task_count[task_id])
            self._extreme_task_mask = None
            # print("******* reset task weight for extreme task set: ", extreme_task_set, " ********")

    def calc_loss(self, info: SacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor
        # reweight loss with alpha temp
        if self._temp_reweight:
            softmax_temp = self._get_temp_reweight(
                info.observation)  # (batch,)
            actor_loss = actor_loss._replace(
                loss=softmax_temp * actor_loss.loss)
            critic_loss = critic_loss._replace(
                loss=softmax_temp * critic_loss.loss)
        # regularization loss added on task encoding to learn good task embedding
        task_reg_loss = 0.0
        if type(
                self._actor_network
        ) == alf.networks.actor_distribution_networks.ActorDistributionCompositionalNetwork:
            param_weight = self._actor_network._task_encoding_net._fc_layers[
                0]._weight
            if self._actor_network._task_encoding_net._last_activation is not None:
                param_weight = self._actor_network._task_encoding_net._last_activation(
                    param_weight)

            def cost_matrix_cos(x, y, p=2):
                # return the m*n sized cost matrix
                "Returns the matrix of $|x_i-y_j|^p$."
                # un squeeze differently so that the tensors can broadcast
                # dim-2 (summed over) is the feature dim
                x_col = x.unsqueeze(1)
                y_lin = y.unsqueeze(0)

                cos = nn.CosineSimilarity(dim=2, eps=1e-6)
                c = torch.clamp(cos(x_col, y_lin), min=0)

                return c

            sparsity_loss = torch.mean(
                torch.linalg.norm(param_weight, ord=1, dim=0))
            coverage_loss = torch.linalg.norm(
                torch.mean(param_weight, dim=1) - (1.0 / self._num_of_tasks),
                ord=2)
            diversity_loss = torch.mean(
                cost_matrix_cos(param_weight, param_weight))
        
        ### no reg loss in PaCo default version. currently manually tuned. 
        ### TODO: experiment and update on this.
        task_reg_loss = self._reg_weight * (
            diversity_loss + 1.0 * sparsity_loss + 1.0 * coverage_loss)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                if self._act_type == ActionType.Mixed:
                    alf.summary.scalar("alpha/discrete",
                                       self._log_alpha[0].exp())
                    alf.summary.scalar("alpha/continuous",
                                       self._log_alpha[1].exp())
                else:
                    for i in range(self._num_of_tasks):
                        alf.summary.scalar(f"alpha{i}",
                                           self._log_alpha.exp()[i])
                if type(
                        self._actor_network
                ) == alf.networks.actor_distribution_networks.ActorDistributionCompositionalNetwork:
                    param_fc_weight = self._actor_network._task_encoding_net._fc_layers[
                        0]._weight.detach().cpu()
                    if self._actor_network._task_encoding_net._last_activation:
                        param_weight = self._actor_network._task_encoding_net._last_activation(
                            param_fc_weight)
                        alf.summary.images(
                            'param weight FC', param_weight, dataformat='HW')
                        alf.summary.text('param_weight_FC matrix',
                                         str(param_weight))
                    else:
                        alf.summary.images(
                            'param weight Final',
                            param_fc_weight,
                            dataformat='HW')
        loss_info = LossInfo(
            loss=math_ops.add_ignore_empty(
                actor_loss.loss,
                critic_loss.loss + alpha_loss + task_reg_loss),
            priority=critic_loss.priority,
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss,
                task_reg=task_reg_loss))
        if self._separate_task_loss or self._mask_extreme_task:
            loss_info = self._convert_loss_into_task_loss(
                loss_info, info.observation)

        return loss_info

    def _obs_to_env_meta_data(self, observation):
        task_id = observation['task_id'][0]  # B, num_of_task
        env_index = task_id.argmax(dim=-1, keepdim=True)  # B, 1
        unique_env_index, env_index_count = env_index.unique(
            dim=0, return_counts=True)
        return env_index, unique_env_index, env_index_count

    def _convert_loss_into_task_loss(self, info: LossInfo, observation):
        num_tasks = self._num_of_tasks
        loss = info.loss
        loss = torch.mean(loss, dim=0).unsqueeze(-1)
        env_index, unique_env_index, env_index_count = self._obs_to_env_meta_data(
            observation)
        task_loss = torch.zeros((num_tasks, 1),
                                dtype=torch.float).scatter_add_(
                                    0, env_index, loss)
        batch_envs = unique_env_index.squeeze(1)
        task_loss[batch_envs] = task_loss[batch_envs] / env_index_count.float(
        ).unsqueeze(1)

        if self._mask_extreme_task:
            loss_threshold = self._task_loss_threshold
            extreme_task_mask = task_loss.squeeze(-1) > loss_threshold
            extreme_task_set = torch.arange(num_tasks)[extreme_task_mask]
            # if exist extreme task in the set, we 1. mask the corresponding loss to zero 2. reset the corresponding weight for that task
            if extreme_task_set.shape[0]:
                # mask out corresponding task
                self._extreme_task_mask = extreme_task_mask
                extreme_ind_mask = sum(
                    env_index == i for i in extreme_task_set).bool().squeeze(
                        -1)  # shape: (batch,)
                loss_new = info.loss
                loss_new[:, extreme_ind_mask] = 0.0
                info = info._replace(loss=loss_new)

        info = info._replace(
            extra=info.extra._replace(
                task_loss=dict(zip(self._env._task_names, list(task_loss)))))
        return info

    def _calc_critic_loss(self, info: SacInfo):
        # We need to put entropy reward in ``experience.reward`` instead of
        # ``target_critics`` because in the case of multi-step TD learning,
        # the entropy should also appear in intermediate steps!
        # This doesn't affect one-step TD loss, however.
        if self._use_entropy_reward:
            with torch.no_grad():
                entropy_reward = nest.map_structure(
                    lambda la, lp: -torch.exp(la) * lp,
                    self._get_log_alpha(info.observation), info.log_pi)
                entropy_reward = sum(nest.flatten(entropy_reward))
                gamma = self._critic_losses[0].gamma
                info = info._replace(
                    reward=info.reward + entropy_reward * gamma)

        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_losses.append(
                l(info=info,
                  value=critic_info.critics[:, :, i, ...],
                  target_value=critic_info.target_critic).loss)

        critic_loss = math_ops.add_n(critic_losses)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']
