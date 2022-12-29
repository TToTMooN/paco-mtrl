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
from functools import partial
from random import sample
import alf
from alf.algorithms.data_transformer import RewardNormalizer
import metaworld
import torch
import torch.nn.functional as F
from alf.environments.meta_world_wrapper import SuccessRateWrapper
from alf.nest.utils import NestConcat
from alf.environments import suite_metaworld
from alf.environments import meta_world_wrapper
from alf.networks import ActorDistributionNetwork, ActorDistributionCompositionalNetwork
from alf.networks import CriticNetwork, CriticCompositionalNetwork
from alf.networks import EncodingNetwork, TaskEncodingNetwork
from alf.networks.new_projection_networks import CompositionalNormalProjectionNetwork, CompositionalBetaProjectionNetwork, CompositionalTruncatedProjectionNetwork
from alf.utils import math_ops
from alf.algorithms.td_loss import TDLoss
from alf.optimizers import AdamTF, adam_tf

from alf.examples import sac_conf
from alf.examples.metaworld import metaworld_conf
from alf.utils.schedulers import StepScheduler

from paco.algorithms.paco_mtsac import PaCoMTSacAlgorithm

'''
MT10 order:
reach-v2
push-v2
pick-place-v2
door-open-v2
drawer-open-v2
drawer-close-v2
button-press-topdown-v2
peg-insert-side-v2
window-open-v2
window-close-v2
'''
##########################
### EXPERIMENT CONFIGS ###
task_num = 10
env_name = "mt10"
num_of_param_set = 5 
sample_goal = True
num_of_parallel_envs = 10

### algorithm hyper params ###
# mask and reset
mask_extreme_tasks = True
reset_task_while_extreme = True
reset_task_method = "interpolate"  # interpolate", "copy"
## copy means directly choose one of the other tasks' w-weight.

# group and weighted tasks
separate_task_groups = False  
weighted_task_sample = False

# network structures/compositional designs
projection_type = 'normal'
use_compositional_encoding = True  # use compositional weight for actor
critic_use_compositional_encoding = True  # use compositional weight for critic

# others
separate_task_loss = True
use_task_input = False
last_activation = math_ops.identity
## candidate: math_ops.identity, torch.nn.functional.softmax, torch.nn.functional.normalize
share_action_projection = False
use_identical_init = True
last_kernel_initializer = None
reg_weight = 0.0

# algorithm config
task_schedule = [(3e6, 3e-4), (10e6, 3e-4), (50e6, 3e-4)]
task_lr = StepScheduler(progress_type='env_steps', schedule=task_schedule)
qf_lr = 3e-4
policy_lr = 3e-4
alpha_lr = 1e-4
### EXPERIMENT CONFIGS ###
##########################

if weighted_task_sample:
    task_sample_weight = [0.1] * 10
    task_sample_weight[2] = 0.3
    sample_strategy = 'weighted'
else:
    task_sample_weight = None
    sample_strategy = 'robin'

alf.config(
    'meta_world_wrapper.MultitaskMetaworldWrapper',
    separate_task_info=True,
    task_sample_weight=task_sample_weight)

alf.config('suite_metaworld.load_mt_benchmark', sample_goal=sample_goal)

alf.config(
    'create_environment',
    env_name=env_name,
    num_parallel_environments=num_of_parallel_envs,
    env_load_fn=partial(
        suite_metaworld.load_mt_benchmark,
        sample_strategy=sample_strategy,
    ),
    nonparallel=False)

alf.config(
    'create_eval_environment',
    env_name=env_name,
    num_parallel_environments=1,
    env_load_fn=partial(
        suite_metaworld.load_mt_benchmark,
        sample_strategy='robin',
    ),
    nonparallel=True)

if critic_use_compositional_encoding:
    critic_obs_mask = metaworld_conf.obs_mask
else:
    critic_obs_mask = metaworld_conf.obs_task_mask

if separate_task_groups:
    task_groups = [[0, 1, 3, 4, 5, 6, 7, 8, 9], [2]]
    param_groups = [[1, 2, 3, 4], [0]]
    custom_fc_init = torch.zeros((num_of_param_set, task_num))
    torch.nn.init.normal_(custom_fc_init[1:, :])
    torch.nn.init.zeros_(custom_fc_init[:, 2])
    custom_fc_init[0, 2] = 1.0
else:
    task_groups = None
    param_groups = None
    custom_fc_init = None  #
    # custom_fc_init = F.one_hot(torch.tensor(range(task_num)))
    ### custom onehot init weight, None if init with initializer

# algorithm used classes
alf.config(
    'alf.layers.CompositionalFC',
    kernel_identical_init=use_identical_init,
)

# algorithm class constructors
task_encoding_network_cls = \
    partial(
        TaskEncodingNetwork,
        preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.task_mask),
        last_layer_size=num_of_param_set,
        last_activation=last_activation,
        last_kernel_initializer=last_kernel_initializer,
        custom_fc_init=custom_fc_init,
        task_groups=task_groups,
        param_groups=param_groups)

if projection_type == 'normal':
    # normal projection actor network
    actor_network_cls = \
        partial(
            ActorDistributionCompositionalNetwork,
            observation_preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.obs_mask),
            task_preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.task_mask),
            use_compositional_encoding=use_compositional_encoding, # encoder use compositional task weights or not
            use_task_input=use_task_input,
            num_of_param_set = num_of_param_set,
            fc_layer_params=metaworld_conf.fc_hidden_layers,
            task_fc_layer_params=(),
            continuous_projection_net_ctor=partial(
                CompositionalNormalProjectionNetwork,
                state_dependent_std=True,
                num_of_param_set=num_of_param_set,  #number of task need to be set here
                disable_compositional=share_action_projection,
                scale_distribution=True,
                std_transform=partial(
                    math_ops.clipped_exp, clip_value_min=-10, clip_value_max=2)))
elif projection_type == 'beta':
    # beta projection actor network
    actor_network_cls = \
        partial(
            ActorDistributionCompositionalNetwork,
            observation_preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.obs_mask),
            task_preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.task_mask),
            use_compositional_encoding=use_compositional_encoding, # encoder use compositional task weights or not
            use_task_input=use_task_input,
            num_of_param_set = num_of_param_set,
            fc_layer_params=metaworld_conf.fc_hidden_layers,
            task_fc_layer_params=(),
            continuous_projection_net_ctor=partial(
                CompositionalBetaProjectionNetwork,
                num_of_param_set=num_of_param_set,  #number of task need to be set here
                disable_compositional=share_action_projection,
                min_concentration=1.,
                max_concentration=1000.)) # deault None, options 200
elif projection_type == 'truncated':
    # truncted projection actor network
    actor_network_cls = \
        partial(
            ActorDistributionCompositionalNetwork,
            observation_preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.obs_mask),
            task_preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.task_mask),
            use_compositional_encoding=use_compositional_encoding, # encoder use compositional task weights or not
            use_task_input=use_task_input,
            num_of_param_set = num_of_param_set,
            fc_layer_params=metaworld_conf.fc_hidden_layers,
            task_fc_layer_params=(),
            continuous_projection_net_ctor=partial(
                CompositionalTruncatedProjectionNetwork,
                state_dependent_scale=True,
                num_of_param_set=num_of_param_set,  #number of task need to be set here
                disable_compositional=share_action_projection,
                scale_transform=partial(
                    math_ops.clipped_exp, clip_value_min=-10, clip_value_max=2)))


critic_network_cls = \
    partial(
        CriticCompositionalNetwork,
        observation_preprocessing_combiner=NestConcat(nest_mask=critic_obs_mask),
        task_preprocessing_combiner=NestConcat(nest_mask=metaworld_conf.task_mask),
        use_compositional_encoding=critic_use_compositional_encoding,
        use_task_input=use_task_input,
        num_of_param_set=num_of_param_set,
        joint_fc_layer_params=metaworld_conf.fc_hidden_layers,
        )

alf.config(
    'PaCoMTSacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    task_encoding_network_cls=task_encoding_network_cls,
    target_update_tau=0.005,
    num_tasks=task_num,  # number of task need to be set here
    reg_weight=reg_weight,
    separate_task_loss=separate_task_loss,
    mask_extreme_task=mask_extreme_tasks,
    reset_task_while_extreme=reset_task_while_extreme,
    task_loss_threshold=3e3,
    reset_task_method=reset_task_method,
    temp_reweight=False,
    max_log_alpha=0.0,
    min_log_alpha=-10.0,
    use_entropy_reward=True,
    actor_optimizer=AdamTF(lr=policy_lr),
    critic_optimizer=AdamTF(lr=qf_lr),
    alpha_optimizer=AdamTF(lr=alpha_lr),
    task_optimizer=AdamTF(lr=task_lr))
# task_optimizer=AdamTF(lr=task_lr, gradient_clipping=1e3, clip_by_global_norm=True))

# training config
alf.config(
    'TrainerConfig',
    algorithm_ctor=PaCoMTSacAlgorithm,
    initial_collect_steps=1500 * task_num,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=1280,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=5e6 * task_num,
    num_checkpoints=200,
    use_rollout_state=1,
    evaluate=True,
    eval_interval=50000,
    num_eval_episodes=5 * task_num,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    summarize_output=False,
    summary_interval=1000,
    replay_buffer_length=1000000,
    summarize_action_distributions=False)
