from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.envs.classes.correlated_actions_env import CorrelatedActionsEnv
from ray.rllib.examples.rl_modules.classes.autoregressive_actions_rlm import (
    AutoregressiveActionsRLM,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

parser = add_rllib_example_script_args(
    default_iters=1000,
    default_timesteps=2000000,
    default_reward=-0.45,
)
parser.set_defaults(enable_new_api_stack=True)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.algo != "PPO":
        raise ValueError("This example only supports PPO.")

    base_config = (
        PPOConfig()
        .environment(CorrelatedActionsEnv)
        .training(
            train_batch_size_per_learner=2000,
            num_epochs=12,
            minibatch_size=256,
            entropy_coeff=0.005,
            lr=0.0003,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=AutoregressiveActionsRLM,
            )
        )
    )

    run_rllib_example_script_experiment(base_config, args)