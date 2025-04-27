from ray.rllib.examples.algorithms.classes.vpg import VPGConfig
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)


parser = add_rllib_example_script_args(
    default_reward=250.0,
    default_iters=1000,
    default_timesteps=750000,
)
parser.set_defaults(enable_new_api_stack=True)


if __name__ == "__main__":
    args = parser.parse_args()

    base_config = (
        VPGConfig()
        .environment("CartPole-v1")
        .training(
            # The only VPG-specific setting. How many episodes per train batch?
            num_episodes_per_train_batch=10,
            # Set other config parameters.
            lr=0.0005,
            # Note that you don't have to set any specific Learner class, because
            # our custom Algorithm already defines the default Learner class to use
            # through its `get_default_learner_class` method, which returns
            # `VPGTorchLearner`.
            # learner_class=VPGTorchLearner,
        )
        # Increase the number of EnvRunners (default is 1 for VPG)
        # or the number of envs per EnvRunner.
        .env_runners(num_env_runners=2, num_envs_per_env_runner=1)
        # Plug in your own RLModule class. VPG doesn't require any specific
        # RLModule APIs, so any RLModule returning `actions` or `action_dist_inputs`
        # from the forward methods works ok.
        # .rl_module(
        #    rl_module_spec=RLModuleSpec(module_class=...),
        # )
    )

    run_rllib_example_script_experiment(base_config, args)