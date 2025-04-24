from envs.simple_park import SimplePark

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.connectors.env_to_module import FlattenObservations

def _env_to_module_pipeline(env):
    return FlattenObservations()


config = (
    PPOConfig()
    .environment(SimplePark,
                 env_config={
                    "nrow": 10,
                    "ncol": 12,
                    "vision_range": 7,
                    "disabled_states": [40,41,42,52,53,54,64,65,66,
                            94,95,106,107,118,119,
                            0,12,24,36,48,60],
                    "entrances_states": [59,2,113],
                 })
    .env_runners(env_to_module_connector=_env_to_module_pipeline)
    .training(
        #train_batch_size_per_learner = 2000,
        lr=0.0004,
        entropy_coeff=0.01,
    )
)

results = tune.Tuner(
    "PPO",
    param_space = config,
    run_config=tune.RunConfig(stop={"num_env_steps_sampled_lifetime":2000}),
).fit()
