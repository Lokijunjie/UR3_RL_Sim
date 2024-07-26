


import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import EnvContext

# 导入你的环境
from reach_env import ReachEnv
# 环境创建函数
def env_creator(env_config: EnvContext):
    return ReachEnv(env_config)

# 注册自定义环境
tune.register_env("my_custom_env", env_creator)

# 创建 PPOConfig 对象
config = PPOConfig()

# 配置自定义环境
config.environment(
    env="my_custom_env",
    env_config={}
)

# 配置训练超参数
config.training(
    gamma=0.99,
    lr=0.0003,
    num_sgd_iter=10,
    sgd_minibatch_size=64,
    train_batch_size=4000
)

# 配置资源
config.resources(
    num_gpus=0,
    num_cpus_per_worker=1,
    num_gpus_per_worker=0
)

# 配置其他参数
config.rollouts(
    num_rollout_workers=2
)

# 初始化 Ray
ray.init()

# 运行训练
tune.run(
    "PPO",
    config=config.to_dict(),  # 将 PPOConfig 对象转换为字典
    stop={"training_iteration": 100}
)
