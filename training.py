import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from environments import make_env, make_vec_envs  # Import the functions


'''TO DO:
manca hyperparameter tuning, da fare con optuna
'''

def train_ppo(env_id="HalfCheetah-v5", total_timesteps=200_000, max_episode_steps=1000, n_envs=8, seed=0):
    train_env = make_vec_envs(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True)
    train_env.save("normalization/ppo_vecnormalize_stats.pkl")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]}
    )
    log_dir = "logs/ppo/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps)

    episode_rewards = []
    obs = train_env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        episode_rewards.append(reward)
        if done.any():
            obs = train_env.reset()
    return model, train_env, episode_rewards

def train_sac(env_id="HalfCheetah-v5", total_timesteps=200_000, max_episode_steps=1000, n_envs=8, seed=0):
    train_env = make_vec_envs(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True)
    train_env.save("normalization/sac_vecnormalize_stats.pkl")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]}
    )
    log_dir = "logs/sac/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps)

    episode_rewards = []
    obs = train_env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        episode_rewards.append(reward)
        if done.any():
            obs = train_env.reset()
    return model, train_env, episode_rewards
