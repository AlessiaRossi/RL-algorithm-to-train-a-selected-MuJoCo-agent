import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit, RecordEpisodeStatistics
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Function to initialize a single environment
def make_env(env_id="HalfCheetah-v5", max_episode_steps=1000, seed=0):
    """
    Creates a gym environment with optional wrappers for monitoring and limiting episode length.
    Args:
        env_id (str): The ID of the environment to create (default: HalfCheetah-v5).
        max_episode_steps (int): Maximum steps allowed per episode.
        seed (int): Seed for reproducibility.
    Returns:
        function: A function that initializes and returns the environment when called.
    """
    def _init():
        env = gym.make(env_id)  # Create the base environment
        env = Monitor(env)  # Add a monitor to track statistics (e.g., rewards, episode length)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)  # Limit the episode length
        env.reset(seed=seed)  # Reset the environment with the given seed
        return env
    return _init

# Function to create vectorized environments
def make_vec_envs(env_id, n_envs, max_episode_steps, seed, normalize=False, norm_obs=False, norm_reward=False):
    """
    Creates a vectorized environment using multiple parallel instances.
    Args:
        env_id (str): The ID of the environment to create.
        n_envs (int): Number of parallel environments to create.
        max_episode_steps (int): Maximum steps allowed per episode.
        seed (int): Seed for reproducibility.
        normalize (bool): Whether to normalize observations and rewards.
        norm_obs (bool): Whether to normalize observations.
        norm_reward (bool): Whether to normalize rewards.
    Returns:
        VecEnv: A vectorized environment.
    """
    env_fns = []
    for i in range(n_envs):
        # Create multiple instances of the environment with different seeds
        env_fns.append(make_env(env_id, max_episode_steps, seed + 1000 * i))
    vec_env = SubprocVecEnv(env_fns)  # Use subprocesses for parallelization

    if normalize:
        # Apply VecNormalize to normalize observations and rewards
        vec_env = VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=10.0)
    return vec_env

# Function to run a random policy and observe its performance
def run_random_policy(env, seed=0):
    """
    Runs a random policy in the given environment to observe its behavior.
    Args:
        env: The environment to run the random policy in.
        seed (int): Seed for reproducibility.
    Returns:
        list: A list of total rewards obtained in each episode.
    """
    print("Running a random policy...")
    episodi = 5  # Number of episodes to run
    rewards_per_episode = []

    for episodio in range(episodi):
        obs, info = env.reset(seed=seed + episodio)  # Reset the environment
        episode_reward = 0

        while True:
            action = env.action_space.sample()  # Sample a random action
            obs, reward, terminated, truncated, info = env.step(action)  # Step in the environment
            episode_reward += reward
            if terminated or truncated:  # Check if the episode has ended
                break

        rewards_per_episode.append(episode_reward)  # Store the total reward

        # Print progress
        progress = 100.0 * (episodio + 1) / episodi
        print(f"[Random Policy] Episode {episodio + 1}/{episodi} - Progress: {progress:.1f}%")

    env.close()  # Close the environment
    return rewards_per_episode

# Function to create a normalized training environment
def create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True, norm_stats_path=None):
    """
    Creates a training environment with optional normalization.
    Args:
        env_id (str): The ID of the environment to create.
        n_envs (int): Number of parallel environments.
        max_episode_steps (int): Maximum steps allowed per episode.
        seed (int): Seed for reproducibility.
        normalize (bool): Whether to normalize observations and rewards.
        norm_obs (bool): Whether to normalize observations.
        norm_reward (bool): Whether to normalize rewards.
        norm_stats_path (str): Path to load normalization statistics from.
    Returns:
        VecEnv: A normalized vectorized environment for training.
    """
    envs = make_vec_envs(
        env_id,
        n_envs,
        max_episode_steps,
        seed,
        normalize=normalize,
        norm_obs=norm_obs,
        norm_reward=norm_reward
    )

    if norm_stats_path:
        # Load normalization statistics if provided
        envs = VecNormalize.load(norm_stats_path, envs)
        envs.training = False  # Disable training for VecNormalize
        envs.norm_reward = False  # Do not normalize rewards during evaluation

    return envs

# Function to create an evaluation environment
def create_eval_env(env_id, max_episode_steps, seed):
    """
    Creates a single evaluation environment.
    Args:
        env_id (str): The ID of the environment to create.
        max_episode_steps (int): Maximum steps allowed per episode.
        seed (int): Seed for reproducibility.
    Returns:
        gym.Env: A single environment for evaluation.
    """
    env = make_env(env_id, max_episode_steps, seed + 999)()  # Use a different seed for evaluation
    return env
