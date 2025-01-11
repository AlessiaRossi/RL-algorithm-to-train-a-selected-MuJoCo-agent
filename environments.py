import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit, RecordEpisodeStatistics
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# Inizializzazione dell'environment
def make_env(env_id="HalfCheetah-v5", max_episode_steps=1000, seed=0):
    def _init():
        env = gym.make(env_id)
        # Limitiamo la durata massima dell'episodio
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # Registra statistiche (reward cumulativo, length, ecc.)
        env = RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return _init

# Vettorizzazione dell'environment
def make_vec_envs(env_id, n_envs, max_episode_steps, seed, normalize=False, norm_obs=False, norm_reward=False):
    env_fns = []
    for i in range(n_envs):
        env_fns.append(make_env(env_id, max_episode_steps, seed + 1000*i))
    vec_env = SubprocVecEnv(env_fns)
    if normalize:
        # Normalizza sia osservazioni che reward
        vec_env = VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=10.0)
    return vec_env

# Run una random policy per osservare il comportamento iniziale
def run_random_policy(env, seed=0):
    print("Running a random policy...")
    episodi= 5
    rewards_per_episode = []
    for episodio in range(episodi):
        obs, info = env.reset(seed=seed + episodio)
        episode_reward = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        rewards_per_episode.append(episode_reward)

        progress = 100.0 * (episodio + 1) / episodi
        print(f"[Random Policy] Episodio {episodio + 1}/{episodi} - Avanzamento: {progress:.1f}%")

    env.close()
    return rewards_per_episode

# Creazione di un ambiente di training normalizzato
def create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True, norm_stats_path=None):
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
        eval_env = VecNormalize.load(norm_stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    return envs

# Creazione di un ambiente di valutazione opzionalmente normalizzato
def create_eval_env(env_id, max_episode_steps, seed):
    eval_env = make_env(env_id, max_episode_steps, seed+999)()
    return eval_env
