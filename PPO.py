import gym
from gym.wrappers import TimeLimit, RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from environments import make_env

def train_ppo(env_id="HalfCheetah-v5",
              total_timesteps=200_000,
              max_episode_steps=1000,
              eval_freq=50000,
              eval_episodes=5,
              seed=0,
              hyperparams=None):
    """
    Allena un modello PPO su un singolo ambiente
    Ritorna (model, train_env, eval_env).
    """
    # Ambiente di training
    train_env = make_env(env_id, max_episode_steps=max_episode_steps, seed=seed)

    # Ambiente di valutazione
    eval_env = make_env(env_id, max_episode_steps=max_episode_steps, seed=seed + 999)

    # Parametri di default
    default_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    if hyperparams is not None:
        default_kwargs.update(hyperparams)

    model = PPO(**default_kwargs)

    # Callback di valutazione
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1
    )

    # Avvio dell'allenamento
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback])

    return model, train_env, eval_env
