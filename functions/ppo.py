import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from environments import create_train_env, create_eval_env, make_env
from functions.progressCallback import ProgressCallback

def ppo_optuna_tuning(env_id, max_episode_steps=1000, n_trials=10, training_steps=50000, eval_episodes=5, n_envs=8, seed=0):
    """
    Tuning di PPO (learning_rate, n_steps, gamma, gae_lambda, clip_range) su un SubprocVecEnv normalizzato.
    """
    def objective(trial: optuna.Trial):
        # Parametri da ottimizzare
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        steps = trial.suggest_int("n_steps", 2048, 4096, step=1024)
        gam = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
        lam = trial.suggest_float("gae_lambda", 0.85, 1.00)
        clipr = trial.suggest_float("clip_range", 0.15, 0.3)

        # Creazione degli ambienti
        train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True)
        eval_env = create_eval_env(env_id, max_episode_steps, seed)

        # Modello PPO
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=0,
            seed=seed,
            learning_rate=lr,
            n_steps=steps,
            gamma=gam,
            gae_lambda=lam,
            clip_range=clipr,
            policy_kwargs={"net_arch": [512, 256, 128]},
        )

        # Training del modello
        model.learn(total_timesteps=training_steps)

        # Valutazione
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)

        # Chiusura degli ambienti
        train_env.close()
        eval_env.close()

        return -mean_reward

    # Configurazione di Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=True)
    best_params = study.best_params
    print("\n[PPO Optuna] Best hyperparameters:", best_params)
    return best_params

def train_ppo(env_id, total_timesteps=200_000, max_episode_steps=1000, eval_freq=50000, eval_episodes=5, n_envs=8, seed=0, hyperparams=None):
    """
    Allena un modello PPO con hyperparameter tuning opzionale.
    """
    # Creazione degli ambienti
    ppo_stats = "results/normalization/ppo_vecnormalize_stats.pkl"
    train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True)
    train_env.save(ppo_stats)

    eval_env = create_train_env(env_id, 1, max_episode_steps, seed+999, normalize=True, norm_obs=True, norm_reward=False, norm_stats_path=ppo_stats)

    default_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4, # provare range tra 1e-5 e 1e-3
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [512, 256, 128]}
    )

    # Gestione dei parametri di hyperparameter tuning
    if hyperparams is not None:
        default_kwargs.update(hyperparams)

    # Creazione del modello PPO
    model = PPO(**default_kwargs)

    # Configurazione del logger
    log_dir = "results/logs/ppo/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    # Callback per il progresso e la valutazione
    progress_callback = ProgressCallback(total_timesteps=total_timesteps, check_freq=20000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=0
    )

    # Avvio del training
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, eval_callback])

    return model, train_env, eval_env
