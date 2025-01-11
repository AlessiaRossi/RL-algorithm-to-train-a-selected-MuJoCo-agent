import os
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from environments import create_train_env, create_eval_env, make_env
from functions.utils import progressCallback

def sac_optuna_tuning(env_id, max_episode_steps=1000, n_trials=10, training_steps=50000, eval_episodes=5, n_envs=8, seed=0):
    """
    Tuning di SAC (learning_rate, buffer_size, batch_size, gamma, tau, ent_coef) su un SubprocVecEnv normalizzato.
    """
    def objective(trial: optuna.Trial):
        # Parametri da ottimizzare
        lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        buff_size = trial.suggest_int("buffer_size", 500000, 2000000, step=100000)
        bs = trial.suggest_int("batch_size", 128, 512, step=64)
        gam = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
        tau_ = trial.suggest_float("tau", 0.01, 0.1, log=True)

        auto_entropy = trial.suggest_categorical("auto_entropy", [True, False])
        if auto_entropy:
            ent_coef = "auto"
        else:
            ent_coef = trial.suggest_float("ent_coef_val", 1e-3, 1.0, log=True)

        # Creazione degli ambienti
        train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True)
        eval_env = make_env(env_id, max_episode_steps, seed)

        # Modello SAC
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=0,
            seed=seed,
            learning_rate=lr,
            buffer_size=buff_size,
            batch_size=bs,
            gamma=gam,
            tau=tau_,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": [256, 256]},
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
    print("\n[SAC Optuna] Best hyperparameters:", best_params)
    return best_params

def train_sac(env_id, total_timesteps=200_000, max_episode_steps=1000, eval_freq=50000, eval_episodes=5, n_envs=8, seed=0, hyperparams=None):
    """
    Allena un modello SAC con hyperparameter tuning opzionale.
    """
    # Creazione degli ambienti
    sac_stats = "results/normalization/sac_vecnormalize_stats.pkl"
    train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True)
    train_env.save(sac_stats)

    eval_env = create_train_env(env_id, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=False, norm_stats_path=sac_stats)

    default_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,  # provare range tra 1e-5 e 1e-3
        buffer_size=2_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        ent_coef="auto",  # Calcolo automatico del coefficiente di entropia
        policy_kwargs={"net_arch": [256, 256]}
    )

    # Gestione dei parametri di hyperparameter tuning
    if hyperparams is not None:
        if "auto_entropy" in hyperparams:
            if hyperparams["auto_entropy"]:
                hyperparams["ent_coef"] = "auto"
            else:
                hyperparams["ent_coef"] = hyperparams.get("ent_coef_val", 0.01)
            # Rimuoviamo i parametri non validi
            hyperparams.pop("auto_entropy", None)
            hyperparams.pop("ent_coef_val", None)
        default_kwargs.update(hyperparams)

    # Creazione del modello SAC
    model = SAC(**default_kwargs)

    # Configurazione del logger
    log_dir = "results/logs/sac/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Callback per il progresso e la valutazione
    progress_callback = progressCallback(total_timesteps=total_timesteps, check_freq=20000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1
    )

    # Avvio del training
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, eval_callback])

    # Chiusura degli ambienti
    train_env.close()
    eval_env.close()

    return model, train_env, eval_env
