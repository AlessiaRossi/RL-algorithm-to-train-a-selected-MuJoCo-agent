import os
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from environments import create_train_env, create_eval_env
from functions.progressCallback import ProgressCallback

# Function for SAC hyperparameter tuning using Optuna
def sac_optuna_tuning(env_id, max_episode_steps=1000, n_trials=10, training_steps=50000, eval_episodes=5, n_envs=8, seed=0):
    """
    Perform hyperparameter tuning for SAC using Optuna.
    Args:
        env_id (str): Environment ID to train on.
        max_episode_steps (int): Maximum steps allowed per episode.
        n_trials (int): Number of trials for hyperparameter tuning.
        training_steps (int): Number of timesteps for each training trial.
        eval_episodes (int): Number of evaluation episodes per trial.
        n_envs (int): Number of parallel environments.
        seed (int): Seed for reproducibility.
    Returns:
        dict: Best hyperparameters found during the tuning process.
    """
    def objective(trial: optuna.Trial):
        """
        Define the objective function for Optuna optimization.
        Args:
            trial (optuna.Trial): Optuna trial object for sampling hyperparameters.
        Returns:
            float: Negative mean reward (for minimization purposes).
        """
        # Sample hyperparameters for SAC
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        buff_size = trial.suggest_int("buffer_size", 500_000, 3_000_000, step=500_000)
        bs = trial.suggest_int("batch_size", 128, 1024, step=64)
        gam = trial.suggest_float("gamma", 0.90, 0.9999, log=True)
        tau_ = trial.suggest_float("tau", 0.005, 0.1, log=True)

        # Entropy coefficient handling
        auto_entropy = trial.suggest_categorical("auto_entropy", [True, False])
        if auto_entropy:
            ent_coef = "auto"  # Automatic entropy calculation
        else:
            ent_coef = trial.suggest_float("ent_coef_val", 1e-3, 1.0, log=True)  # Fixed entropy coefficient

        # Create training and evaluation environments
        train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True)
        eval_env = create_eval_env(env_id, max_episode_steps, seed)

        # Initialize the SAC model with sampled hyperparameters
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
            policy_kwargs={"net_arch": [256, 256]},  # Network architecture
        )

        # Train the model
        model.learn(total_timesteps=training_steps)

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)

        # Close environments after use
        train_env.close()
        eval_env.close()

        # Return negative mean reward for minimization
        return -mean_reward

    # Configure Optuna study to minimize negative rewards
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=True)

    # Print and return the best parameters
    best_params = study.best_params
    print("\n[SAC Optuna] Best hyperparameters:", best_params)
    return best_params

# Function to train SAC with optional hyperparameter tuning
def train_sac(env_id, total_timesteps=200_000, max_episode_steps=1000, eval_freq=50000, eval_episodes=5, n_envs=8, seed=0, hyperparams=None):
    """
    Train an SAC model with optional hyperparameter tuning.
    Args:
        env_id (str): Environment ID to train on.
        total_timesteps (int): Total number of timesteps for training.
        max_episode_steps (int): Maximum steps allowed per episode.
        eval_freq (int): Frequency of evaluations during training.
        eval_episodes (int): Number of evaluation episodes during training.
        n_envs (int): Number of parallel environments.
        seed (int): Seed for reproducibility.
        hyperparams (dict): Optional dictionary of hyperparameters to override defaults.
    Returns:
        tuple: Trained model, training environment, and evaluation environment.
    """
    # Path to save normalization statistics
    sac_stats = "results/normalization/sac_vecnormalize_stats.pkl"

    # Create training environment and save normalization stats
    train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True)
    train_env.save(sac_stats)

    # Create evaluation environment with loaded stats
    eval_env = create_train_env(env_id, 1, max_episode_steps, seed + 999, normalize=True, norm_obs=True, norm_reward=False, norm_stats_path=sac_stats)

    # Default hyperparameters for SAC
    default_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,  # Default learning rate
        buffer_size=2_000_000,  # Replay buffer size
        batch_size=256,  # Batch size
        gamma=0.99,  # Discount factor
        tau=0.02,  # Soft update coefficient
        ent_coef="auto",  # Automatic entropy calculation
        policy_kwargs={"net_arch": [256, 256]}  # Network architecture
    )

    # Update default hyperparameters with tuned values, if provided
    if hyperparams is not None:
        if "auto_entropy" in hyperparams:
            if hyperparams["auto_entropy"]:
                hyperparams["ent_coef"] = "auto"  # Enable automatic entropy
            else:
                hyperparams["ent_coef"] = hyperparams.get("ent_coef_val", 0.01)  # Use fixed entropy coefficient
            # Remove irrelevant keys
            hyperparams.pop("auto_entropy", None)
            hyperparams.pop("ent_coef_val", None)
        default_kwargs.update(hyperparams)

    # Initialize the SAC model
    model = SAC(**default_kwargs)

    # Configure logging
    log_dir = "results/logs/sac/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["csv", "tensorboard"])  # Enable CSV and TensorBoard logging
    model.set_logger(new_logger)

    # Define callbacks for training progress and evaluation
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

    # Train the SAC model with the specified callbacks
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, eval_callback])

    # Return the trained model and environments
    return model, train_env, eval_env
