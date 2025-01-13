import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from environments import create_train_env, create_eval_env
from functions.progressCallback import ProgressCallback

# Function for hyperparameter tuning of PPO using Optuna
def ppo_optuna_tuning(env_id, max_episode_steps=1000, n_trials=10, training_steps=50000, eval_episodes=5, n_envs=8, seed=0):
    """
    Performs hyperparameter tuning for PPO using Optuna.
    Args:
        env_id (str): ID of the environment to train on.
        max_episode_steps (int): Maximum number of steps per episode.
        n_trials (int): Number of Optuna trials.
        training_steps (int): Number of steps for each training run.
        eval_episodes (int): Number of episodes for evaluation.
        n_envs (int): Number of parallel environments.
        seed (int): Seed for reproducibility.
    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    def objective(trial: optuna.Trial):
        """
        Objective function for Optuna. Defines the hyperparameters to tune and evaluates the model.
        Args:
            trial (optuna.Trial): An Optuna trial object to sample hyperparameters.
        Returns:
            float: Negative mean reward (for minimization purposes).
        """
        # Define hyperparameters to tune
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        steps = trial.suggest_int("n_steps", 2048, 4096, step=1024)
        gam = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
        lam = trial.suggest_float("gae_lambda", 0.85, 1.00)
        clipr = trial.suggest_float("clip_range", 0.15, 0.3)

        # Create training and evaluation environments
        train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True)
        eval_env = create_eval_env(env_id, max_episode_steps, seed)

        # Initialize PPO model with sampled hyperparameters
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
            policy_kwargs={"net_arch": [512, 256, 128]},  # Neural network architecture
        )

        # Train the model
        model.learn(total_timesteps=training_steps)

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)

        # Close the environments
        train_env.close()
        eval_env.close()

        # Return negative mean reward for minimization
        return -mean_reward

    # Configure Optuna study
    study = optuna.create_study(direction="minimize")  # Minimize negative mean reward
    study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=True)  # Run optimization

    # Retrieve and print the best hyperparameters
    best_params = study.best_params
    print("\n[PPO Optuna] Best hyperparameters:", best_params)
    return best_params

# Function to train PPO with optional hyperparameter tuning
def train_ppo(env_id, total_timesteps=200_000, max_episode_steps=1000, eval_freq=50000, eval_episodes=5, n_envs=8, seed=0, hyperparams=None):
    """
    Trains a PPO model with optional hyperparameter tuning.
    Args:
        env_id (str): ID of the environment to train on.
        total_timesteps (int): Total number of training timesteps.
        max_episode_steps (int): Maximum number of steps per episode.
        eval_freq (int): Frequency of evaluations during training.
        eval_episodes (int): Number of episodes for evaluation.
        n_envs (int): Number of parallel environments.
        seed (int): Seed for reproducibility.
        hyperparams (dict): Dictionary of hyperparameters to override defaults.
    Returns:
        tuple: Trained model, training environment, evaluation environment.
    """
    # Define path for saving normalization statistics
    ppo_stats = "results/normalization/ppo_vecnormalize_stats.pkl"

    # Create training environment and save normalization stats
    train_env = create_train_env(env_id, n_envs, max_episode_steps, seed, normalize=True, norm_obs=True, norm_reward=True)
    train_env.save(ppo_stats)

    # Create evaluation environment and load saved normalization stats
    eval_env = create_train_env(env_id, 1, max_episode_steps, seed + 999, normalize=True, norm_obs=True, norm_reward=False, norm_stats_path=ppo_stats)

    # Default hyperparameters for PPO
    default_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,  # Learning rate range: 1e-5 to 1e-3
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [512, 256, 128]}  # Neural network architecture
    )

    # Override defaults with provided hyperparameters
    if hyperparams is not None:
        default_kwargs.update(hyperparams)

    # Initialize PPO model
    model = PPO(**default_kwargs)

    # Configure logging for training
    log_dir = "results/logs/ppo/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["csv", "tensorboard"])  # Log to CSV and TensorBoard
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

    # Start training with the specified callbacks
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, eval_callback])

    # Return the trained model and environments
    return model, train_env, eval_env
