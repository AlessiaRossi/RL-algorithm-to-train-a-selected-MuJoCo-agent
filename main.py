import os
import time
import numpy as np
import warnings

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from functions.record import record_agent_video

# Suppress specific warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.record_video")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from environments import make_env, run_random_policy
from analysis_results import evaluate_model, save_metrics
from functions.ppo import ppo_optuna_tuning, train_ppo
from functions.sac import sac_optuna_tuning, train_sac
from functions.utils import ensure_dir, load_config
from analysis_results import plot_comparison

def main():
    # Load configuration file
    print("Loading configuration file...")
    config = load_config()

    # Debug mode flag
    DEBUG = config["general"]["debug"]

    # Environment parameters
    ENV_ID = config["environment"]["env_id"]
    MAX_EPISODE_STEPS = config["environment"]["max_episode_steps"]
    SEED = config["environment"]["seed"]
    EVAL_FREQ = config["evaluation"]["eval_freq"]
    N_EPISODES = config["evaluation"]["n_episodes"]
    N_ENVS = config["environment"]["n_envs"]

    # PPO specific parameters
    PPO_TIMESTEPS = config["ppo"]["timesteps"]
    PPO_TRAINING_STEPS = config["ppo"]["training_steps"]
    N_TRIALS_PPO = config["ppo"]["n_trials"]

    # SAC specific parameters
    SAC_TIMESTEPS = config["sac"]["timesteps"]
    SAC_TRAINING_STEPS = config["sac"]["training_steps"]
    N_TRIALS_SAC = config["sac"]["n_trials"]

    # Create necessary directories for results
    print("Creating result directories...")
    ensure_dir("results/normalization")
    ensure_dir("results/metrics")
    ensure_dir("results/videos")
    ensure_dir("results/plots")
    ensure_dir("results/logs")
    
    print("\n### Inizialization ###")
    
    # Run Random Policy to get baseline metrics
    print("\nRunning a random policy...")
    env = make_env()()
    random_rewards = run_random_policy(env)
    random_metrics = {
        "media_reward": np.mean(random_rewards),
        "dev_std_reward": np.std(random_rewards),
        "varianza_reward": np.var(random_rewards),
        "somma_reward": np.sum(random_rewards),
        "max_reward": np.max(random_rewards),
        "min_reward": np.min(random_rewards),
        "episodio_rewards": random_rewards
    }
    save_metrics(random_metrics, "results/metrics/random_metrics.txt")

    print(f"\nRandom Policy rewards: {random_rewards}")
    print(f"\nRandom Policy Average Reward: {random_metrics['media_reward']}")

    if DEBUG:
        # Debug mode: load pre-trained models
        debug_paths = config["general"]["debug_paths"]
        print("\nDebug mode activated: Loading saved models...")
        ppo_model = PPO.load(debug_paths["ppo_model"])
        sac_model = SAC.load(debug_paths["sac_model"])

        # Load evaluation environments with normalization statistics
        eval_env_ppo = DummyVecEnv([lambda: make_env(env_id=ENV_ID, max_episode_steps=MAX_EPISODE_STEPS, seed=SEED)()])
        eval_env_sac = DummyVecEnv([lambda: make_env(env_id=ENV_ID, max_episode_steps=MAX_EPISODE_STEPS, seed=SEED)()])

        # Load PPO normalization statistics
        if os.path.exists(debug_paths["ppo_stats"]):
            eval_env_ppo = VecNormalize.load(debug_paths["ppo_stats"], eval_env_ppo)
            eval_env_ppo.training = False
            eval_env_ppo.norm_reward = False
            print(f"PPO statistics loaded from: {debug_paths['ppo_stats']}")

        # Load SAC normalization statistics
        if os.path.exists(debug_paths["sac_stats"]):
            eval_env_sac = VecNormalize.load(debug_paths["sac_stats"], eval_env_sac)
            eval_env_sac.training = False
            eval_env_sac.norm_reward = False
            print(f"SAC statistics loaded from: {debug_paths['sac_stats']}")

        # Evaluate PPO model
        print("\nValutazione del modello PPO...")
        ppo_metrics = evaluate_model(ppo_model, eval_env_ppo, N_EPISODES)
        save_metrics(ppo_metrics, "results/metrics/ppo_metrics.txt")
        print(f"SAC statistics loaded from: {ppo_metrics['media_reward']}")

        # Evaluate SAC model
        print("\nValutazione del modello SAC...")
        sac_metrics = evaluate_model(sac_model, eval_env_sac, N_EPISODES)
        save_metrics(sac_metrics, "results/metrics/sac_metrics.txt")
        print(f"SAC Media dei reward: {sac_metrics['media_reward']}")
    else:
        # Tuning PPO using Optuna for hyperparameter tuning
        print("\n[Optuna] Tuning PPO hyperparameters...")
        best_ppo_params = ppo_optuna_tuning(
            env_id=ENV_ID,
            max_episode_steps=MAX_EPISODE_STEPS,
            n_trials=N_TRIALS_PPO,
            training_steps=PPO_TRAINING_STEPS,
            eval_episodes=N_EPISODES,
            n_envs=N_ENVS,
            seed=SEED,
        )

        # Tuning SAC using Optuna for hyperparameter tuning
        print("\n[Optuna] Tuning SAC hyperparameters...")
        best_sac_params = sac_optuna_tuning(
            env_id=ENV_ID,
            max_episode_steps=MAX_EPISODE_STEPS,
            n_trials=N_TRIALS_SAC,
            training_steps=SAC_TRAINING_STEPS,
            eval_episodes=N_EPISODES,
            n_envs=N_ENVS,
            seed=SEED,
        )

        # Train PPO agent with the best hyperparameters and evaluate
        print("\nTraining PPO...")
        ppo_model, ppo_train_env, ppo_eval_env = train_ppo(
            env_id=ENV_ID,
            total_timesteps=PPO_TIMESTEPS,
            max_episode_steps=MAX_EPISODE_STEPS,
            eval_freq=EVAL_FREQ,
            eval_episodes=N_EPISODES,
            n_envs=N_ENVS,
            seed=SEED,
            hyperparams=best_ppo_params,
        )
        ppo_metrics = evaluate_model(ppo_model, ppo_eval_env, N_EPISODES)
        save_metrics(ppo_metrics, "results/metrics/ppo_metrics.txt")
        print(f"\nPPO Average Reward: {ppo_metrics['media_reward']}")

        # Train SAC agent with the best hyperparameters and evaluate
        print("\nTraining SAC...")
        sac_model, sac_train_env, sac_eval_env = train_sac(
            env_id=ENV_ID,
            total_timesteps=SAC_TIMESTEPS,
            max_episode_steps=MAX_EPISODE_STEPS,
            eval_freq=EVAL_FREQ,
            eval_episodes=N_EPISODES,
            n_envs=N_ENVS,
            seed=SEED,
            hyperparams=best_sac_params,
        )
        sac_metrics = evaluate_model(sac_model, sac_eval_env, N_EPISODES)
        save_metrics(sac_metrics, "results/metrics/sac_metrics.txt")
        print(f"\nSAC Average Reward: {sac_metrics['media_reward']}")

    
    # Record videos of the episodes
    print("\nRecording videos of episodes...")
    video_time = time.strftime("%H-%M_%d-%m-%Y")
    video_ppo = f"results/videos/ppo/{video_time}"
    video_sac = f"results/videos/sac/{video_time}"

    print(f"\nRecording PPO videos: {video_ppo}")
    OFFSET_SEED = 9999  # Offset seed for video recording to avoid overlapping with training
    record_agent_video(
        model=ppo_model,
        env_id=ENV_ID,
        max_steps=MAX_EPISODE_STEPS,
        seed=SEED + OFFSET_SEED, 
        video_dir=video_ppo,
        video_prefix="ppo_agent",
        episodes=N_EPISODES
    )

    print(f"\nRecording SAC videos: {video_sac}")
    record_agent_video(
        model=sac_model,
        env_id=ENV_ID,
        max_steps=MAX_EPISODE_STEPS,
        seed=SEED + OFFSET_SEED,
        video_dir=video_sac,
        video_prefix="sac_agent",
        episodes=N_EPISODES
    )


    # Plot comparison of performance
    plot_comparison(
        random_metrics,
        ppo_metrics,
        sac_metrics,
        output_dir="results/plots"
    )
    if not DEBUG:
        # Close environments after training
        ppo_train_env.close()
        ppo_eval_env.close()
        sac_train_env.close()
        sac_eval_env.close()

if __name__ == "__main__":
    main()
