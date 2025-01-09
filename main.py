import numpy as np
from environments import make_env, run_random_policy
from analysis_results import evaluate_model, plot, save_metrics
from functions.ppo import ppo_optuna_tuning, train_ppo
from functions.sac import sac_optuna_tuning, train_sac
from functions.utils import ensure_dir, load_config

def main():
    # Caricamento della configurazione
    config = load_config()

    # Parametri principali dal file di configurazione
    ENV_ID = config["env_id"]
    MAX_EPISODE_STEPS = config["max_episode_steps"]
    SEED = config["seed"]
    EVAL_FREQ = config["eval_freq"]
    PPO_TIMESTEPS = config["ppo_timesteps"]
    SAC_TIMESTEPS = config["sac_timesteps"]
    PPO_TRAINING_STEPS = config["ppo_training_steps"]
    SAC_TRAINING_STEPS = config["sac_training_steps"]
    N_ENVS = config["n_envs"]
    N_EPISODES = config["n_episodes"]
    N_TRIALS_PPO = config["n_trials_ppo"]
    N_TRIALS_SAC = config["n_trials_sac"]

    # Creazione delle directory per risultati
    ensure_dir("results/normalization")
    ensure_dir("results/metrics")
    ensure_dir("results/videos")
    ensure_dir("results/plots")
    ensure_dir("results/logs")
    
    # Esecuzione di una random policy
    print("\nEseguendo la Random Policy...")
    env = make_env()()
    random_rewards = run_random_policy(env)
    random_metrics = {
        "media_reward": np.mean(random_rewards),
        "dev_std_reward": np.std(random_rewards),
        "varianza_reward": np.var(random_rewards),
        "somma_reward": np.sum(random_rewards),
    }
    save_metrics(random_metrics, "results/metrics/random_metrics.txt")

    print(f"Random Policy Rewards per episode: {random_rewards}")
    print(f"Random Policy Media dei reward: {random_metrics['media_reward']}")

    # Tuning PPO con Optuna
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

    # Tuning SAC con Optuna
    print("[Optuna] Tuning SAC hyperparameters...")
    best_sac_params = sac_optuna_tuning(
        env_id=ENV_ID,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_trials=N_TRIALS_SAC,
        training_steps=SAC_TRAINING_STEPS,
        eval_episodes=N_EPISODES,
        n_envs=N_ENVS,
        seed=SEED,
    )

    # Training dell'agente con algoritmo PPO
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
    ppo_metrics = evaluate_model(ppo_model, ppo_train_env, N_EPISODES)
    save_metrics(ppo_metrics, "results/metrics/ppo_metrics.txt")
    print(f"PPO Media dei reward: {ppo_metrics['media_reward']}")

    # Training dell'agente con algoritmo SAC
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
    sac_metrics = evaluate_model(sac_model, sac_train_env, N_EPISODES)
    save_metrics(sac_metrics, "results/metrics/sac_metrics.txt")
    print(f"SAC Media dei reward: {sac_metrics['media_reward']}")

    # print("\nGenerazione del grafico di confronto...")
    # plot(random_metrics, ppo_metrics, sac_metrics, "results/plots/rewards_comparison.png")
    # print("Grafico salvato in results/plots/rewards_comparison.png")


if __name__ == "__main__":
    main()
