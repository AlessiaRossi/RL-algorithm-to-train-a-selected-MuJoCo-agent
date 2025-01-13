import os
import time
import numpy as np
import warnings

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from functions.record import record_agent_video

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.record_video")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from environments import make_env, run_random_policy
from analysis_results import evaluate_model, save_metrics
from functions.ppo import ppo_optuna_tuning, train_ppo
from functions.sac import sac_optuna_tuning, train_sac
from functions.utils import ensure_dir, load_config

def main():
    print("\nCaricamento del file di configurazione...")
    # Caricamento della configurazione
    config = load_config()

    # Debug mode
    DEBUG = config["general"]["debug"]

    # Parametri principali dal file di configurazione
    ENV_ID = config["environment"]["env_id"]
    MAX_EPISODE_STEPS = config["environment"]["max_episode_steps"]
    SEED = config["environment"]["seed"]
    EVAL_FREQ = config["evaluation"]["eval_freq"]
    N_EPISODES = config["evaluation"]["n_episodes"]
    N_ENVS = config["environment"]["n_envs"]

    # Parametri specifici di PPO
    PPO_TIMESTEPS = config["ppo"]["timesteps"]
    PPO_TRAINING_STEPS = config["ppo"]["training_steps"]
    N_TRIALS_PPO = config["ppo"]["n_trials"]

    # Parametri specifici di SAC
    SAC_TIMESTEPS = config["sac"]["timesteps"]
    SAC_TRAINING_STEPS = config["sac"]["training_steps"]
    N_TRIALS_SAC = config["sac"]["n_trials"]

    # Creazione delle directory per risultati
    print("Creazione delle directory per i risultati...")
    ensure_dir("results/normalization")
    ensure_dir("results/metrics")
    ensure_dir("results/videos")
    ensure_dir("results/plots")
    ensure_dir("results/logs")
    
    print("\n### Inizializzazione dell'ambiente ###")
    
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

    if DEBUG:
        debug_paths = config["general"]["debug_paths"]

        print("\nModalità debug attivata: caricamento dei modelli salvati...")
        ppo_model = PPO.load(debug_paths["ppo_model"])
        sac_model = SAC.load(debug_paths["sac_model"])

        # Creazione dell'ambiente di valutazione con statistiche caricate
        eval_env_ppo = DummyVecEnv([lambda: make_env(env_id=ENV_ID, max_episode_steps=MAX_EPISODE_STEPS, seed=SEED)()])
        eval_env_sac = DummyVecEnv([lambda: make_env(env_id=ENV_ID, max_episode_steps=MAX_EPISODE_STEPS, seed=SEED)()])

        if os.path.exists(debug_paths["ppo_stats"]):
            eval_env_ppo = VecNormalize.load(debug_paths["ppo_stats"], eval_env_ppo)
            eval_env_ppo.training = False
            eval_env_ppo.norm_reward = False
            print(f"Statistiche PPO caricate da: {debug_paths['ppo_stats']}")

        if os.path.exists(debug_paths["sac_stats"]):
            eval_env_sac = VecNormalize.load(debug_paths["sac_stats"], eval_env_sac)
            eval_env_sac.training = False
            eval_env_sac.norm_reward = False
            print(f"Statistiche SAC caricate da: {debug_paths['sac_stats']}")

        # Valutazione dei modelli
        print("\nValutazione del modello PPO...")
        ppo_metrics = evaluate_model(ppo_model, eval_env_ppo, N_EPISODES)
        print(f"PPO Media dei reward: {ppo_metrics['media_reward']}")

        print("\nValutazione del modello SAC...")
        sac_metrics = evaluate_model(sac_model, eval_env_sac, N_EPISODES)
        print(f"SAC Media dei reward: {sac_metrics['media_reward']}")
    else:
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
        ppo_metrics = evaluate_model(ppo_model, ppo_eval_env, N_EPISODES)
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
        sac_metrics = evaluate_model(sac_model, sac_eval_env, N_EPISODES)
        save_metrics(sac_metrics, "results/metrics/sac_metrics.txt")
        print(f"SAC Media dei reward: {sac_metrics['media_reward']}")

    # print("\nGenerazione del grafico di confronto...")
    # plot(random_metrics, ppo_metrics, sac_metrics, "results/plots/rewards_comparison.png")
    # print("Grafico salvato in results/plots/rewards_comparison.png")
    
    # Registrazione dei video degli episodi
    print("\nRegistrazione dei video degli episodi...")
    video_time = time.strftime("%H-%M_%d-%m-%Y")
    video_ppo = f"results/videos/ppo/{video_time}"
    video_sac = f"results/videos/sac/{video_time}"
    
    print(f"Registrazione video PPO: {video_ppo}")
    OFFSET_SEED = 9999 # Seed diverso per i video, cosi da registrati situazioni diverse da quelle viste durante il training
    record_agent_video(
        model=ppo_model,
        env_id=ENV_ID,
        max_steps=MAX_EPISODE_STEPS,
        seed=SEED + OFFSET_SEED, 
        video_dir=video_ppo,
        video_prefix="ppo_agent",
        episodes=N_EPISODES
    )

    print(f"\nRegistrazione video SAC: {video_sac}")
    record_agent_video(
        model=sac_model,
        env_id=ENV_ID,
        max_steps=MAX_EPISODE_STEPS,
        seed=SEED + OFFSET_SEED,
        video_dir=video_sac,
        video_prefix="sac_agent",
        episodes=N_EPISODES
    )

    if not DEBUG:
        # Chiudiamo env
        ppo_train_env.close()
        ppo_eval_env.close()
        sac_train_env.close()
        sac_eval_env.close()

if __name__ == "__main__":
    main()
