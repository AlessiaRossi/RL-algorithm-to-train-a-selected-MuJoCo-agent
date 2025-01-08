import os
import numpy as np
from training import train_ppo, train_sac
from environments import make_env, run_random_policy
from analysis_results import evaluate_model, plot, save_metrics

'''TO DO:
implemenatione main con config , cartelle di salvataggio metriche video e grafici
'''

def main():
    # Esecuzione di una random policy
    env = make_env()()
    random_rewards = run_random_policy(env)
    random_metrics = {
        "media_reward": sum(random_rewards) / len(random_rewards),
        "dev_std_reward": np.std(random_rewards),
        "varianza_reward": np.var(random_rewards),
        "somma_reward": sum(random_rewards),
    }
    save_metrics(random_metrics, "results/metrics/random_metrics.txt")

    print(f"Random Policy Rewards per episode: {random_rewards}")
    print(f"Random Policy Media dei reward: {random_metrics['media_reward']}")

    # Training dell'agente con algoritmo PPO
    ppo_model, ppo_env, ppo_rewards = train_ppo()
    ppo_metrics = evaluate_model(ppo_model, ppo_env)
    save_metrics(ppo_metrics, "results/metrics/ppo_metrics.txt")
    print(f"PPO Media dei reward: {ppo_metrics['media_reward']}")



    # Training dell'agente con algoritmo SAC
    sac_model, sac_env, sac_rewards = train_sac()
    sac_metrics = evaluate_model(sac_model, sac_env)
    save_metrics(sac_metrics, 'results/metrics/sac_metrics.txt')
    print(f"SAC Media dei reward: {sac_metrics['media_reward']}")

    #plot(random_metrics, ppo_metrics, sac_metrics,"results/rewards_comparison.png")
    #




if __name__ == "__main__":
    main()