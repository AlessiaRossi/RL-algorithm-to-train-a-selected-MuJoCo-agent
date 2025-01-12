import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

# TO DO : implementare la funzione evaluate_model e plot_comparison, vedere di gestiore info

def evaluate_model(model, env, n_eval_episodes=10, success_threshold=200):
    # Valutazione delle performance dell'agente

    episodio_rewards = []
    # Esecuzione di n_eval_episodes episodi
    for _ in range(n_eval_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        # = reset_result[1] if isinstance(reset_result, tuple) else {}
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            terminated = step_result[2]
            truncated = step_result[3] if len(step_result) > 3 else False
            #info = step_result[4] if len(step_result) > 4 else {}

            done = np.any(terminated) or np.any(truncated) if isinstance(terminated, np.ndarray) else (terminated or truncated)
            total_reward += np.sum(reward) if isinstance(reward, np.ndarray) else reward

        episodio_rewards.append(total_reward)

    metriche = {
        "media_reward": np.mean(episodio_rewards),
        "dev_std_reward": np.std(episodio_rewards),
        "varianza_reward": np.var(episodio_rewards),
        "somma_reward": np.sum(episodio_rewards),
    }
    return metriche

def evaluate_model_test(model, env, episodes=10, success_threshold=None):
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=episodes, deterministic=True
    )

    # Costruisci le metriche di valutazione
    metrics = {
        "media_reward": mean_reward,
        "dev_std_reward": std_reward,
    }

    # Verifica se il reward medio supera una soglia (se specificata)
    if success_threshold is not None:
        metrics["success"] = mean_reward >= success_threshold

    return metrics

def save_metrics(metrics, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")

'''def plot(ppo_metrics, sac_metrics, random_metrics, filename):
    # Confronto delle metriche tra PPO e SAC e visualizzazione tramite grafici

    labels = ["Random","PPO", "SAC"]
    rewards = [ppo_metrics.get("media_reward", 0), sac_metrics.get("media_reward", 0), random_metrics.get("media_reward", 0)]

    # Grafico Ricompensa media
    plt.figure(figsize=(8, 6))
    plt.bar(labels, rewards, color=['blue', 'orange','green'], alpha=0.7)
    plt.ylabel("Ricompensa Media")
    plt.title("Confronto Ricompense Medie")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()'''

