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
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            terminated = step_result[2]
            truncated = step_result[3] if len(step_result) > 3 else False

            done = np.any(terminated) or np.any(truncated) if isinstance(terminated, np.ndarray) else (terminated or truncated)
            total_reward += np.sum(reward) if isinstance(reward, np.ndarray) else reward

        episodio_rewards.append(total_reward)

    metriche = {
        "media_reward": np.mean(episodio_rewards),
        "dev_std_reward": np.std(episodio_rewards),
        "varianza_reward": np.var(episodio_rewards),
        "somma_reward": np.sum(episodio_rewards),
        "max_reward": np.max(episodio_rewards),
        "min_reward": np.min(episodio_rewards),
        "episodio_rewards": episodio_rewards,
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
    with open(filename, "w") as f:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):  # Per numeri
                f.write(f"{key}: {value:.2f}\n")
            elif isinstance(value, list):  # Per liste (es. episodio_rewards)
                f.write(f"{key}: {value}\n")

    print(f"Metriche salvate in: {filename}")





def plot_comparison(random_metrics, ppo_metrics, sac_metrics, output_path="results/plots/performance_comparison.png"):
    """
    Confronta le performance tra Random, PPO e SAC utilizzando le metriche fornite.
    """
    # Estrai le metriche principali
    labels = ["Random Policy", "PPO", "SAC"]
    mean_rewards = [
        random_metrics.get("media_reward", 0),
        ppo_metrics.get("media_reward", 0),
        sac_metrics.get("media_reward", 0),
    ]
    std_devs = [
        random_metrics.get("dev_std_reward", 0),
        ppo_metrics.get("dev_std_reward", 0),
        sac_metrics.get("dev_std_reward", 0),
    ]
    medians = [
        random_metrics.get("mediana_reward", 0),
        ppo_metrics.get("mediana_reward", 0),
        sac_metrics.get("mediana_reward", 0),
    ]
    max_rewards = [
        random_metrics.get("max_reward", 0),
        ppo_metrics.get("max_reward", 0),
        sac_metrics.get("max_reward", 0),
    ]
    min_rewards = [
        random_metrics.get("min_reward", 0),
        ppo_metrics.get("min_reward", 0),
        sac_metrics.get("min_reward", 0),
    ]

    # Creazione del plot a barre
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 8))
    bars = plt.bar(x, mean_rewards, yerr=std_devs, capsize=10, alpha=0.8, color=["gray", "blue", "orange"], edgecolor="black")

    # Aggiungi i valori sopra le barre
    for bar, mean in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{mean:.2f}",
                 ha='center', va='bottom', fontsize=10, color='black', weight='bold')

    # Etichette e titoli
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Reward Medio", fontsize=12)
    plt.title("Confronto delle Performance: Random vs PPO vs SAC", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(min(0, min(mean_rewards) - max(std_devs) * 1.2), max(mean_rewards) + max(std_devs) * 1.2)

    # Salvataggio e visualizzazione
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Plot per il confronto delle performance salvato in: {output_path}")

    # Creazione del box plot
    plt.figure(figsize=(12, 8))
    data = [random_metrics["episodio_rewards"], ppo_metrics["episodio_rewards"], sac_metrics["episodio_rewards"]]
    plt.boxplot(data, labels=labels)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Distribuzione dei Reward: Random vs PPO vs SAC", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/plots/reward_distribution.png")
    plt.show()
    print("Plot per il confronto dei reward salvato in: results/plots/reward_distribution.png")