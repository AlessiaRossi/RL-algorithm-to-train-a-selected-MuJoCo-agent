
import numpy as np
import matplotlib.pyplot as plt

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

def plot(ppo_metrics, sac_metrics):
    # Confronto delle metriche tra PPO e SAC e visualizzazione tramite grafici

    labels = ["PPO", "SAC"]
    rewards = [ppo_metrics["media_reward"], sac_metrics["media_reward"]]

    # Grafico Ricompensa media
    plt.figure(figsize=(8, 6))
    plt.bar(labels, rewards, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel("Ricompensa Media")
    plt.title("Confronto Ricompensa Media")
    plt.show()

