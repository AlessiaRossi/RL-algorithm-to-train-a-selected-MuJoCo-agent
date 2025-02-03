import numpy as np
import os
import matplotlib.pyplot as plt
import json
from stable_baselines3.common.evaluation import evaluate_policy

# Evaluate the model's performance on a given environment for a number of episodes
def evaluate_model(model, env, n_eval_episodes=10, success_threshold=200):
    """
        Evaluate the performance of a reinforcement learning agent.
        Args:
            model: Trained RL model to evaluate.
            env: Evaluation environment.
            n_eval_episodes: Number of episodes to evaluate the model.
            success_threshold: Optional threshold for determining success.
        Returns:
            A dictionary containing evaluation metrics such as average reward, standard deviation, etc.
        """
    try:
        episodio_rewards = [] # Store the rewards of each episode

        # Run the evaluation loop for n_eval_episodes
        for _ in range(n_eval_episodes):
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            done = False
            total_reward = 0

            while not done:
                # Use the model to predict the next action
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action) # Take a step in the environment
                obs = step_result[0]
                reward = step_result[1]
                terminated = step_result[2]
                truncated = step_result[3] if len(step_result) > 3 else False

                # Determine if the episode is done (either terminated or truncated)
                done = np.any(terminated) or np.any(truncated) if isinstance(terminated, np.ndarray) else (terminated or truncated)
                total_reward += np.sum(reward) if isinstance(reward, np.ndarray) else reward

            episodio_rewards.append(total_reward) # Store the total reward of the episode

        # Calculate evaluation metrics
        metriche = {
            "media_reward": np.mean(episodio_rewards),     # Average reward
            "dev_std_reward": np.std(episodio_rewards),    # Standard deviation of rewards
            "varianza_reward": np.var(episodio_rewards),   # Variance of rewards
            "somma_reward": np.sum(episodio_rewards),      # Total reward
            "max_reward": np.max(episodio_rewards),        # Maximum reward
            "min_reward": np.min(episodio_rewards),        # Minimum reward
            "episodio_rewards": episodio_rewards,          # Rewards of each episode
        }
        return metriche
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"error": str(e)}

# Save evaluation metrics to a text file
def save_metrics(metrics, filename):
    """
       Save evaluation metrics to a file.
       Args:
           metrics: Dictionary containing evaluation metrics.
           filename: File path where metrics will be saved.
       """
    os.makedirs(os.path.dirname(filename), exist_ok=True) # Create the directory if it doesn't exist
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to: {filename}")

# Compare the performance of Random, PPO, and SAC models using plots
def plot_comparison(random_metrics, ppo_metrics, sac_metrics, output_dir):
    """
    Plot a comparison of performance metrics between Random, PPO, and SAC policies.
    Args:
        random_metrics: Metrics for the Random policy.
        ppo_metrics: Metrics for the PPO policy.
        sac_metrics: Metrics for the SAC policy.
        output_path: File path to save the generated plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the relevant metrics for plotting
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

    # Create the bar plot for mean rewards
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 8))
    bars = plt.bar(x, mean_rewards, yerr=std_devs, capsize=10, alpha=0.8, color=["gray", "blue", "orange"], edgecolor="black")
    for bar, mean in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{mean:.2f}",
                 ha='center', va='bottom', fontsize=10, color='black', weight='bold')
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Performance Comparison: Random vs PPO vs SAC", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(min(0, min(mean_rewards) - max(std_devs) * 1.2), max(mean_rewards) + max(std_devs) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_reward_comparison.png"))
    plt.show()
    print(f"Performance comparison plot saved to: {os.path.join(output_dir,'average_reward_comparison.png')}")

    # Create a box plot for reward distributions
    plt.figure(figsize=(12, 8))
    data = [random_metrics["episodio_rewards"], ppo_metrics["episodio_rewards"], sac_metrics["episodio_rewards"]]
    plt.boxplot(data, labels=labels)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Reward Distribution: Random vs PPO vs SAC", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_distribution.png"))
    plt.show()
    print(f"Reward distribution plot saved to: {os.path.join(output_dir, 'reward_distribution.png')}")

# Plot the rewards for each episode
def plot_episode_rewards(metrics, output_dir, model_name, title="Episode Rewards Over Time"):
    """
    Plot the rewards for each episode to show improvements over time.
    Args:
        metrics: Dictionary containing evaluation metrics.
        output_dir: Directory to save the plot.
        model_name: Name of the model (e.g., 'PPO' or 'SAC').
        title: Title of the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    episodio_rewards = metrics["episodio_rewards"]

    plt.figure(figsize=(12, 8))
    plt.plot(range(len(episodio_rewards)), episodio_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title(f"{title} - {model_name}", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot with the model name included
    plot_filename = os.path.join(output_dir, f"episode_rewards_{model_name.lower()}.png")
    plt.savefig(plot_filename)
    plt.show()

    print(f"Episode rewards plot saved to: {plot_filename}")


