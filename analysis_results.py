import numpy as np
import os
import matplotlib.pyplot as plt
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

#Test a model using a simpler evaluation method
def evaluate_model_test(model, env, episodes=10, success_threshold=None):
    """
        Evaluate the model using Stable-Baselines3's built-in evaluation method.
        Args:
            model: Trained RL model.
            env: Evaluation environment.
            episodes: Number of evaluation episodes.
            success_threshold: Optional threshold for determining success.
        Returns:
            A dictionary containing the average and standard deviation of rewards.
        """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True)

    # Build the metrics dictionary
    metrics = {
        "media_reward": mean_reward,      # Average reward
        "dev_std_reward": std_reward,     # Standard deviation of rewards
    }

    # Add success metric if a threshold is provided
    if success_threshold is not None:
        metrics["success"] = mean_reward >= success_threshold

    return metrics

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
        for key, value in metrics.items():
            if isinstance(value, (int, float)):  # For integers and floats
                f.write(f"{key}: {value:.2f}\n")
            elif isinstance(value, list):  # Handle lists (episode rewards)
                f.write(f"{key}: {value}\n")

    print(f"Metrics saved to: {filename}")




# Compare the performance of Random, PPO, and SAC models using plots
def plot_comparison(random_metrics, ppo_metrics, sac_metrics, output_path="results/plots/performance_comparison.png"):
    """
    Plot a comparison of performance metrics between Random, PPO, and SAC policies.
    Args:
        random_metrics: Metrics for the Random policy.
        ppo_metrics: Metrics for the PPO policy.
        sac_metrics: Metrics for the SAC policy.
        output_path: File path to save the generated plot.
    """
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

    # Add text labels for each bar
    for bar, mean in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{mean:.2f}",
                 ha='center', va='bottom', fontsize=10, color='black', weight='bold')

    # Customize the plot
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Performance Comparison: Random vs PPO vs SAC", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(min(0, min(mean_rewards) - max(std_devs) * 1.2), max(mean_rewards) + max(std_devs) * 1.2)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Performance comparison plot saved to: {output_path}")

    # Create a box plot for reward distributions
    plt.figure(figsize=(12, 8))
    data = [random_metrics["episodio_rewards"], ppo_metrics["episodio_rewards"], sac_metrics["episodio_rewards"]]
    plt.boxplot(data, labels=labels)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Reward Distribution: Random vs PPO vs SAC", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/plots/reward_distribution.png")
    plt.show()
    print("Reward distribution plot saved to: results/plots/reward_distribution.png")