import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit, RecordEpisodeStatistics

# Function to record an RL agent's behavior and save videos of its episodes
def record_agent_video(model, env_id, max_steps=1000, seed=9999, video_dir="results/videos/", video_prefix="agent", episodes=2):
    """
    Records the behavior of a trained RL model and saves videos for a given number of episodes.

    Args:
        model: The trained RL model to be evaluated.
        env_id (str): ID of the Gym environment.
        max_steps (int): Maximum number of steps per episode.
        seed (int): Seed for environment reproducibility.
        video_dir (str): Directory to save the videos.
        video_prefix (str): Prefix for the saved video filenames.
        episodes (int): Number of episodes to record.
    """
    # Create a Gym environment with video recording enabled
    env = gym.make(env_id, render_mode="rgb_array")  # Environment with video rendering
    env = TimeLimit(env, max_episode_steps=max_steps)  # Limit the number of steps per episode
    env = RecordEpisodeStatistics(env)  # Track episode statistics (e.g., rewards, length)
    env = RecordVideo(  # Enable video recording for the environment
        env,
        video_folder=video_dir,  # Directory to store recorded videos
        episode_trigger=lambda e: True,  # Record all episodes
        name_prefix=video_prefix  # Prefix for video filenames
    )
    env.reset(seed=seed)  # Initialize the environment with the specified seed

    # Run the model for the specified number of episodes
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)  # Reset the environment for each episode
        done = False
        episode_reward = 0  # Accumulate rewards for the episode

        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Predict the next action
            obs, reward, terminated, truncated, info = env.step(action)  # Execute the action
            done = terminated or truncated  # Check if the episode has ended
            episode_reward += reward  # Update the total reward for the episode

        # Print the episode result
        print(f"Episode {episode + 1}/{episodes} completed. Total reward: {episode_reward}")

    # Close the environment after recording
    env.close()
    print(f"Videos for episodes '{video_prefix.replace('_agent', '')}' saved in: {video_dir}")
