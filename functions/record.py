import os
import gym
from gym.wrappers import RecordVideo, TimeLimit, RecordEpisodeStatistics

# Registra il comportamento di un modello RL e salva i video degli episodi
def record_agent_video(model, env_id, max_steps=1000, seed=9999, video_dir="results/videos/", video_prefix="agent", episodes=2):
    # Creazione della directory per i video
    os.makedirs(video_dir, exist_ok=True)

    # Creazione dell'ambiente Gym con registrazione dei video
    env = gym.make(env_id, render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda e: True,
        name_prefix=video_prefix
    )
    env.reset(seed=seed)

    # Esecuzione del modello per un numero di episodi
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        print(f"Episodio {episode + 1}/{episodes} completato. Reward totale: {episode_reward}")

    # Chiusura dell'ambiente
    env.close()
    print(f"Video degli episodi '{video_prefix.replace("_agent", "")}' salvati in: {video_dir}")
