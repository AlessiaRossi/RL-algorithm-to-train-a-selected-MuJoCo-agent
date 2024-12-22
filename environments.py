import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit, RecordEpisodeStatistics

# Inizializzazione dell'environment
env_name = "HalfCheetah-v4"  # Specificazione dell'environment utilizzato (HalfCheetah-v5)
max_episode_steps=1000
seed=0
render_mode=None
env = gym.make(env_name) # Creazione dell'environment
# Limitiamo la durata massima dell'episodio
env = TimeLimit(env, max_episode_steps=max_episode_steps)
# Registra statistiche (reward cumulativo, length, ecc.)
env = RecordEpisodeStatistics(env)
# Seed dell'ambiente per riproducibilità
env.reset(seed=seed)

# Run una random policy per osservare il comportamento iniziale
print("Running a random policy...")
episodi= 5
rewards_per_episode = []
for episodio in range(episodi):
    obs, info = env.reset(seed=seed + episodio)
    episode_reward = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break

    rewards_per_episode.append(episode_reward)

    progress = 100.0 * (episodio + 1) / episodi
    print(f"[Random Policy] Episodio {episodio + 1}/{episodi} - Avanzamento: {progress:.1f}%")

env.close()
