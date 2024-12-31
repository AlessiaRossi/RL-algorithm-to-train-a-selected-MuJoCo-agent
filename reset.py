import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()
print(obs)  # Dovrebbe stampare l'osservazione iniziale
print(info) # Dovrebbe stampare un dizionario con informazioni aggiuntive
