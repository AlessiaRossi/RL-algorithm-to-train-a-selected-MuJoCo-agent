import os
from training import train_ppo, train_sac
from environments import make_env, run_random_policy

'''TO DO:
implemenatione main con config , cartelle di salvataggio metriche video e grafici
'''

def main():
    # Esecuzione di una random policy per osservare il comportamento iniziale
    env = make_env()
    random_rewards = run_random_policy(env)
    # Print dei risultati
    print(f"Random Policy Rewards per episode: {random_rewards}")
    print(f"Random Policy Media dei reward: {sum(random_rewards) / len(random_rewards)}")

    # Training dell'agente con algoritmo PPO
    ppo_model, ppo_env = train_ppo()
    ppo_rewards = ppo_env.get_attr('episode_rewards')
    # Print dei risultati
    print(f"PPO Rewards per episode: {ppo_rewards}")
    print(f"PPO Media dei reward: {sum(ppo_rewards) / len(ppo_rewards)}")

    # Training dell'agente con algoritmo SAC
    sac_model, sac_env = train_sac()
    sac_rewards = sac_env.get_attr('episode_rewards')
    # Print dei risultati
    print(f"SAC Rewards per episode: {sac_rewards}")
    print(f"SAC Media dei reward: {sum(sac_rewards) / len(sac_rewards)}")

if __name__ == "__main__":
    main()