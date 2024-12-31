import os
from training import train_ppo, train_sac
from environments import make_env, run_random_policy
from analysis_results import evaluate_model, plot

'''TO DO:
implemenatione main con config , cartelle di salvataggio metriche video e grafici
'''

def main():
    # Esecuzione di una random policy per osservare il comportamento iniziale
    env = make_env()()
    random_rewards = run_random_policy(env)
    # Print dei risultati
    print(f"Random Policy Rewards per episode: {random_rewards}")
    print(f"Random Policy Media dei reward: {sum(random_rewards) / len(random_rewards)}")

    # Training dell'agente con algoritmo PPO
    ppo_model, ppo_env, ppo_rewards = train_ppo()
    # Print dei risultati
    #print(f"PPO Rewards per episode: {ppo_rewards}") non forniscono un grande contenuto informativo
    print(f"PPO Media dei reward: {sum(ppo_rewards) / len(ppo_rewards)}")
    print("Valutazione PPO...")
    ppo_metriche= evaluate_model(ppo_model, ppo_env)

    # Training dell'agente con algoritmo SAC
    sac_model, sac_env, sac_rewards = train_sac()
    # Print dei risultati
    #print(f"SAC Rewards per episode: {sac_rewards}")
    print(f"SAC Media dei reward: {sum(sac_rewards) / len(sac_rewards)}")
    print("Valutazione SAC...")
    sac_metriche = evaluate_model(sac_model, sac_env)

    # Stampa delle metriche
    print("\nMetriche PPO:")
    for k, v in ppo_metriche.items():
        print(f"{k}: {v:.2f}")
    print("\nMetriche SAC:")
    for k, v in sac_metriche.items():
        print(f"{k}: {v:.2f}")

    # Visualizzazione grafica
    plot(ppo_metriche, sac_metriche)


if __name__ == "__main__":
    main()