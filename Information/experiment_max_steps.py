"""
Experimentos de entrenamiento con PPO variando `max_steps`.
Compara las curvas de recompensa por episodio para distintos límites de pasos.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from maps import MAPA
from env import PacmanEnv

GHOST_CHASE_PROB = 0.7


class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards, self.wins, self.caught = [], [], []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.rewards.append(info["episode"]["r"])
                self.wins.append(float(info["win"]))
                self.caught.append(float(info["caught"]))
        return True


def media_movil(x, k=100):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    k = min(k, len(x))
    return np.convolve(x, np.ones(k) / k, mode="valid")


def run_max_steps_experiment(
    max_steps_list=None, total_timesteps=50_000, show_plots=True
):
    if max_steps_list is None:
        max_steps_list = [10, 50, 60, 75, 85, 100, 150, 200, 300]

    results = {}

    for ms in max_steps_list:
        print(f"\nEntrenando con max_steps = {ms}")

        def make_env_ms(max_s=ms):
            return PacmanEnv(MAPA, max_steps=max_s, ghost_chase_prob=GHOST_CHASE_PROB)

        vec_env_ms = make_vec_env(make_env_ms, n_envs=8, seed=0)

        model_ms = PPO(
            "MlpPolicy",
            vec_env_ms,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]),
            device="cpu",
            verbose=0,
            seed=0,
        )

        log_ms = LogCallback()
        model_ms.learn(total_timesteps=total_timesteps, callback=log_ms)
        results[ms] = log_ms.rewards

    # Plot individual
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, ms in enumerate(max_steps_list):
        if i < len(axes):
            axes[i].plot(media_movil(results[ms], k=50))
            axes[i].set_title(f"max_steps = {ms}")
            axes[i].set_xlabel("episodios")
            axes[i].set_ylabel("recompensa")
    plt.tight_layout()
    if show_plots:
        plt.show()

    # Plot comparativo
    plt.figure(figsize=(10, 6))
    for ms in max_steps_list:
        plt.plot(media_movil(results[ms], k=50), label=f"max_steps = {ms}")
    plt.title("Comparación de Recompensas por max_steps")
    plt.xlabel("episodios")
    plt.ylabel("recompensa (media móvil)")
    plt.legend()
    if show_plots:
        plt.show()

    return results


if __name__ == "__main__":
    run_max_steps_experiment()
