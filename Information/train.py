"""
Entrenamiento principal de Pac-Man con PPO (Stable-Baselines3).
Usa 8 entornos en paralelo, grafica curvas de aprendizaje y guarda el modelo entrenado.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from maps import MAPA
from env import PacmanEnv

MAX_STEPS = 100
GHOST_CHASE_PROB = 0.7
TOTAL_TIMESTEPS = 400_000
MODEL_PATH = "ppo_pacman"


def make_env():
    return PacmanEnv(MAPA, max_steps=MAX_STEPS, ghost_chase_prob=GHOST_CHASE_PROB)


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


def plot_learning_curves(log, show_plots=True):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(media_movil(log.rewards))
    ax[0].set_title("Recompensa por episodio (media móvil)")
    ax[0].set_xlabel("episodios")

    ax[1].plot(media_movil(log.wins), label="gana")
    ax[1].plot(media_movil(log.caught), label="lo atrapan")
    ax[1].set_title("Tasa de victorias / capturas")
    ax[1].set_xlabel("episodios")
    ax[1].legend()

    plt.tight_layout()
    if show_plots:
        plt.show()


def train(total_timesteps=TOTAL_TIMESTEPS, save_path=MODEL_PATH, show_plots=True):
    vec_env = make_vec_env(make_env, n_envs=8, seed=0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,          # por entorno -> 256*8 = 2048 transiciones por update
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # bonus de entropía: favorece la exploración
        policy_kwargs=dict(net_arch=[128, 128]),
        device="cpu",         # para una MLP chica la CPU alcanza y sobra
        verbose=1,
        seed=0,
    )

    log = LogCallback()
    model.learn(total_timesteps=total_timesteps, callback=log)

    if save_path:
        model.save(save_path)
        print(f"Modelo guardado en '{save_path}.zip'")

    plot_learning_curves(log, show_plots=show_plots)
    return model, log


if __name__ == "__main__":
    train()
