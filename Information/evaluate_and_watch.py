"""
Evaluación y visualización / animación del agente PPO jugando a Pac-Man.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

from maps import MAPA
from env import PacmanEnv
from train import MAX_STEPS, GHOST_CHASE_PROB, MODEL_PATH, make_env


def evaluar(model, n=200, deterministic=False):
    env = make_env()
    wins = caught = timeouts = 0
    pasos = []
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(int(a))
            done = term or trunc
        wins += info["win"]
        caught += info["caught"]
        timeouts += trunc and not term
        pasos.append(env.t)
    print(
        f"Victorias: {wins/n:.1%} | Capturas: {caught/n:.1%} | "
        f"Timeouts: {timeouts/n:.1%} | Pasos medios: {np.mean(pasos):.1f}"
    )


def jugar_episodio(model, deterministic=False, seed=None):
    env = PacmanEnv(
        MAPA,
        max_steps=MAX_STEPS,
        ghost_chase_prob=GHOST_CHASE_PROB,
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(a))
        frames.append(env.render())
        done = term or trunc
    return frames, info


def animar(frames, intervalo=250, show=True, save_html_path=None):
    fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 64, frames[0].shape[0] / 64))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=intervalo, blit=True
    )

    if save_html_path:
        with open(save_html_path, "w", encoding="utf-8") as f:
            f.write(anim.to_jshtml())

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim


if __name__ == "__main__":
    model = PPO.load(MODEL_PATH, device="cpu")

    print("Evaluando con deterministic=False:")
    evaluar(model, deterministic=False)

    print("Evaluando con deterministic=True:")
    evaluar(model, deterministic=True)

    frames, info = jugar_episodio(model, deterministic=True, seed=42)
    print(info, "| pasos:", len(frames) - 1)
    animar(frames)
