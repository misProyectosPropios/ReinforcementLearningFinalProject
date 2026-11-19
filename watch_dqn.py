"""
Visualizador interactivo para ver el entorno Acrobot-v1 y las decisiones del agente DQN en vivo.

Abre una ventana que muestra durante cada episodio:
- El péndulo doble (Acrobot-v1) en movimiento.
- La acción elegida en cada paso (-1, 0, +1 torque) y los Q-values predichos por la red.
- La altura alcanzada por la punta del brazo vs. la línea objetivo (1.0).

Uso:
    python watch_dqn.py                              # Usa dqn_acrobot_best.pt o dqn_acrobot.pt
    python watch_dqn.py --model checkpoints/dqn_acrobot_ep0100.pt
    python watch_dqn.py --random                     # Comparar con un agente aleatorio
    python watch_dqn.py --mode human                 # Ventana nativa de Pygame + log en consola
    python watch_dqn.py --save-gif acrobot_demo.gif  # Guardar animación como GIF
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from dqn_acrobot import QNetwork

ACTION_LABELS = ["Torque -1\n(izq)", "Torque 0\n(libre)", "Torque +1\n(der)"]
ACTION_SHORT = {0: "-1 (izq)", 1: "0 (libre)", 2: "+1 (der)"}


def resolve_model_path(requested_path: Optional[str]) -> Optional[str]:
    if requested_path:
        return requested_path
    for candidate in ("dqn_acrobot_best.pt", "dqn_acrobot.pt"):
        if os.path.exists(candidate):
            return candidate
    return None


def tip_height_from_obs(obs: np.ndarray) -> float:
    """
    Calcula la altura de la punta del segundo eslabón a partir de la observación:
    obs = [cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]
    altura = -cos(t1) - cos(t1 + t2), donde cos(t1 + t2) = cos(t1)cos(t2) - sin(t1)sin(t2).
    El episodio termina exitosamente cuando altura > 1.0.
    """
    cos_t1, sin_t1, cos_t2, sin_t2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    cos_t1_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
    return -cos_t1 - cos_t1_t2


def load_agent(model_path: Optional[str], obs_dim: int, n_actions: int, device: torch.device) -> Optional[QNetwork]:
    if model_path is None:
        return None
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos '{model_path}'. "
            "Entrená primero con 'python dqn_acrobot.py' o usá '--random'."
        )
    q_net = QNetwork(obs_dim, n_actions).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    q_net.eval()
    return q_net


@torch.no_grad()
def select_action(
    q_net: Optional[QNetwork],
    obs: np.ndarray,
    env: gym.Env,
    device: torch.device,
) -> tuple[int, Optional[np.ndarray]]:
    if q_net is None:
        return int(env.action_space.sample()), None
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q_vals = q_net(obs_t).squeeze(0).cpu().numpy()
    action = int(np.argmax(q_vals))
    return action, q_vals


def run_human_mode(
    q_net: Optional[QNetwork],
    env_id: str,
    episodes: int,
    seed: int,
    fps: int,
    device: torch.device,
) -> None:
    """Abre la ventana nativa de Pygame de Gymnasium e imprime las acciones en consola."""
    env = gym.make(env_id, render_mode="human")
    delay = 1.0 / max(1, fps)
    agent_name = "DQN" if q_net is not None else "Aleatorio"

    print(f"Mostrando {episodes} episodio(s) en ventana Pygame ({agent_name})...")
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            action, q_vals = select_action(q_net, obs, env, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            total_reward += float(reward)
            done = terminated or truncated

            if step % 10 == 0 or done:
                q_str = (
                    f" | Q=[{q_vals[0]:6.2f}, {q_vals[1]:6.2f}, {q_vals[2]:6.2f}]"
                    if q_vals is not None
                    else ""
                )
                print(
                    f"Ep {ep} | Step {step:3d} | Acción: {ACTION_SHORT[action]:9s} | "
                    f"Altura: {tip_height_from_obs(obs):+.2f} | Reward: {total_reward:6.1f}{q_str}"
                )
            time.sleep(delay)

        status = "OBJETIVO ALCANZADO" if terminated else "TRUNCADO (500 pasos)"
        print(f"==> Fin Ep {ep}: {status} en {step} pasos (recompensa total: {total_reward:.1f})\n")

    env.close()


def run_dashboard_mode(
    q_net: Optional[QNetwork],
    env_id: str,
    episodes: int,
    seed: int,
    fps: int,
    device: torch.device,
    save_gif: Optional[str],
    headless: bool,
) -> None:
    """Abre una ventana interactiva con el péndulo, los Q-values y la altura de la punta en vivo."""
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    env = gym.make(env_id, render_mode="rgb_array")
    agent_name = "DQN entrenado" if q_net is not None else "Política aleatoria"
    gif_frames: List[np.ndarray] = []

    if not headless:
        plt.ion()

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, width_ratios=[1.35, 1.0], figure=fig)
    ax_env = fig.add_subplot(gs[:, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1])

    # Inicializar paneles
    obs, _ = env.reset(seed=seed + 1)
    first_frame = env.render()
    img_artist = ax_env.imshow(first_frame)
    ax_env.axis("off")

    bars = ax_q.bar(ACTION_LABELS, [0.0, 0.0, 0.0], color=["#4c72b0", "#4c72b0", "#4c72b0"])
    ax_q.set_ylabel("Q(s, a)")
    ax_q.set_title("Q-values predichos por la red (acción elegida en verde)")
    ax_q.grid(True, axis="y", alpha=0.3)

    (height_line,) = ax_h.plot([], [], color="#1f77b4", linewidth=2.0, label="Altura punta")
    ax_h.axhline(1.0, color="green", linestyle="--", linewidth=1.4, label="Objetivo (1.0)")
    ax_h.set_ylim(-2.1, 2.1)
    ax_h.set_xlabel("Paso del episodio")
    ax_h.set_ylabel("Altura")
    ax_h.set_title("Progreso hacia la altura objetivo")
    ax_h.grid(True, alpha=0.3)
    ax_h.legend(loc="lower right")

    info_text = ax_env.text(
        0.02,
        0.98,
        "",
        transform=ax_env.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
    )

    fig.tight_layout()
    pause_dt = 1.0 / max(1, fps)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step = 0
        total_reward = 0.0
        steps_hist: List[int] = [0]
        heights_hist: List[float] = [tip_height_from_obs(obs)]

        while not done:
            if not headless and not plt.fignum_exists(fig.number):
                env.close()
                return

            action, q_vals = select_action(q_net, obs, env, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            total_reward += float(reward)
            done = terminated or truncated

            h = tip_height_from_obs(obs)
            steps_hist.append(step)
            heights_hist.append(h)

            frame = env.render()
            img_artist.set_data(frame)

            # Actualizar barras de Q-values (o indicador de acción si es aleatorio)
            if q_vals is not None:
                for idx, (bar, val) in enumerate(zip(bars, q_vals)):
                    bar.set_height(float(val))
                    bar.set_color("#2ca02c" if idx == action else "#8da0cb")
                q_min, q_max = float(np.min(q_vals)), float(np.max(q_vals))
                margin = max(1.0, 0.25 * (q_max - q_min))
                ax_q.set_ylim(q_min - margin, q_max + margin)
            else:
                for idx, bar in enumerate(bars):
                    bar.set_height(1.0 if idx == action else 0.0)
                    bar.set_color("#2ca02c" if idx == action else "#cccccc")
                ax_q.set_ylim(0.0, 1.3)
                ax_q.set_ylabel("Acción activa")

            # Actualizar curva de altura
            height_line.set_data(steps_hist, heights_hist)
            ax_h.set_xlim(0, max(50, step + 5))

            status = "OBJETIVO!" if terminated else ("TRUNCADO" if truncated else "En curso")
            info_text.set_text(
                f"Agente : {agent_name}\n"
                f"Ep     : {ep}/{episodes} ({status})\n"
                f"Paso   : {step}\n"
                f"Reward : {total_reward:.0f}\n"
                f"Acción : {ACTION_SHORT[action]}\n"
                f"Altura : {h:+.2f} / 1.00"
            )
            ax_env.set_title(f"{env_id} — Episodio {ep} | Acción: {ACTION_SHORT[action]}")

            fig.canvas.draw()
            if save_gif:
                rgba = np.asarray(fig.canvas.buffer_rgba())
                gif_frames.append(rgba[:, :, :3].copy())

            if not headless:
                plt.pause(pause_dt)

        status_msg = "OBJETIVO ALCANZADO" if terminated else "TRUNCADO"
        print(f"Episodio {ep}/{episodes}: {status_msg} en {step} pasos | Recompensa total: {total_reward:.1f}")
        if not headless and ep < episodes:
            plt.pause(0.8)

    env.close()

    if save_gif and gif_frames:
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in gif_frames]
        duration_ms = int(1000 / max(1, fps))
        pil_frames[0].save(
            save_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Animación guardada en: {save_gif} ({len(gif_frames)} frames)")

    if not headless:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualizar agente DQN en vivo sobre Acrobot-v1")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta a los pesos .pt (por defecto busca dqn_acrobot_best.pt o dqn_acrobot.pt)",
    )
    parser.add_argument("--random", action="store_true", help="Usar acciones aleatorias en vez de un modelo")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "human"],
        default="dashboard",
        help="dashboard = ventana con péndulo + Q-values + altura; human = ventana Pygame nativa",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Cantidad de episodios a mostrar")
    parser.add_argument("--fps", type=int, default=20, help="Velocidad de reproducción (frames por segundo)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla inicial del entorno")
    parser.add_argument("--save-gif", type=str, default=None, help="Guardar la visualización en un archivo .gif")
    parser.add_argument("--headless", action="store_true", help="No abrir ventana (útil junto con --save-gif)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_probe = gym.make("Acrobot-v1")
    obs_dim = int(np.array(env_probe.observation_space.shape).prod())
    n_actions = int(env_probe.action_space.n)
    env_probe.close()

    if args.random:
        q_net = None
        print("Modo aleatorio activado (sin red neuronal).")
    else:
        model_path = resolve_model_path(args.model)
        if model_path is None:
            raise FileNotFoundError(
                "No se encontró dqn_acrobot_best.pt ni dqn_acrobot.pt. "
                "Ejecutá primero 'python dqn_acrobot.py' o usá '--random'."
            )
        print(f"Cargando modelo: {model_path}")
        q_net = load_agent(model_path, obs_dim, n_actions, device)

    if args.mode == "human":
        run_human_mode(q_net, "Acrobot-v1", args.episodes, args.seed, args.fps, device)
    else:
        run_dashboard_mode(
            q_net,
            "Acrobot-v1",
            args.episodes,
            args.seed,
            args.fps,
            device,
            save_gif=args.save_gif,
            headless=args.headless,
        )


if __name__ == "__main__":
    main()
