"""
Entrenamiento y comparación de DQN en el entorno de Observabilidad Parcial en
Línea Recta (`PartialObsPacmanEnv`), comparando un agente sin memoria (solo el
frame actual, `k = 1`) contra agentes que reciben un historial de `k` frames
pasados apilados (`k in [1, 2, 4, 8, 16]`).

¿Por qué importa el historial de frames en observabilidad parcial?
- Con `k = 1`, cuando Pac-Man dobla una esquina deja de ver los pasillos anteriores
  y olvida dónde quedaban bolitas o por dónde venía un fantasma.
- Al apilar los últimos `k` frames (`k = 2, 4, 8, 16`), el agente dispone de una
  ventana temporal del pasado reciente que le permite recordar qué pasillos ya
  limpió y dónde vio por última vez al fantasma antes de perder línea de visión.
"""

from __future__ import annotations

import argparse
from collections import deque
import random
import time
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maps import MAPA
from env_partial import PartialObsPacmanEnv
from train_dqn import (
    PacmanQNetwork,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_ACTIVATION,
    ReplayBuffer,
    linear_schedule,
    media_movil,
    set_seed,
)


# ===========================================================================
# 1. WRAPPER DE HISTORIAL DE FRAMES PASADOS (FRAME STACKING VECTORIAL)
# ===========================================================================
class PartialFrameStackWrapper(gym.Wrapper):
    """
    Mantiene una cola (`deque`) con las últimas `num_stack` observaciones parciales
    `(obs_{t - num_stack + 1}, ..., obs_{t - 1}, obs_t)` y las concatena en un único
    vector de dimensión `(num_stack * base_obs_dim,)`.

    Si `num_stack = 1`, devuelve exactamente la observación parcial actual sin historial.
    """

    def __init__(self, env: PartialObsPacmanEnv, num_stack: int = 1) -> None:
        super().__init__(env)
        if num_stack < 1:
            raise ValueError("num_stack debe ser >= 1.")
        self.num_stack = num_stack
        self.base_obs_dim = int(env.observation_space.shape[0])
        self.history: deque = deque(maxlen=num_stack)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_stack * self.base_obs_dim,),
            dtype=np.float32,
        )

    def _get_stacked_obs(self) -> np.ndarray:
        if self.num_stack == 1:
            return self.history[0]
        return np.concatenate(list(self.history), axis=0)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.history.clear()
        for _ in range(self.num_stack):
            self.history.append(obs.astype(np.float32))
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs.astype(np.float32))
        return self._get_stacked_obs(), reward, terminated, truncated, info


# ===========================================================================
# 2. ENTRENAMIENTO Y EVALUACIÓN DE DQN CON `num_stack` FRAMES PASADOS
# ===========================================================================
def make_stacked_partial_env(
    num_stack: int = 1,
    mapa: List[str] = MAPA,
    max_steps: int = 100,
    ghost_chase_prob: float = 0.7,
) -> PartialFrameStackWrapper:
    base_env = PartialObsPacmanEnv(
        mapa,
        max_steps=max_steps,
        ghost_chase_prob=ghost_chase_prob,
    )
    return PartialFrameStackWrapper(base_env, num_stack=num_stack)


def evaluate_stacked_dqn(
    q_net: nn.Module,
    num_stack: int,
    device: torch.device,
    max_steps: int = 100,
    ghost_chase_prob: float = 0.7,
    n_episodes: int = 100,
    epsilon: float = 0.02,
) -> Dict[str, float]:
    env = make_stacked_partial_env(
        num_stack=num_stack,
        max_steps=max_steps,
        ghost_chase_prob=ghost_chase_prob,
    )
    q_net.eval()
    wins = caught = timeouts = 0
    rewards_list: List[float] = []
    pellets_pct_list: List[float] = []
    steps_list: List[int] = []

    with torch.no_grad():
        for i in range(n_episodes):
            obs, info = env.reset(seed=6000 + i)
            init_pellets = max(1, info["pellets_left"])
            done = False
            ep_r = 0.0
            while not done:
                if random.random() < epsilon:
                    action = int(env.action_space.sample())
                else:
                    obs_t = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action = int(torch.argmax(q_net(obs_t), dim=1).item())
                obs, r, term, trunc, info = env.step(action)
                ep_r += r
                done = term or trunc

            wins += int(info["win"])
            caught += int(info["caught"])
            timeouts += int(trunc and not term)
            rewards_list.append(ep_r)
            pellets_pct_list.append(
                (init_pellets - info["pellets_left"]) / float(init_pellets)
            )
            steps_list.append(env.env.t)

    q_net.train()
    return {
        "win_rate": wins / n_episodes,
        "caught_rate": caught / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "pellets_eaten_pct": float(np.mean(pellets_pct_list)),
        "mean_reward": float(np.mean(rewards_list)),
        "mean_steps": float(np.mean(steps_list)),
    }


def train_partial_dqn_for_stack(
    num_stack: int,
    total_timesteps: int = 120_000,
    hidden_layers: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
    activation: str = DEFAULT_ACTIVATION,
    max_steps: int = 100,
    ghost_chase_prob: float = 0.7,
    seed: int = 0,
    eval_episodes: int = 100,
    save_model: bool = True,
) -> Dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_stacked_partial_env(
        num_stack=num_stack,
        max_steps=max_steps,
        ghost_chase_prob=ghost_chase_prob,
    )
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    q_net = PacmanQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device)
    target_net = PacmanQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    num_params = sum(p.numel() for p in q_net.parameters())
    label = (
        "1 frame (solo actual, sin pasado)"
        if num_stack == 1
        else f"{num_stack} frames (actual + {num_stack - 1} pasados)"
    )
    print("\n" + "=" * 78)
    print(f"Entrenando DQN en PartialObsPacmanEnv | Historial: {label}")
    print(
        f"  Dimensión de entrada: {obs_dim:,} | Capas ReLU: {list(hidden_layers)} | "
        f"Parámetros: {num_params:,} | Pasos: {total_timesteps:,}"
    )
    print("=" * 78)

    optimizer = optim.Adam(q_net.parameters(), lr=3e-4)
    # Ajustar capacidad del ReplayBuffer para que 16 frames no consuma memoria excesiva
    buffer_capacity = 40_000 if num_stack <= 8 else 25_000
    rb = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim)

    batch_size = 128
    learning_starts = 2_000
    train_frequency = 4
    target_update_freq = 500
    gamma = 0.99
    exploration_steps = int(0.4 * total_timesteps)

    ep_rewards: List[float] = []
    ep_wins: List[float] = []
    ep_caught: List[float] = []
    ep_pellets_pct: List[float] = []
    ep_end_step: List[int] = []

    start_time = time.perf_counter()
    obs, info = env.reset(seed=seed)
    init_pellets = max(1, info["pellets_left"])
    cur_reward = 0.0

    for step in range(1, total_timesteps + 1):
        epsilon = linear_schedule(1.0, 0.05, exploration_steps, step)

        if random.random() < epsilon:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = int(torch.argmax(q_net(obs_t), dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rb.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        cur_reward += reward

        if step >= learning_starts and step % train_frequency == 0:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = rb.sample(
                batch_size, device
            )

            with torch.no_grad():
                next_actions = q_net(b_next_obs).argmax(dim=1, keepdim=True)
                next_q_target = target_net(b_next_obs).gather(1, next_actions)
                td_target = b_rewards + gamma * next_q_target * (1.0 - b_dones)

            current_q = q_net(b_obs).gather(1, b_actions)
            loss = F.smooth_l1_loss(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            optimizer.step()

        if step >= learning_starts and step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            ep_rewards.append(cur_reward)
            ep_wins.append(float(info["win"]))
            ep_caught.append(float(info["caught"]))
            eaten_pct = (init_pellets - info["pellets_left"]) / float(init_pellets)
            ep_pellets_pct.append(eaten_pct)
            ep_end_step.append(step)

            if len(ep_rewards) % 250 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"  [stack={num_stack:2d}] Paso {step:7d}/{total_timesteps} | "
                    f"Ep {len(ep_rewards):5d} | Tiempo: {elapsed:5.1f}s | "
                    f"Recomp(100): {np.mean(ep_rewards[-100:]):6.2f} | "
                    f"Victorias(100): {np.mean(ep_wins[-100:]):5.1%} | "
                    f"Bolitas(100): {np.mean(ep_pellets_pct[-100:]):5.1%}"
                )

            obs, info = env.reset()
            init_pellets = max(1, info["pellets_left"])
            cur_reward = 0.0

    total_time = time.perf_counter() - start_time

    if save_model:
        save_path = f"dqn_partial_stack{num_stack}.pt"
        torch.save(
            {
                "model_state_dict": q_net.state_dict(),
                "num_stack": num_stack,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_layers": list(hidden_layers),
                "activation": activation,
            },
            save_path,
        )

    eval_metrics = evaluate_stacked_dqn(
        q_net,
        num_stack=num_stack,
        device=device,
        max_steps=max_steps,
        ghost_chase_prob=ghost_chase_prob,
        n_episodes=eval_episodes,
    )
    print(
        f"  -> Evaluación Final [stack={num_stack:2d}]: "
        f"Victorias: {eval_metrics['win_rate']:5.1%} | "
        f"Capturas: {eval_metrics['caught_rate']:5.1%} | "
        f"Bolitas comidas: {eval_metrics['pellets_eaten_pct']:5.1%} | "
        f"Recompensa media: {eval_metrics['mean_reward']:6.2f}"
    )

    return {
        "num_stack": num_stack,
        "obs_dim": obs_dim,
        "num_params": num_params,
        "total_time_sec": total_time,
        "ep_rewards": ep_rewards,
        "ep_wins": ep_wins,
        "ep_caught": ep_caught,
        "ep_pellets_pct": ep_pellets_pct,
        "ep_end_step": ep_end_step,
        **eval_metrics,
    }


# ===========================================================================
# 3. TABLA RESUMEN Y GRÁFICOS COMPARATIVOS (1 vs 2, 4, 8, 16 FRAMES)
# ===========================================================================
def print_stack_comparison_table(results: Dict[int, Dict]) -> None:
    print("\n" + "=" * 105)
    print("COMPARACIÓN DQN EN OBSERVABILIDAD PARCIAL SEGÚN CANTIDAD DE FRAMES PASADOS (1 vs 2, 4, 8, 16)")
    print("=" * 105)
    header = (
        f"{'Frames (k)':<12} | {'Dim Obs':>8} | {'Parámetros':>11} | "
        f"{'Tiempo':>8} | {'Victorias':>9} | {'Capturas':>9} | "
        f"{'% Bolitas':>9} | {'Recompensa Media':>16}"
    )
    print(header)
    print("-" * 105)
    for k, res in results.items():
        tag = f"k={k} (actual)" if k == 1 else f"k={k} (+{k-1} pas.)"
        print(
            f"{tag:<12} | {res['obs_dim']:>8,} | {res['num_params']:>11,} | "
            f"{res['total_time_sec']:>7.1f}s | {res['win_rate']:>8.1%} | "
            f"{res['caught_rate']:>8.1%} | {res['pellets_eaten_pct']:>8.1%} | "
            f"{res['mean_reward']:>16.2f}"
        )
    print("=" * 105)


def plot_stack_comparison(
    results: Dict[int, Dict],
    show_plots: bool = True,
    save_path: str = "partial_obs_frame_stack_comparison.png",
) -> None:
    stacks = list(results.keys())
    labels = [f"k={k}" for k in stacks]
    x = np.arange(len(stacks))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    window = 50

    # 1) Curvas de aprendizaje: Recompensa por episodio (media móvil)
    ax = axes[0, 0]
    for k, res in results.items():
        r_smooth = media_movil(res["ep_rewards"], k=window)
        steps_x = res["ep_end_step"][len(res["ep_end_step"]) - len(r_smooth) :]
        ax.plot(steps_x, r_smooth, label=f"k = {k} frames")
    ax.set_title("Curvas de Aprendizaje: Recompensa Media Móvil vs. Pasos")
    ax.set_xlabel("Pasos de entorno (timesteps)")
    ax.set_ylabel("Recompensa por episodio")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Curvas de aprendizaje: % de Bolitas Comidas durante el entrenamiento
    ax = axes[0, 1]
    for k, res in results.items():
        p_smooth = media_movil(res["ep_pellets_pct"], k=window) * 100.0
        steps_x = res["ep_end_step"][len(res["ep_end_step"]) - len(p_smooth) :]
        ax.plot(steps_x, p_smooth, label=f"k = {k} frames")
    ax.set_title("Progreso en el Mapa: % de Bolitas Comidas vs. Pasos")
    ax.set_xlabel("Pasos de entorno (timesteps)")
    ax.set_ylabel("Bolitas comidas (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) Gráfico de barras final: Tasa de Victorias (%) y % de Bolitas Comidas
    ax = axes[1, 0]
    width = 0.36
    win_rates = [results[k]["win_rate"] * 100.0 for k in stacks]
    pellet_rates = [results[k]["pellets_eaten_pct"] * 100.0 for k in stacks]
    ax.bar(x - width / 2, win_rates, width, label="Tasa de Victorias (%)")
    ax.bar(x + width / 2, pellet_rates, width, label="Bolitas Comidas (%)")
    ax.set_title("Evaluación Final por Cantidad de Frames Pasados (k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Porcentaje (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4) Gráfico de barras final: Recompensa Media en Evaluación
    ax = axes[1, 1]
    mean_rewards = [results[k]["mean_reward"] for k in stacks]
    bars = ax.bar(x, mean_rewards, width=0.5, color="tab:green", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Recompensa Media Final según Historial de Frames (k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Recompensa media por episodio")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mean_rewards):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.4 if val >= 0 else -1.2),
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nGráfico comparativo guardado en '{save_path}'")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def run_frame_stack_experiment(
    stack_sizes: Sequence[int] = (1, 2, 4, 8, 16),
    total_timesteps: int = 100_000,
    hidden_layers: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
    eval_episodes: int = 100,
    show_plots: bool = True,
) -> Dict[int, Dict]:
    results: Dict[int, Dict] = {}
    for k in stack_sizes:
        results[k] = train_partial_dqn_for_stack(
            num_stack=k,
            total_timesteps=total_timesteps,
            hidden_layers=hidden_layers,
            eval_episodes=eval_episodes,
        )

    print_stack_comparison_table(results)
    plot_stack_comparison(results, show_plots=show_plots)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar DQN en PartialObsPacmanEnv sin memoria (k=1) vs. con k frames anteriores (2, 4, 8, 16)."
    )
    parser.add_argument(
        "--stacks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Cantidades de frames apilados a comparar (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Pasos de entrenamiento por cada configuración de frames (default: 100000)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_LAYERS,
        help="Capas ocultas ReLU de la red DQN (default: 256 256 128)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Episodios de evaluación final para cada modelo (default: 100)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar ventana de matplotlib al terminar (igual guarda el PNG)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_frame_stack_experiment(
        stack_sizes=args.stacks,
        total_timesteps=args.timesteps,
        hidden_layers=args.layers,
        eval_episodes=args.eval_episodes,
        show_plots=not args.no_show,
    )
