"""
Comparación entre:
  1. **DQN con Información Completa del Entorno (`full_info_obs=True` + MLP ReLU)**
  2. **DQN aprendiendo desde Imágenes / Píxeles (estilo paper original de Mnih et al., 2013/2015)**
     usando una Red Neuronal Convolucional (CNN) sobre la imagen renderizada del juego
     con `frame_stack` (apilado de los últimos K frames para capturar velocidad/dirección).

Permite medir y comparar:
  - **Tiempo de cómputo (segundos)** y **velocidad (FPS / pasos por segundo)** de cada enfoque.
  - **Eficiencia muestral (Sample Efficiency)**: cuántos episodios / pasos de entorno
    necesita cada agente para empezar a comer bolitas y alcanzar distintos umbrales de
    tasa de victorias (ej. 25%, 50%, 75%).
  - **Desempeño final** evaluado en igualdad de condiciones (Victorias %, Capturas %,
    % de Bolitas Comidas y Recompensa media).
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maps import MAPA
from env import PacmanEnv
from train_dqn import (
    DQNConfig,
    PacmanQNetwork,
    DEFAULT_HIDDEN_LAYERS,
    linear_schedule,
    media_movil,
    set_seed,
)


# ===========================================================================
# 1. WRAPPER DE IMAGEN Y FRAME STACKING (ESTILO PAPER ORIGINAL DQN)
# ===========================================================================
class PacmanImageEnv(gym.Wrapper):
    """
    Envuelve `PacmanEnv` para que la observación sea directamente la imagen en píxeles
    renderizada por `_draw_rgb(S=tile_size)`, opcionalmente en escala de grises o RGB,
    y apilando los últimos `frame_stack` frames (como los 4 frames del paper de DQN).

    Formato de salida: `(C, H_img, W_img)` en `uint8` [0..255] para ahorrar memoria en
    el ReplayBuffer; la red CNN lo normaliza a `[0.0, 1.0]` en el `forward()`.
    """

    def __init__(
        self,
        env: PacmanEnv,
        tile_size: int = 8,
        grayscale: bool = True,
        frame_stack: int = 4,
    ) -> None:
        super().__init__(env)
        self.tile_size = tile_size
        self.grayscale = grayscale
        self.frame_stack = frame_stack

        self.img_h = env.H * tile_size
        self.img_w = env.W * tile_size
        self.channels_per_frame = 1 if grayscale else 3
        self.total_channels = self.channels_per_frame * frame_stack

        self.frames: deque = deque(maxlen=frame_stack)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.total_channels, self.img_h, self.img_w),
            dtype=np.uint8,
        )

    def _get_pixel_frame(self) -> np.ndarray:
        # Renderiza el tablero a resolución (H * tile_size, W * tile_size, 3)
        rgb = self.env._draw_rgb(S=self.tile_size)
        if self.grayscale:
            # Luminancia perceptual con pesos distintos para distinguir pared, bolita, Pac-Man y fantasma
            gray = (
                0.299 * rgb[:, :, 0]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            ).astype(np.uint8)
            return gray[np.newaxis, :, :]  # (1, H_img, W_img)
        # Formato PyTorch channels-first: (3, H_img, W_img)
        return np.transpose(rgb, (2, 0, 1)).copy()

    def _stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        frame = self._get_pixel_frame()
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._stacked_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._get_pixel_frame())
        return self._stacked_obs(), reward, terminated, truncated, info


# ===========================================================================
# 2. RED NEURONAL CONVOLUCIONAL (CNN ESTILO PAPER DQN - EDITABLE)
# ===========================================================================
class PacmanCNNQNetwork(nn.Module):
    """
    Arquitectura CNN inspirada en el paper original de DQN (Mnih et al., 2013/2015),
    adaptada al tamaño de imagen del tablero de Pac-Man:
      - 3 capas convolucionales con activación ReLU
      - Seguida de 2 capas lineales (fully-connected) con ReLU que predicen Q(s, a).

    Podés modificar `conv_channels` o `fc_layers` para experimentar con distintas CNNs.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        conv_channels: Sequence[int] = (32, 64, 64),
        fc_layers: Sequence[int] = (256,),
    ) -> None:
        super().__init__()
        in_channels, img_h, img_w = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcular automáticamente el tamaño aplanado tras las convoluciones
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            flat_dim = int(self.conv(dummy).shape[1])

        mlp_layers: List[nn.Module] = []
        in_dim = flat_dim
        for h_dim in fc_layers:
            mlp_layers.append(nn.Linear(in_dim, h_dim))
            mlp_layers.append(nn.ReLU())
            in_dim = h_dim
        mlp_layers.append(nn.Linear(in_dim, action_dim))

        self.fc = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normaliza píxeles uint8 [0..255] a float32 [0.0..1.0]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:
            x = x / 255.0
        features = self.conv(x)
        return self.fc(features)


# ===========================================================================
# 3. REPLAY BUFFER PARA IMÁGENES (UINT8 EFICIENTE EN MEMORIA)
# ===========================================================================
class ImageReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int]) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idxs], device=device),
            torch.as_tensor(self.actions[idxs], device=device),
            torch.as_tensor(self.rewards[idxs], device=device),
            torch.as_tensor(self.next_obs[idxs], device=device),
            torch.as_tensor(self.dones[idxs], device=device),
        )


# ===========================================================================
# 4. ENTRENAMIENTO Y REGISTRO DE CURVAS DE APRENDIZAJE / TIEMPOS
# ===========================================================================
def make_image_env(
    max_steps: int = 100,
    ghost_chase_prob: float = 0.7,
    tile_size: int = 8,
    grayscale: bool = True,
    frame_stack: int = 4,
) -> PacmanImageEnv:
    base_env = PacmanEnv(
        MAPA,
        max_steps=max_steps,
        ghost_chase_prob=ghost_chase_prob,
        full_info_obs=False,
    )
    return PacmanImageEnv(
        base_env,
        tile_size=tile_size,
        grayscale=grayscale,
        frame_stack=frame_stack,
    )


def train_agent_tracked(
    mode: str,
    total_timesteps: int = 120_000,
    seed: int = 0,
    max_steps: int = 100,
    ghost_chase_prob: float = 0.7,
    mlp_layers: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
    tile_size: int = 8,
    grayscale: bool = True,
    frame_stack: int = 4,
    eval_episodes: int = 100,
) -> Dict:
    """
    Entrena un agente DQN en modo:
      - `mode="full_info"`: vector con toda la información del entorno + MLP ReLU.
      - `mode="image_cnn"`: píxeles crudos apilados (`frame_stack`) + CNN ReLU (paper DQN).
    Registra paso a paso el tiempo transcurrido, episodios para alcanzar metas de victoria
    y métricas de evaluación final.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "full_info":
        env = PacmanEnv(
            MAPA,
            max_steps=max_steps,
            ghost_chase_prob=ghost_chase_prob,
            full_info_obs=True,
        )
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)
        q_net = PacmanQNetwork(obs_dim, action_dim, hidden_layers=mlp_layers).to(device)
        target_net = PacmanQNetwork(obs_dim, action_dim, hidden_layers=mlp_layers).to(device)
        from train_dqn import ReplayBuffer

        rb = ReplayBuffer(capacity=50_000, obs_dim=obs_dim)
        desc = f"DQN Full-Info MLP ({list(mlp_layers)} ReLU, dim={obs_dim})"
    elif mode == "image_cnn":
        env = make_image_env(
            max_steps=max_steps,
            ghost_chase_prob=ghost_chase_prob,
            tile_size=tile_size,
            grayscale=grayscale,
            frame_stack=frame_stack,
        )
        obs_shape = env.observation_space.shape
        action_dim = int(env.action_space.n)
        q_net = PacmanCNNQNetwork(obs_shape, action_dim).to(device)
        target_net = PacmanCNNQNetwork(obs_shape, action_dim).to(device)
        rb = ImageReplayBuffer(capacity=40_000, obs_shape=obs_shape)
        desc = f"DQN Image CNN (shape={obs_shape}, stack={frame_stack})"
    else:
        raise ValueError(f"Modo desconocido: {mode}")

    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    num_params = sum(p.numel() for p in q_net.parameters())
    print("\n" + "=" * 78)
    print(f"Entrenando: {desc}")
    print(f"  Parámetros de la red: {num_params:,} | Pasos: {total_timesteps:,} | Dispositivo: {device}")
    print("=" * 78)

    optimizer = optim.Adam(q_net.parameters(), lr=3e-4)
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
    ep_end_time: List[float] = []

    # Hitos de eficiencia muestral (primer paso/episodio donde la media móvil de 50 ep supera X% victorias)
    thresholds = [0.10, 0.25, 0.50, 0.75]
    milestones: Dict[float, Optional[Dict]] = {th: None for th in thresholds}

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
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                if mode == "full_info":
                    obs_t = obs_t.float()
                q_values = q_net(obs_t)
                action = int(torch.argmax(q_values, dim=1).item())

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
            elapsed = time.perf_counter() - start_time
            ep_rewards.append(cur_reward)
            ep_wins.append(float(info["win"]))
            ep_caught.append(float(info["caught"]))
            eaten_pct = (init_pellets - info["pellets_left"]) / float(init_pellets)
            ep_pellets_pct.append(eaten_pct)
            ep_end_step.append(step)
            ep_end_time.append(elapsed)

            # Verificar hitos de aprendizaje (sobre ventana de últimos 50 episodios)
            if len(ep_wins) >= 50:
                rolling_win = float(np.mean(ep_wins[-50:]))
                for th in thresholds:
                    if milestones[th] is None and rolling_win >= th:
                        milestones[th] = {
                            "episode": len(ep_wins),
                            "step": step,
                            "time_sec": elapsed,
                        }

            if len(ep_rewards) % 200 == 0:
                fps = step / max(1e-6, elapsed)
                print(
                    f"  Paso {step:7d}/{total_timesteps} | Ep {len(ep_rewards):5d} | "
                    f"Tiempo: {elapsed:5.1f}s ({fps:5.0f} FPS) | "
                    f"Recomp(100): {np.mean(ep_rewards[-100:]):6.2f} | "
                    f"Victorias(100): {np.mean(ep_wins[-100:]):5.1%} | "
                    f"Bolitas(100): {np.mean(ep_pellets_pct[-100:]):5.1%}"
                )

            obs, info = env.reset()
            init_pellets = max(1, info["pellets_left"])
            cur_reward = 0.0

    total_time = time.perf_counter() - start_time
    avg_fps = total_timesteps / max(1e-6, total_time)

    # Evaluación final del agente
    q_net.eval()
    eval_wins = eval_caught = eval_timeouts = 0
    eval_rewards: List[float] = []
    eval_pellets: List[float] = []
    eval_steps: List[int] = []

    with torch.no_grad():
        for i in range(eval_episodes):
            obs, info = env.reset(seed=8000 + i)
            init_p = max(1, info["pellets_left"])
            done = False
            ep_r = 0.0
            t_count = 0
            while not done:
                t_count += 1
                if random.random() < 0.02:
                    action = int(env.action_space.sample())
                else:
                    obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                    if mode == "full_info":
                        obs_t = obs_t.float()
                    action = int(torch.argmax(q_net(obs_t), dim=1).item())
                obs, r, term, trunc, info = env.step(action)
                ep_r += r
                done = term or trunc

            eval_wins += int(info["win"])
            eval_caught += int(info["caught"])
            eval_timeouts += int(trunc and not term)
            eval_rewards.append(ep_r)
            eval_pellets.append((init_p - info["pellets_left"]) / float(init_p))
            eval_steps.append(t_count)

    return {
        "mode": mode,
        "description": desc,
        "num_params": num_params,
        "total_timesteps": total_timesteps,
        "total_episodes": len(ep_rewards),
        "total_time_sec": total_time,
        "fps": avg_fps,
        "ep_rewards": ep_rewards,
        "ep_wins": ep_wins,
        "ep_pellets_pct": ep_pellets_pct,
        "ep_end_step": ep_end_step,
        "ep_end_time": ep_end_time,
        "milestones": milestones,
        "eval_win_rate": eval_wins / eval_episodes,
        "eval_caught_rate": eval_caught / eval_episodes,
        "eval_timeout_rate": eval_timeouts / eval_episodes,
        "eval_pellets_pct": float(np.mean(eval_pellets)),
        "eval_mean_reward": float(np.mean(eval_rewards)),
        "eval_mean_steps": float(np.mean(eval_steps)),
    }


# ===========================================================================
# 5. RESUMEN COMPARATIVO Y GRÁFICOS (EFICIENCIA MUESTRAL Y TIEMPO)
# ===========================================================================
def print_comparison_report(full_res: Dict, img_res: Dict) -> None:
    print("\n" + "=" * 95)
    print("COMPARACIÓN: DQN INFORMACIÓN COMPLETA (MLP) vs. DQN IMÁGENES / PÍXELES (CNN - PAPER DQN)")
    print("=" * 95)
    print(
        f"{'Métrica':<38} | {'Full-Info (MLP ReLU)':>25} | {'Imágenes (CNN + FrameStack)':>25}"
    )
    print("-" * 95)
    print(
        f"{'Parámetros entrenables':<38} | {full_res['num_params']:>25,} | {img_res['num_params']:>25,}"
    )
    print(
        f"{'Tiempo total de entrenamiento':<38} | {full_res['total_time_sec']:>24.1f}s | {img_res['total_time_sec']:>24.1f}s"
    )
    print(
        f"{'Velocidad de entrenamiento (FPS)':<38} | {full_res['fps']:>21.0f} FPS | {img_res['fps']:>21.0f} FPS"
    )
    print(
        f"{'Episodios completados':<38} | {full_res['total_episodes']:>25,} | {img_res['total_episodes']:>25,}"
    )
    print("-" * 95)
    print(
        f"{'Evaluación Final - Tasa de Victorias':<38} | {full_res['eval_win_rate']:>24.1%} | {img_res['eval_win_rate']:>24.1%}"
    )
    print(
        f"{'Evaluación Final - Tasa de Capturas':<38} | {full_res['eval_caught_rate']:>24.1%} | {img_res['eval_caught_rate']:>24.1%}"
    )
    print(
        f"{'Evaluación Final - % Bolitas Comidas':<38} | {full_res['eval_pellets_pct']:>24.1%} | {img_res['eval_pellets_pct']:>24.1%}"
    )
    print(
        f"{'Evaluación Final - Recompensa Media':<38} | {full_res['eval_mean_reward']:>25.2f} | {img_res['eval_mean_reward']:>25.2f}"
    )
    print("-" * 95)
    print("CUÁNTO TARDA EN APRENDER (Pasos / Tiempo para alcanzar X% de victorias en media móvil de 50 ep):")
    for th in [0.10, 0.25, 0.50, 0.75]:
        m_full = full_res["milestones"][th]
        m_img = img_res["milestones"][th]
        s_full = (
            f"{m_full['step']:,} pasos ({m_full['time_sec']:.1f}s)"
            if m_full
            else "No alcanzado"
        )
        s_img = (
            f"{m_img['step']:,} pasos ({m_img['time_sec']:.1f}s)"
            if m_img
            else "No alcanzado"
        )
        print(f"  Alcanzar >= {th * 100:4.0f}% victorias          | {s_full:>25} | {s_img:>25}")
    print("=" * 95)


def plot_full_vs_image(
    full_res: Dict,
    img_res: Dict,
    show_plots: bool = True,
    save_path: str = "comparison_full_info_vs_image_dqn.png",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    window = 50

    # 1) Recompensa vs. Pasos de Entorno (Sample Efficiency)
    ax = axes[0, 0]
    for res, label, color in [
        (full_res, "Full-Info (MLP ReLU)", "tab:blue"),
        (img_res, "Imágenes Píxeles (CNN Paper DQN)", "tab:orange"),
    ]:
        r_smooth = media_movil(res["ep_rewards"], k=window)
        steps_x = res["ep_end_step"][len(res["ep_end_step"]) - len(r_smooth) :]
        ax.plot(steps_x, r_smooth, label=label, color=color)
    ax.set_title("Eficiencia Muestral: Recompensa vs. Pasos de Entorno")
    ax.set_xlabel("Pasos de entorno (timesteps)")
    ax.set_ylabel("Recompensa por episodio (media móvil)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Recompensa vs. Tiempo Real en Segundos (Wall-Clock Efficiency)
    ax = axes[0, 1]
    for res, label, color in [
        (full_res, "Full-Info (MLP ReLU)", "tab:blue"),
        (img_res, "Imágenes Píxeles (CNN Paper DQN)", "tab:orange"),
    ]:
        r_smooth = media_movil(res["ep_rewards"], k=window)
        time_x = res["ep_end_time"][len(res["ep_end_time"]) - len(r_smooth) :]
        ax.plot(time_x, r_smooth, label=label, color=color)
    ax.set_title("Velocidad Real: Recompensa vs. Tiempo de Reloj (segundos)")
    ax.set_xlabel("Tiempo de entrenamiento (segundos)")
    ax.set_ylabel("Recompensa por episodio (media móvil)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) Tasa de Victorias (%) vs. Pasos de Entorno
    ax = axes[1, 0]
    for res, label, color in [
        (full_res, "Full-Info (MLP ReLU)", "tab:blue"),
        (img_res, "Imágenes Píxeles (CNN Paper DQN)", "tab:orange"),
    ]:
        w_smooth = media_movil(res["ep_wins"], k=window) * 100.0
        steps_x = res["ep_end_step"][len(res["ep_end_step"]) - len(w_smooth) :]
        ax.plot(steps_x, w_smooth, label=label, color=color)
    ax.set_title("Evolución de Tasa de Victorias (%) vs. Pasos de Entorno")
    ax.set_xlabel("Pasos de entorno (timesteps)")
    ax.set_ylabel("Tasa de Victorias (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4) Porcentaje de Bolitas Comidas (%) vs. Pasos de Entorno
    ax = axes[1, 1]
    for res, label, color in [
        (full_res, "Full-Info (MLP ReLU)", "tab:blue"),
        (img_res, "Imágenes Píxeles (CNN Paper DQN)", "tab:orange"),
    ]:
        p_smooth = media_movil(res["ep_pellets_pct"], k=window) * 100.0
        steps_x = res["ep_end_step"][len(res["ep_end_step"]) - len(p_smooth) :]
        ax.plot(steps_x, p_smooth, label=label, color=color)
    ax.set_title("Progreso en el Mapa (% Bolitas Comidas) vs. Pasos")
    ax.set_xlabel("Pasos de entorno (timesteps)")
    ax.set_ylabel("Bolitas comidas por episodio (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nGráfico guardado en '{save_path}'")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar cuánto tarda en aprender DQN con Información Completa vs. DQN desde Imágenes (CNN)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=120_000,
        help="Pasos de entrenamiento para cada agente (default: 120000)",
    )
    parser.add_argument(
        "--mlp-layers",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_LAYERS,
        help="Capas ocultas ReLU para el agente Full-Info (default: 256 256 128)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=8,
        help="Tamaño en píxeles de cada celda para el renderizado de imagen CNN (default: 8 -> imagen 72x88)",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Usar imágenes RGB en lugar de escala de grises para la CNN",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Cantidad de frames apilados para la CNN (default: 4, igual al paper de DQN)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Episodios de evaluación al finalizar (default: 100)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir ventana de matplotlib (igual guarda el PNG)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    full_info_results = train_agent_tracked(
        mode="full_info",
        total_timesteps=args.timesteps,
        mlp_layers=args.mlp_layers,
        eval_episodes=args.eval_episodes,
    )

    image_cnn_results = train_agent_tracked(
        mode="image_cnn",
        total_timesteps=args.timesteps,
        tile_size=args.tile_size,
        grayscale=not args.rgb,
        frame_stack=args.frame_stack,
        eval_episodes=args.eval_episodes,
    )

    print_comparison_report(full_info_results, image_cnn_results)
    plot_full_vs_image(
        full_info_results,
        image_cnn_results,
        show_plots=not args.no_show,
    )
