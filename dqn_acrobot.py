"""
Implementación completa de Deep Q-Network (DQN) para Gymnasium Acrobot-v1.

Basada en:
- CleanRL DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
- Hiperparámetros de referencia: https://huggingface.co/cleanrl/Acrobot-v1-dqn-seed1

Diferencias clave pensadas para experimentación y claridad:
- Bucle principal explícito por episodios (en lugar de un único bucle plano de pasos).
- ReplayBuffer propio con NumPy (sin dependencias externas como stable-baselines3).
- Gráficos de recompensa por episodio y agregada cada 20 episodios (+ evaluación greedy).
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Usar backend sin ventana si se ejecuta sin display o con --no-show
if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Configuración e hiperparámetros (basados en cleanrl/Acrobot-v1-dqn-seed1)
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    env_id: str = "Acrobot-v1"
    seed: int = 1

    # Iteración por episodios
    num_episodes: int = 600
    eval_every: int = 20          # Cada cuántos episodios resumir / evaluar / agregar en gráfico
    eval_episodes: int = 5        # Cantidad de episodios greedy por evaluación

    # Hiperparámetros de CleanRL (Acrobot-v1-dqn-seed1)
    learning_rate: float = 2.5e-4
    buffer_size: int = 10_000
    gamma: float = 0.99
    tau: float = 1.0              # 1.0 = copia dura (hard update); ej. 0.005 = soft update
    target_network_frequency: int = 500  # En pasos de entorno (env steps)
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5    # Fracción de episodios durante la cual decae epsilon
    learning_starts: int = 10_000        # Pasos iniciales de exploración antes de entrenar
    train_frequency: int = 10            # 1 actualización de gradiente cada N pasos de entorno

    # Opciones de ejecución y salida
    use_cuda: bool = True
    save_model_path: str = "dqn_acrobot.pt"
    plot_path: str = "dqn_acrobot_rewards.png"
    show_plot: bool = True


# ---------------------------------------------------------------------------
# 2. Utilidades: semillas y schedule de epsilon
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Decaimiento lineal de epsilon estilo CleanRL."""
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# ---------------------------------------------------------------------------
# 3. Replay Buffer (Experience Replay)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Buffer circular eficiente en NumPy para almacenar transiciones (s, a, r, s', done)."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
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
        # Importante: guardamos `terminated` (no `truncated`) para que el límite
        # de 500 pasos de Acrobot-v1 siga haciendo bootstrapping correctamente.
        self.dones[self.ptr] = float(terminated)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_obs[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# 4. Red Q (arquitectura exacta de CleanRL dqn.py: obs -> 120 -> 84 -> actions)
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# 5. Evaluación greedy (sin exploración, epsilon = 0)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(
    q_network: QNetwork,
    env_id: str,
    device: torch.device,
    n_episodes: int = 5,
    seed: int = 1000,
) -> Tuple[float, float]:
    eval_env = gym.make(env_id)
    returns: List[float] = []

    for i in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + i)
        done = False
        ep_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(torch.argmax(q_network(obs_t), dim=1).item())
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        returns.append(ep_reward)

    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ---------------------------------------------------------------------------
# 6. Entrenamiento DQN iterando episodio por episodio
# ---------------------------------------------------------------------------
def train_dqn(cfg: DQNConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")

    env = gym.make(cfg.env_id)
    obs_dim = int(np.array(env.observation_space.shape).prod())
    n_actions = int(env.action_space.n)

    q_network = QNetwork(obs_dim, n_actions).to(device)
    target_network = QNetwork(obs_dim, n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    replay_buffer = ReplayBuffer(cfg.buffer_size, obs_dim)

    # Historiales para graficar
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_epsilons: List[float] = []
    episode_losses: List[float] = []

    block_episodes: List[int] = []
    block_mean_rewards: List[float] = []
    eval_mean_rewards: List[float] = []
    eval_std_rewards: List[float] = []

    global_step = 0
    exploration_episodes = max(1, int(cfg.exploration_fraction * cfg.num_episodes))

    print(f"Entrenando DQN en {cfg.env_id} | dispositivo: {device} | episodios: {cfg.num_episodes}")
    print(
        f"Hiperparámetros CleanRL: lr={cfg.learning_rate}, gamma={cfg.gamma}, "
        f"buffer={cfg.buffer_size}, batch={cfg.batch_size}, "
        f"learning_starts={cfg.learning_starts}, train_freq={cfg.train_frequency}, "
        f"target_freq={cfg.target_network_frequency}, tau={cfg.tau}"
    )
    print("-" * 80)

    for episode in range(1, cfg.num_episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + episode)
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_losses: List[float] = []

        # Epsilon decae linealmente con el número de episodio (fácil de controlar en bucle por episodios)
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, exploration_episodes, episode - 1)

        while not done:
            global_step += 1
            ep_length += 1

            # Política epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_network(obs_t)
                    action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(obs, action, float(reward), next_obs, terminated)
            obs = next_obs
            ep_reward += float(reward)

            # Paso de entrenamiento (idéntico a CleanRL: tras learning_starts, cada train_frequency pasos)
            if global_step >= cfg.learning_starts and len(replay_buffer) >= cfg.batch_size:
                if global_step % cfg.train_frequency == 0:
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = replay_buffer.sample(
                        cfg.batch_size, device
                    )

                    with torch.no_grad():
                        target_max, _ = target_network(b_next_obs).max(dim=1, keepdim=True)
                        td_target = b_rewards + cfg.gamma * target_max * (1.0 - b_dones)

                    old_val = q_network(b_obs).gather(1, b_actions)
                    loss = F.mse_loss(old_val, td_target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ep_losses.append(float(loss.item()))

                # Actualización de la red objetivo (hard update si tau=1.0, soft si tau<1.0)
                if global_step % cfg.target_network_frequency == 0:
                    for target_param, q_param in zip(
                        target_network.parameters(), q_network.parameters()
                    ):
                        target_param.data.copy_(
                            cfg.tau * q_param.data + (1.0 - cfg.tau) * target_param.data
                        )

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_epsilons.append(epsilon)
        episode_losses.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))

        # Resumen y evaluación cada `eval_every` episodios (por defecto: cada 20 episodios)
        if episode % cfg.eval_every == 0:
            recent_mean = float(np.mean(episode_rewards[-cfg.eval_every :]))
            eval_mean, eval_std = evaluate_policy(
                q_network,
                cfg.env_id,
                device,
                n_episodes=cfg.eval_episodes,
                seed=cfg.seed + 10_000 + episode,
            )

            block_episodes.append(episode)
            block_mean_rewards.append(recent_mean)
            eval_mean_rewards.append(eval_mean)
            eval_std_rewards.append(eval_std)

            mean_loss = np.nanmean(episode_losses[-cfg.eval_every :])
            loss_str = f"{mean_loss:.4f}" if not np.isnan(mean_loss) else "warming-up"
            print(
                f"Ep {episode:4d}/{cfg.num_episodes} | Steps: {global_step:7d} | "
                f"Eps: {epsilon:.3f} | Train({cfg.eval_every}ep): {recent_mean:7.2f} | "
                f"Eval(greedy): {eval_mean:7.2f} ± {eval_std:5.2f} | Loss: {loss_str}"
            )

    env.close()

    if cfg.save_model_path:
        torch.save(q_network.state_dict(), cfg.save_model_path)
        print(f"\nModelo guardado en: {cfg.save_model_path}")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_epsilons": episode_epsilons,
        "episode_losses": episode_losses,
        "block_episodes": block_episodes,
        "block_mean_rewards": block_mean_rewards,
        "eval_mean_rewards": eval_mean_rewards,
        "eval_std_rewards": eval_std_rewards,
        "total_steps": global_step,
    }


# ---------------------------------------------------------------------------
# 7. Gráficos de recompensas (por episodio y cada 20 episodios)
# ---------------------------------------------------------------------------
def plot_training_results(history: dict, cfg: DQNConfig) -> None:
    rewards = np.array(history["episode_rewards"], dtype=np.float32)
    episodes = np.arange(1, len(rewards) + 1)
    epsilons = np.array(history["episode_epsilons"], dtype=np.float32)
    losses = np.array(history["episode_losses"], dtype=np.float32)

    block_ep = np.array(history["block_episodes"])
    block_mean = np.array(history["block_mean_rewards"])
    eval_mean = np.array(history["eval_mean_rewards"])
    eval_std = np.array(history["eval_std_rewards"])

    window = cfg.eval_every
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ma_episodes = episodes[window - 1 :]
    else:
        moving_avg = rewards
        ma_episodes = episodes

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"DQN en {cfg.env_id} (seed={cfg.seed})", fontsize=14, fontweight="bold")

    # Panel 1: Recompensa después de CADA episodio + media móvil
    ax1 = axes[0]
    ax1.plot(episodes, rewards, alpha=0.35, color="#1f77b4", label="Recompensa por episodio")
    ax1.plot(
        ma_episodes,
        moving_avg,
        color="#d62728",
        linewidth=2.0,
        label=f"Media móvil ({window} episodios)",
    )
    ax1.axhline(-100.0, color="green", linestyle="--", linewidth=1.2, label="Referencia convergencia (-100)")
    ax1.set_ylabel("Recompensa")
    ax1.set_title("Recompensa obtenida después de cada episodio")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    # Panel 2: Recompensa cada 20 episodios (promedio por bloque + evaluación greedy)
    ax2 = axes[1]
    if len(block_ep) > 0:
        ax2.plot(
            block_ep,
            block_mean,
            marker="o",
            linewidth=2.0,
            color="#ff7f0e",
            label=f"Promedio entrenamiento (cada {window} ep.)",
        )
        ax2.plot(
            block_ep,
            eval_mean,
            marker="s",
            linewidth=2.0,
            color="#2ca02c",
            label=f"Evaluación greedy cada {window} ep. (ε=0)",
        )
        ax2.fill_between(
            block_ep,
            eval_mean - eval_std,
            eval_mean + eval_std,
            color="#2ca02c",
            alpha=0.18,
        )
    ax2.axhline(-100.0, color="green", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Recompensa")
    ax2.set_title(f"Recompensa cada {window} episodios (entrenamiento vs. evaluación greedy)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    # Panel 3: Epsilon y pérdida TD (MSE)
    ax3 = axes[2]
    color_eps = "#9467bd"
    ax3.plot(episodes, epsilons, color=color_eps, linewidth=2.0, label="Epsilon (ε)")
    ax3.set_xlabel("Episodio")
    ax3.set_ylabel("Epsilon", color=color_eps)
    ax3.tick_params(axis="y", labelcolor=color_eps)
    ax3.grid(True, alpha=0.3)

    ax3_twin = ax3.twinx()
    color_loss = "#8c564b"
    valid_mask = ~np.isnan(losses)
    if np.any(valid_mask):
        ax3_twin.plot(
            episodes[valid_mask],
            losses[valid_mask],
            color=color_loss,
            alpha=0.6,
            linewidth=1.3,
            label="TD Loss (MSE)",
        )
    ax3_twin.set_ylabel("TD Loss (MSE)", color=color_loss)
    ax3_twin.tick_params(axis="y", labelcolor=color_loss)
    ax3.set_title("Exploración (ε) y pérdida de entrenamiento (MSE)")

    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    if cfg.plot_path:
        plt.savefig(cfg.plot_path, dpi=150)
        print(f"Gráfico guardado en: {cfg.plot_path}")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(description="DQN completo para Gymnasium Acrobot-v1")
    parser.add_argument("--episodes", type=int, default=600, help="Cantidad de episodios a entrenar")
    parser.add_argument("--eval-every", type=int, default=20, help="Frecuencia de resumen/evaluación/plot en episodios")
    parser.add_argument("--seed", type=int, default=1, help="Semilla aleatoria")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate de Adam")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento")
    parser.add_argument("--buffer-size", type=int, default=10_000, help="Capacidad del Replay Buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño de minibatch")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="Pasos iniciales antes de entrenar")
    parser.add_argument("--train-frequency", type=int, default=10, help="Pasos de entorno entre actualizaciones")
    parser.add_argument("--target-freq", type=int, default=500, help="Pasos entre copias a la target network")
    parser.add_argument("--tau", type=float, default=1.0, help="Coeficiente de actualización de la target network")
    parser.add_argument("--exploration-fraction", type=float, default=0.5, help="Fracción de episodios de decaimiento de epsilon")
    parser.add_argument("--plot-path", type=str, default="dqn_acrobot_rewards.png", help="Ruta para guardar el gráfico")
    parser.add_argument("--model-path", type=str, default="dqn_acrobot.pt", help="Ruta para guardar los pesos")
    parser.add_argument("--no-show", action="store_true", help="No abrir ventana interactiva de matplotlib")
    args = parser.parse_args()

    return DQNConfig(
        num_episodes=args.episodes,
        eval_every=args.eval_every,
        seed=args.seed,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_frequency=args.train_frequency,
        target_network_frequency=args.target_freq,
        tau=args.tau,
        exploration_fraction=args.exploration_fraction,
        plot_path=args.plot_path,
        save_model_path=args.model_path,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    config = parse_args()
    history = train_dqn(config)
    plot_training_results(history, config)
