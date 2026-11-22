"""
Entrenamiento y evaluación de Pac-Man usando Deep Q-Network (DQN) en PyTorch.

Características principales:
1. **Información completa del entorno (`full_info_obs=True`)**:
   El agente recibe los 8 canales espaciales del mapa (paredes, bolitas, Pac-Man,
   mapa general de fantasmas y 4 canales individuales por política de fantasma:
   chaser, ambusher, patrol, random) más 10 variables escalares de estado
   (tiempo restante, fracción de bolitas restantes, distancia BFS a la bolita más
   cercana, distancia BFS al fantasma más cercano, movimientos válidos desde la
   celda actual y dirección actual de Pac-Man).
2. **Red Neuronal fácilmente modificable (`PacmanQNetwork`)**:
   Por defecto usa una MLP con activación **ReLU** y **2 o 3 capas ocultas**
   (configurable cambiando `HIDDEN_LAYERS`, `ACTIVATION`, `USE_DUELING`, `DROPOUT`
   al inicio de este archivo o pasándolo por parámetro/CLI).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maps import MAPA, MAPA_MULTI_GHOST, TRAIN_MAPS, TEST_MAPS
from env import PacmanEnv
from train_multimap import MultiMapPacmanEnv, MAX_H, MAX_W


# ===========================================================================
# 1. CONFIGURACIÓN DE LA RED NEURONAL (EDITÁ ACÁ PARA PROBAR ARQUITECTURAS)
# ===========================================================================
# Podés cambiar entre 2 capas (ej. [256, 256]), 3 capas (ej. [256, 256, 128])
# o cualquier otra combinación de capas/neuronas y funciones de activación.
DEFAULT_HIDDEN_LAYERS: List[int] = [256, 256, 128]  # 3 capas ocultas (o poné [256, 256] para 2 capas)
DEFAULT_ACTIVATION: str = "relu"                    # Opciones: "relu", "leaky_relu", "elu", "gelu", "tanh"
DEFAULT_USE_DUELING: bool = False                   # Si True, separa cabezal de Valor V(s) y Ventaja A(s,a)
DEFAULT_USE_LAYER_NORM: bool = False                # Si True, agrega LayerNorm después de cada capa lineal
DEFAULT_DROPOUT: float = 0.0                        # Dropout entre capas (ej. 0.1)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class PacmanQNetwork(nn.Module):
    """
    Red Neuronal Q(s, a) modular para Pac-Man.
    Para modificar o experimentar con otra red:
      - Cambiá `DEFAULT_HIDDEN_LAYERS` arriba (ej. 2 capas `[256, 128]` o 3 capas `[256, 256, 128]`),
      - O editá directamente el método `forward()` / las capas en `__init__()`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
        activation: str = DEFAULT_ACTIVATION,
        use_dueling: bool = DEFAULT_USE_DUELING,
        use_layer_norm: bool = DEFAULT_USE_LAYER_NORM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if activation.lower() not in ACTIVATIONS:
            raise ValueError(
                f"Activación '{activation}' no reconocida. Opciones: {list(ACTIVATIONS.keys())}"
            )
        act_cls = ACTIVATIONS[activation.lower()]
        self.use_dueling = use_dueling

        layers: List[nn.Module] = []
        in_features = obs_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        if self.use_dueling:
            self.value_head = nn.Linear(in_features, 1)
            self.advantage_head = nn.Linear(in_features, action_dim)
        else:
            self.q_head = nn.Linear(in_features, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        if self.use_dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return self.q_head(features)


def build_q_network(
    obs_dim: int,
    action_dim: int,
    hidden_layers: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
    activation: str = DEFAULT_ACTIVATION,
    use_dueling: bool = DEFAULT_USE_DUELING,
    use_layer_norm: bool = DEFAULT_USE_LAYER_NORM,
    dropout: float = DEFAULT_DROPOUT,
) -> nn.Module:
    """
    Fábrica de la red neuronal. Si en el futuro querés reemplazar `PacmanQNetwork`
    por tu propia clase `nn.Module` (por ejemplo una CNN), solo tenés que devolverla acá.
    """
    return PacmanQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        use_dueling=use_dueling,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
    )


# ===========================================================================
# 2. CONFIGURACIÓN DEL ENTRENAMIENTO DQN
# ===========================================================================
@dataclass
class DQNConfig:
    seed: int = 0
    max_steps: int = 100
    ghost_chase_prob: float = 0.7
    full_info_obs: bool = True      # Entrega toda la información del entorno a la red
    use_multimap: bool = False      # False = entrena en MAPA; True = entrena en TRAIN_MAPS

    # Arquitectura de red
    hidden_layers: List[int] = field(default_factory=lambda: list(DEFAULT_HIDDEN_LAYERS))
    activation: str = DEFAULT_ACTIVATION
    use_dueling: bool = DEFAULT_USE_DUELING
    use_layer_norm: bool = DEFAULT_USE_LAYER_NORM
    dropout: float = DEFAULT_DROPOUT

    # Hiperparámetros DQN
    total_timesteps: int = 250_000
    learning_rate: float = 3e-4
    buffer_size: int = 50_000
    gamma: float = 0.99
    batch_size: int = 128
    learning_starts: int = 2_000
    train_frequency: int = 4
    target_network_frequency: int = 500
    tau: float = 1.0                # 1.0 = copia dura (hard update)
    use_double_dqn: bool = True     # Double DQN reduce sobreestimación de valores Q
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.4

    # Guardado y evaluación
    save_path: str = "dqn_pacman.pt"
    eval_episodes: int = 100
    show_plots: bool = True
    reward_kwargs: dict = field(default_factory=dict)


# ===========================================================================
# 3. REPLAY BUFFER Y UTILIDADES
# ===========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class ReplayBuffer:
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


def media_movil(x: Sequence[float], k: int = 100) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return arr
    k = min(k, len(arr))
    return np.convolve(arr, np.ones(k) / k, mode="valid")


# ===========================================================================
# 4. CREACIÓN DEL ENTORNO, ENTRENAMIENTO Y EVALUACIÓN
# ===========================================================================
def make_pacman_env(cfg: DQNConfig, mapa: Optional[List[str]] = None) -> PacmanEnv:
    if cfg.use_multimap and mapa is None:
        return MultiMapPacmanEnv(
            TRAIN_MAPS,
            target_h=MAX_H,
            target_w=MAX_W,
            max_steps=cfg.max_steps,
            ghost_chase_prob=cfg.ghost_chase_prob,
            full_info_obs=cfg.full_info_obs,
            **cfg.reward_kwargs,
        )
    chosen_map = mapa if mapa is not None else MAPA
    return PacmanEnv(
        chosen_map,
        max_steps=cfg.max_steps,
        ghost_chase_prob=cfg.ghost_chase_prob,
        full_info_obs=cfg.full_info_obs,
        **cfg.reward_kwargs,
    )


def evaluate_dqn(
    q_net: nn.Module,
    cfg: DQNConfig,
    device: torch.device,
    n_episodes: int = 100,
    epsilon: float = 0.0,
) -> dict:
    env = make_pacman_env(cfg)
    q_net.eval()
    wins = caught = timeouts = 0
    steps_list: List[int] = []
    rewards_list: List[float] = []

    with torch.no_grad():
        for i in range(n_episodes):
            obs, _ = env.reset(seed=1000 + i)
            done = False
            ep_reward = 0.0
            while not done:
                if random.random() < epsilon:
                    action = int(env.action_space.sample())
                else:
                    obs_t = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    q_values = q_net(obs_t)
                    action = int(torch.argmax(q_values, dim=1).item())

                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                done = term or trunc

            wins += int(info["win"])
            caught += int(info["caught"])
            timeouts += int(trunc and not term)
            steps_list.append(env.t)
            rewards_list.append(ep_reward)

    q_net.train()
    print(
        f"Evaluación DQN ({n_episodes} ep, eps={epsilon:.2f}) -> "
        f"Victorias: {wins / n_episodes:.1%} | Capturas: {caught / n_episodes:.1%} | "
        f"Timeouts: {timeouts / n_episodes:.1%} | Recompensa media: {np.mean(rewards_list):.2f} | "
        f"Pasos medios: {np.mean(steps_list):.1f}"
    )
    return {
        "win_rate": wins / n_episodes,
        "caught_rate": caught / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_steps": float(np.mean(steps_list)),
    }


def train_dqn(
    cfg: Optional[DQNConfig] = None,
    custom_network_builder: Optional[Callable[[int, int], nn.Module]] = None,
):
    if cfg is None:
        cfg = DQNConfig()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_pacman_env(cfg)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    print("=" * 70)
    print("Entrenando Pac-Man con DQN (PyTorch)")
    print(f"  Observación completa (full_info_obs): {cfg.full_info_obs} (dim = {obs_dim})")
    print(f"  Capas ocultas: {cfg.hidden_layers} ({len(cfg.hidden_layers)} capas) | Activación: {cfg.activation}")
    print(f"  Double DQN: {cfg.use_double_dqn} | Dueling DQN: {cfg.use_dueling} | Dispositivo: {device}")
    print("=" * 70)

    if custom_network_builder is not None:
        q_network = custom_network_builder(obs_dim, action_dim).to(device)
        target_network = custom_network_builder(obs_dim, action_dim).to(device)
    else:
        q_network = build_q_network(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
            use_dueling=cfg.use_dueling,
            use_layer_norm=cfg.use_layer_norm,
            dropout=cfg.dropout,
        ).to(device)
        target_network = build_q_network(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
            use_dueling=cfg.use_dueling,
            use_layer_norm=cfg.use_layer_norm,
            dropout=cfg.dropout,
        ).to(device)

    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    rb = ReplayBuffer(cfg.buffer_size, obs_dim)

    ep_rewards: List[float] = []
    ep_wins: List[float] = []
    ep_caught: List[float] = []

    obs, _ = env.reset(seed=cfg.seed)
    current_ep_reward = 0.0
    exploration_steps = int(cfg.exploration_fraction * cfg.total_timesteps)

    for step in range(1, cfg.total_timesteps + 1):
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, exploration_steps, step)

        if random.random() < epsilon:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_values = q_network(obs_t)
                action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # En ReplayBuffer guardamos `terminated` real (no `truncated` por límite de tiempo)
        rb.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        current_ep_reward += reward

        # Paso de optimización de la red Q
        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = rb.sample(
                cfg.batch_size, device
            )

            with torch.no_grad():
                if cfg.use_double_dqn:
                    next_actions = q_network(b_next_obs).argmax(dim=1, keepdim=True)
                    next_q_target = target_network(b_next_obs).gather(1, next_actions)
                else:
                    next_q_target = target_network(b_next_obs).max(
                        dim=1, keepdim=True
                    )[0]
                td_target = b_rewards + cfg.gamma * next_q_target * (1.0 - b_dones)

            current_q = q_network(b_obs).gather(1, b_actions)
            loss = F.smooth_l1_loss(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
            optimizer.step()

        # Actualización de la red objetivo (target network)
        if step >= cfg.learning_starts and step % cfg.target_network_frequency == 0:
            for target_param, param in zip(
                target_network.parameters(), q_network.parameters()
            ):
                target_param.data.copy_(
                    cfg.tau * param.data + (1.0 - cfg.tau) * target_param.data
                )

        if done:
            ep_rewards.append(current_ep_reward)
            ep_wins.append(float(info["win"]))
            ep_caught.append(float(info["caught"]))

            if len(ep_rewards) % 200 == 0:
                recent_r = np.mean(ep_rewards[-100:])
                recent_w = np.mean(ep_wins[-100:])
                recent_c = np.mean(ep_caught[-100:])
                print(
                    f"Paso {step:7d}/{cfg.total_timesteps} | Episodio {len(ep_rewards):5d} | "
                    f"Eps: {epsilon:.2f} | Recompensa(100): {recent_r:6.2f} | "
                    f"Victorias(100): {recent_w:5.1%} | Capturas(100): {recent_c:5.1%}"
                )

            obs, _ = env.reset()
            current_ep_reward = 0.0

    if cfg.save_path:
        torch.save(
            {
                "model_state_dict": q_network.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_layers": cfg.hidden_layers,
                "activation": cfg.activation,
                "use_dueling": cfg.use_dueling,
                "use_layer_norm": cfg.use_layer_norm,
                "dropout": cfg.dropout,
                "full_info_obs": cfg.full_info_obs,
            },
            cfg.save_path,
        )
        print(f"\nModelo DQN guardado en '{cfg.save_path}'")

    print("\nEvaluación final del modelo DQN:")
    evaluate_dqn(q_network, cfg, device, n_episodes=cfg.eval_episodes, epsilon=0.0)
    evaluate_dqn(q_network, cfg, device, n_episodes=cfg.eval_episodes, epsilon=0.05)

    if cfg.show_plots:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(media_movil(ep_rewards))
        ax[0].set_title(
            f"DQN ({cfg.hidden_layers}, {cfg.activation}): Recompensa (media móvil)"
        )
        ax[0].set_xlabel("episodios")

        ax[1].plot(media_movil(ep_wins), label="gana")
        ax[1].plot(media_movil(ep_caught), label="lo atrapan")
        ax[1].set_title("DQN: Tasa de victorias / capturas")
        ax[1].set_xlabel("episodios")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    return q_network, {"rewards": ep_rewards, "wins": ep_wins, "caught": ep_caught}


def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(
        description="Entrenar Pac-Man con DQN y red neuronal ReLU de 2 o 3 capas configurable."
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_LAYERS,
        help="Tamaños de las capas ocultas. Ej: --layers 256 256 (2 capas) o --layers 256 256 128 (3 capas)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=DEFAULT_ACTIVATION,
        choices=list(ACTIVATIONS.keys()),
        help="Función de activación de la red neuronal (default: relu)",
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        default=DEFAULT_USE_DUELING,
        help="Activar arquitectura Dueling DQN",
    )
    parser.add_argument(
        "--multimap",
        action="store_true",
        help="Entrenar en todos los mapas de TRAIN_MAPS en lugar del mapa único",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=250_000,
        help="Cantidad total de pasos de entrenamiento",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar ventana de gráficos al finalizar",
    )
    args = parser.parse_args()

    return DQNConfig(
        hidden_layers=list(args.layers),
        activation=args.activation,
        use_dueling=args.dueling,
        use_multimap=args.multimap,
        total_timesteps=args.timesteps,
        show_plots=not args.no_show,
    )


if __name__ == "__main__":
    config = parse_args()
    train_dqn(config)
