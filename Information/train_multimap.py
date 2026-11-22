"""
Entrenamiento multi-mapa de Pac-Man con PPO (Stable-Baselines3) y comparación
contra el modelo entrenado en un único mapa (`train.py`).

Como `PPO("MlpPolicy")` requiere un `observation_space` de tamaño fijo `(4 * H * W,)`,
el entorno `MultiMapPacmanEnv` normaliza todos los mapas al tamaño máximo `(max_H, max_W)`
rellenando los bordes sobrantes con paredes (`#`). Además permite muestrear un mapa
distinto de `TRAIN_MAPS` en cada `reset()` para que el agente generalice.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from maps import MAPA, TRAIN_MAPS, TEST_MAPS
from env import PacmanEnv
import train as single_map_train

MAX_STEPS = 100
GHOST_CHASE_PROB = 0.7
TOTAL_TIMESTEPS = 400_000
MULTI_MODEL_PATH = "ppo_pacman_multimap"


def get_max_dimensions(map_lists):
    """Obtiene el alto (H) y ancho (W) máximos entre varias listas de mapas."""
    all_maps = [m for ml in map_lists for m in ml]
    max_h = max(len(m) for m in all_maps)
    max_w = max(len(m[0]) for m in all_maps)
    return max_h, max_w


MAX_H, MAX_W = get_max_dimensions([TRAIN_MAPS, TEST_MAPS, [MAPA]])


def pad_map(mapa, target_h=MAX_H, target_w=MAX_W):
    """
    Rellena un mapa con paredes '#' hasta alcanzar `(target_h, target_w)`
    centrando el mapa original dentro de la grilla.
    """
    h = len(mapa)
    w = len(mapa[0])
    if h > target_h or w > target_w:
        raise ValueError(
            f"El mapa ({h}x{w}) excede las dimensiones objetivo ({target_h}x{target_w})."
        )
    if h == target_h and w == target_w:
        return list(mapa)

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    full_wall_row = "#" * target_w
    padded = [full_wall_row for _ in range(pad_top)]
    for row in mapa:
        padded.append(("#" * pad_left) + row + ("#" * pad_right))
    padded.extend(full_wall_row for _ in range(pad_bottom))
    return padded


class MultiMapPacmanEnv(PacmanEnv):
    """
    Extensión de `PacmanEnv` que elige un mapa al azar de `maps_list` en cada episodio
    y estandariza las dimensiones a `(target_h, target_w)` para mantener fija la observación.
    """

    def __init__(
        self,
        maps_list,
        target_h=MAX_H,
        target_w=MAX_W,
        max_steps=MAX_STEPS,
        ghost_chase_prob=GHOST_CHASE_PROB,
        render_mode=None,
        **kwargs,
    ):
        if not maps_list:
            raise ValueError("La lista de mapas no puede estar vacía.")
        self.target_h = target_h
        self.target_w = target_w
        self.padded_maps = [pad_map(m, target_h, target_w) for m in maps_list]
        self.current_map_idx = 0

        super().__init__(
            self.padded_maps[0],
            max_steps=max_steps,
            ghost_chase_prob=ghost_chase_prob,
            render_mode=render_mode,
            **kwargs,
        )

    def _load_map(self, mapa):
        self._parse_and_init_map(mapa)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "map_idx" in options:
            self.current_map_idx = int(options["map_idx"]) % len(self.padded_maps)
        else:
            self.current_map_idx = int(self.np_random.integers(len(self.padded_maps)))

        self._load_map(self.padded_maps[self.current_map_idx])
        self.pac = self.pac_start
        self.pac_dir = (0, 1)
        self.ghosts = list(self.ghost_starts)
        self.prev_ghosts = list(self.ghost_starts)
        self.pellets = self.pellets_init.copy()
        self.t = 0
        return self._obs(), {
            "pellets_left": int(self.pellets.sum()),
            "map_idx": self.current_map_idx,
            "num_ghosts": len(self.ghosts),
            "ghost_policies": list(self.ghost_policies),
        }


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


def make_multimap_train_env(reward_kwargs=None):
    rk = reward_kwargs or {}
    return MultiMapPacmanEnv(
        TRAIN_MAPS,
        target_h=MAX_H,
        target_w=MAX_W,
        max_steps=MAX_STEPS,
        ghost_chase_prob=GHOST_CHASE_PROB,
        **rk,
    )


def train_multimap(
    total_timesteps=TOTAL_TIMESTEPS,
    save_path=MULTI_MODEL_PATH,
    show_plots=True,
    reward_kwargs=None,
    verbose=1,
):
    print(
        f"Entrenando PPO Multi-Mapa con {len(TRAIN_MAPS)} mapas de entrenamiento "
        f"(grilla normalizada a {MAX_H}x{MAX_W})..."
    )
    vec_env = make_vec_env(
        lambda: make_multimap_train_env(reward_kwargs=reward_kwargs),
        n_envs=8,
        seed=0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
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
        verbose=verbose,
        seed=0,
    )

    log = LogCallback()
    model.learn(total_timesteps=total_timesteps, callback=log)

    if save_path:
        model.save(save_path)
        print(f"Modelo multi-mapa guardado en '{save_path}.zip'")

    if show_plots:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(media_movil(log.rewards))
        ax[0].set_title("Multi-Mapa: Recompensa por episodio (media móvil)")
        ax[0].set_xlabel("episodios")

        ax[1].plot(media_movil(log.wins), label="gana")
        ax[1].plot(media_movil(log.caught), label="lo atrapan")
        ax[1].set_title("Multi-Mapa: Tasa de victorias / capturas")
        ax[1].set_xlabel("episodios")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    return model, log


def evaluate_on_map_set(
    model, maps_list, set_name="Mapas", episodes_per_map=50, use_padding=True, deterministic=False
):
    """
    Evalúa un modelo sobre cada mapa de `maps_list`.
    Si `use_padding=False`, solo evalúa aquellos mapas cuyo tamaño coincida con el del modelo.
    """
    expected_dim = model.observation_space.shape[0]
    total_wins = total_caught = total_timeouts = total_eps = 0
    all_steps = []
    all_rewards = []

    print(f"\n--- Evaluación en {set_name} ({len(maps_list)} mapas) ---")
    for idx, m in enumerate(maps_list):
        if use_padding:
            eval_map = pad_map(m, MAX_H, MAX_W)
        else:
            eval_map = m

        h, w = len(eval_map), len(eval_map[0])
        if 4 * h * w != expected_dim:
            print(
                f"  Mapa #{idx + 1} ({len(m)}x{len(m[0])}): omitido "
                f"(dimensión {4 * h * w} incompatible con modelo de {expected_dim})"
            )
            continue

        env = PacmanEnv(
            eval_map, max_steps=MAX_STEPS, ghost_chase_prob=GHOST_CHASE_PROB
        )
        wins = caught = timeouts = 0
        steps, rewards = [], []

        for ep in range(episodes_per_map):
            obs, _ = env.reset(seed=2000 + idx * 100 + ep)
            done = False
            ep_r = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, r, term, trunc, info = env.step(int(action))
                ep_r += r
                done = term or trunc
            wins += int(info["win"])
            caught += int(info["caught"])
            timeouts += int(trunc and not term)
            steps.append(env.t)
            rewards.append(ep_r)

        total_wins += wins
        total_caught += caught
        total_timeouts += timeouts
        total_eps += episodes_per_map
        all_steps.extend(steps)
        all_rewards.extend(rewards)

        print(
            f"  Mapa #{idx + 1:02d} ({len(m)}x{len(m[0])}) -> "
            f"Victorias: {wins / episodes_per_map:5.1%} | "
            f"Capturas: {caught / episodes_per_map:5.1%} | "
            f"Timeouts: {timeouts / episodes_per_map:5.1%} | "
            f"Recompensa media: {np.mean(rewards):6.2f}"
        )

    if total_eps > 0:
        print(
            f"TOTAL {set_name} ({total_eps} episodios) -> "
            f"Victorias: {total_wins / total_eps:5.1%} | "
            f"Capturas: {total_caught / total_eps:5.1%} | "
            f"Timeouts: {total_timeouts / total_eps:5.1%} | "
            f"Recompensa media: {np.mean(all_rewards):6.2f} | "
            f"Pasos medios: {np.mean(all_steps):.1f}"
        )

    return {
        "win_rate": total_wins / total_eps if total_eps else 0.0,
        "caught_rate": total_caught / total_eps if total_eps else 0.0,
        "timeout_rate": total_timeouts / total_eps if total_eps else 0.0,
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
    }


def compare_trainings(total_timesteps=TOTAL_TIMESTEPS, show_plots=True):
    """
    Entrena tanto el modelo de un solo mapa (`train.py`) como el modelo multi-mapa,
    grafica sus curvas lado a lado y compara su desempeño en `TRAIN_MAPS` y `TEST_MAPS`.
    """
    print("=" * 70)
    print("1) ENTRENAMIENTO EN MAPA ÚNICO (train.py)")
    print("=" * 70)
    single_model, single_log = single_map_train.train(
        total_timesteps=total_timesteps, show_plots=False
    )

    print("\n" + "=" * 70)
    print("2) ENTRENAMIENTO EN MÚLTIPLES MAPAS (train_multimap.py)")
    print("=" * 70)
    multi_model, multi_log = train_multimap(
        total_timesteps=total_timesteps, show_plots=False
    )

    # Evaluación comparativa
    print("\n" + "=" * 70)
    print("EVALUACIÓN MODELO MAPA ÚNICO (en mapas compatibles 9x11)")
    print("=" * 70)
    evaluate_on_map_set(
        single_model, TRAIN_MAPS, set_name="TRAIN_MAPS (Single-Map Model)", use_padding=False
    )
    evaluate_on_map_set(
        single_model, TEST_MAPS, set_name="TEST_MAPS (Single-Map Model)", use_padding=False
    )

    print("\n" + "=" * 70)
    print("EVALUACIÓN MODELO MULTI-MAPA (en TODOS los mapas)")
    print("=" * 70)
    evaluate_on_map_set(
        multi_model, TRAIN_MAPS, set_name="TRAIN_MAPS (Multi-Map Model)", use_padding=True
    )
    evaluate_on_map_set(
        multi_model, TEST_MAPS, set_name="TEST_MAPS (Multi-Map Model)", use_padding=True
    )

    # Gráfico comparativo de entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(media_movil(single_log.rewards), label="Mapa Único (train.py)")
    axes[0].plot(media_movil(multi_log.rewards), label=f"Multi-Mapa ({len(TRAIN_MAPS)} mapas)")
    axes[0].set_title("Comparación de Recompensa durante Entrenamiento")
    axes[0].set_xlabel("episodios")
    axes[0].set_ylabel("recompensa (media móvil)")
    axes[0].legend()

    axes[1].plot(media_movil(single_log.wins), label="Victorias - Mapa Único")
    axes[1].plot(media_movil(multi_log.wins), label="Victorias - Multi-Mapa")
    axes[1].set_title("Comparación de Tasa de Victorias durante Entrenamiento")
    axes[1].set_xlabel("episodios")
    axes[1].set_ylabel("tasa de victorias (media móvil)")
    axes[1].legend()

    plt.tight_layout()
    if show_plots:
        plt.show()

    return single_model, multi_model


if __name__ == "__main__":
    compare_trainings()
