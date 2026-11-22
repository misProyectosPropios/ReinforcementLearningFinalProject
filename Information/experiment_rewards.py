"""
Experimentos comparativos de esquemas de recompensas (Reward Shaping) entre:
  1. PPO Mapa Único (`train.py` style)
  2. PPO Multi-Mapa (`train_multimap.py`)
  3. DQN con red ReLU de 2/3 capas e información completa del entorno (`train_dqn.py`)

Evalúa y compara cómo difieren los algoritmos bajo una lista de configuraciones
de recompensas agrupadas en 4 categorías:
  - **Baseline**: Recompensas originales del notebook.
  - **Sparse Rewards (Recompensas dispersas)**:
      * `sparse_terminal`: Solo da recompensa al terminar (+10 si gana, -10 si muere; 0 por bolita/paso/pared).
      * `sparse_pellets_only`: Solo +1 al comer bolita (0 por paso/pared/ganar/morir).
  - **Informative Rewards (Recompensas con más información del sistema)**:
      * `informative_shaping`: Agrega señal continua por acercarse por BFS a la bolita más
        cercana (`r_closer_pellet`), alejarse de fantasmas cercanos (`r_farther_ghost`)
        y bonus progresivo a medida que limpia el mapa (`r_progress_coef`).
  - **Exaggerated Rewards (Recompensas que exageran cierta información)**:
      * `exaggerated_survival`: Castigo extremo por morir (`r_dead=-100.0`) y alto bonus por
        alejarse del fantasma (agente ultra-conservador).
      * `exaggerated_greed`: Recompensa muy alta por bolita (`r_pellet=10.0`) y victoria (`r_win=50.0`),
        con penalización baja por muerte (`r_dead=-2.0`) (agente temerario).
      * `exaggerated_urgency`: Fuerte penalización por cada paso (`r_step=-0.5`) y contra paredes
        (`r_wall=-2.0`), forzando trayectorias mínimas.

Nota clave para comparación justa:
Durante el entrenamiento cada agente recibe la función de recompensa del esquema probado,
pero en la **evaluación final** todos los agentes se miden por su **Tasa de Victorias (%)**,
**Tasa de Capturas (%)**, **Fracción de Bolitas Comidas (%)** y **Recompensa Estándar (Baseline)**,
para comparar su desempeño real en igualdad de condiciones.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from maps import MAPA, TRAIN_MAPS, TEST_MAPS
from env import PacmanEnv
import train as single_ppo
import train_multimap as multi_ppo
import train_dqn as dqn_module


# ===========================================================================
# 1. LISTA DE ESQUEMAS DE RECOMPENSA A EXPERIMENTAR (EDITABLE)
# ===========================================================================
REWARD_SCHEMES: List[Dict] = [
    {
        "name": "baseline",
        "category": "Baseline",
        "description": "Original (+1 bolita, -0.05 paso, -0.1 pared, +10 ganar, -10 morir)",
        "kwargs": {
            "r_pellet": 1.0,
            "r_step": -0.05,
            "r_wall": -0.1,
            "r_win": 10.0,
            "r_dead": -10.0,
            "r_win_remaining_coef": 0.1,
            "r_closer_pellet": 0.0,
            "r_farther_ghost": 0.0,
            "r_progress_coef": 0.0,
        },
    },
    {
        "name": "sparse_terminal",
        "category": "Sparse",
        "description": "Dispersa pura: 0 en todo salvo +10 al ganar o -10 al ser atrapado",
        "kwargs": {
            "r_pellet": 0.0,
            "r_step": 0.0,
            "r_wall": 0.0,
            "r_win": 10.0,
            "r_dead": -10.0,
            "r_win_remaining_coef": 0.0,
            "r_closer_pellet": 0.0,
            "r_farther_ghost": 0.0,
            "r_progress_coef": 0.0,
        },
    },
    {
        "name": "sparse_pellets_only",
        "category": "Sparse",
        "description": "Dispersa por bolitas: solo +1 por bolita, 0 por paso/pared/ganar/morir",
        "kwargs": {
            "r_pellet": 1.0,
            "r_step": 0.0,
            "r_wall": 0.0,
            "r_win": 0.0,
            "r_dead": 0.0,
            "r_win_remaining_coef": 0.0,
            "r_closer_pellet": 0.0,
            "r_farther_ghost": 0.0,
            "r_progress_coef": 0.0,
        },
    },
    {
        "name": "informative_shaping",
        "category": "Informative",
        "description": "Informativa: premia reducir distancia BFS a bolita, alejarse del fantasma y progreso %",
        "kwargs": {
            "r_pellet": 1.5,
            "r_step": -0.03,
            "r_wall": -0.25,
            "r_win": 15.0,
            "r_dead": -15.0,
            "r_win_remaining_coef": 0.15,
            "r_closer_pellet": 0.2,
            "r_farther_ghost": 0.25,
            "r_progress_coef": 2.0,
        },
    },
    {
        "name": "exaggerated_survival",
        "category": "Exaggerated",
        "description": "Exagerada (Supervivencia): castigo masivo por morir (-100) y prioridad a huir",
        "kwargs": {
            "r_pellet": 1.0,
            "r_step": -0.02,
            "r_wall": -0.2,
            "r_win": 10.0,
            "r_dead": -100.0,
            "r_win_remaining_coef": 0.1,
            "r_closer_pellet": 0.05,
            "r_farther_ghost": 0.8,
            "r_progress_coef": 0.0,
        },
    },
    {
        "name": "exaggerated_greed",
        "category": "Exaggerated",
        "description": "Exagerada (Codicia): +10 por bolita y +50 por ganar, bajo castigo por muerte (-2)",
        "kwargs": {
            "r_pellet": 10.0,
            "r_step": -0.05,
            "r_wall": -0.1,
            "r_win": 50.0,
            "r_dead": -2.0,
            "r_win_remaining_coef": 0.5,
            "r_closer_pellet": 0.4,
            "r_farther_ghost": 0.0,
            "r_progress_coef": 5.0,
        },
    },
    {
        "name": "exaggerated_urgency",
        "category": "Exaggerated",
        "description": "Exagerada (Urgencia/Paredes): -0.5 por cada paso y -2.0 por chocar contra pared",
        "kwargs": {
            "r_pellet": 2.0,
            "r_step": -0.5,
            "r_wall": -2.0,
            "r_win": 20.0,
            "r_dead": -15.0,
            "r_win_remaining_coef": 0.3,
            "r_closer_pellet": 0.15,
            "r_farther_ghost": 0.0,
            "r_progress_coef": 0.0,
        },
    },
]


# ===========================================================================
# 2. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN ESTANDARIZADA
# ===========================================================================
def train_single_ppo_with_rewards(reward_kwargs: dict, total_timesteps: int):
    """Entrena el PPO de mapa único con `reward_kwargs` personalizados."""

    def make_env_custom():
        return PacmanEnv(
            MAPA,
            max_steps=single_ppo.MAX_STEPS,
            ghost_chase_prob=single_ppo.GHOST_CHASE_PROB,
            **reward_kwargs,
        )

    vec_env = make_vec_env(make_env_custom, n_envs=8, seed=0)
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
        verbose=0,
        seed=0,
    )
    log = single_ppo.LogCallback()
    model.learn(total_timesteps=total_timesteps, callback=log)
    return model, log


def evaluate_policy_standardized(
    predict_fn,
    eval_env: PacmanEnv,
    n_episodes: int = 100,
) -> Dict[str, float]:
    """
    Evalúa cualquier agente (`predict_fn(obs) -> action`) sobre `eval_env`
    (que usa recompensas Baseline estándar) para comparar manzanas con manzanas.
    """
    wins = caught = timeouts = 0
    standard_rewards: List[float] = []
    pellets_eaten_ratios: List[float] = []
    steps_list: List[int] = []

    for i in range(n_episodes):
        obs, info = eval_env.reset(seed=5000 + i)
        initial_pellets = max(1, info["pellets_left"])
        done = False
        ep_r = 0.0
        while not done:
            action = int(predict_fn(obs))
            obs, r, term, trunc, info = eval_env.step(action)
            ep_r += r
            done = term or trunc

        wins += int(info["win"])
        caught += int(info["caught"])
        timeouts += int(trunc and not term)
        eaten_ratio = (initial_pellets - info["pellets_left"]) / float(initial_pellets)
        pellets_eaten_ratios.append(eaten_ratio)
        standard_rewards.append(ep_r)
        steps_list.append(eval_env.t)

    return {
        "win_rate": wins / n_episodes,
        "caught_rate": caught / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "pellets_eaten_pct": float(np.mean(pellets_eaten_ratios)),
        "standard_reward": float(np.mean(standard_rewards)),
        "mean_steps": float(np.mean(steps_list)),
    }


def run_reward_experiments(
    schemes: List[Dict] = REWARD_SCHEMES,
    ppo_timesteps: int = 80_000,
    dqn_timesteps: int = 60_000,
    dqn_layers: List[int] = None,
    eval_episodes: int = 100,
    include_multimap_ppo: bool = True,
    show_plots: bool = True,
):
    if dqn_layers is None:
        dqn_layers = list(dqn_module.DEFAULT_HIDDEN_LAYERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "PPO_SingleMap": {},
        "PPO_MultiMap": {},
        "DQN_FullInfo": {},
    }

    for idx, scheme in enumerate(schemes, start=1):
        name = scheme["name"]
        cat = scheme["category"]
        desc = scheme["description"]
        r_kwargs = scheme["kwargs"]

        print("\n" + "#" * 78)
        print(f"[{idx}/{len(schemes)}] ESQUEMA: {name} ({cat})")
        print(f"Descripción: {desc}")
        print(f"Parámetros:  {r_kwargs}")
        print("#" * 78)

        # -------------------------------------------------------------------
        # A) PPO Mapa Único (train.py configuration)
        # -------------------------------------------------------------------
        print(f"\n-> (1) Entrenando PPO Mapa Único [{name}] ({ppo_timesteps} pasos)...")
        ppo_model, ppo_log = train_single_ppo_with_rewards(r_kwargs, ppo_timesteps)
        std_env_ppo = PacmanEnv(
            MAPA,
            max_steps=single_ppo.MAX_STEPS,
            ghost_chase_prob=single_ppo.GHOST_CHASE_PROB,
            full_info_obs=False,
        )
        ppo_metrics = evaluate_policy_standardized(
            lambda obs, m=ppo_model: m.predict(obs, deterministic=False)[0],
            std_env_ppo,
            n_episodes=eval_episodes,
        )
        ppo_metrics["train_rewards"] = ppo_log.rewards
        ppo_metrics["train_wins"] = ppo_log.wins
        results["PPO_SingleMap"][name] = ppo_metrics
        print(
            f"   [PPO Mapa Único | {name}] Victorias: {ppo_metrics['win_rate']:5.1%} | "
            f"Capturas: {ppo_metrics['caught_rate']:5.1%} | "
            f"Bolitas comidas: {ppo_metrics['pellets_eaten_pct']:5.1%} | "
            f"Recompensa Estándar: {ppo_metrics['standard_reward']:6.2f}"
        )

        # -------------------------------------------------------------------
        # B) PPO Multi-Mapa (train_multimap.py)
        # -------------------------------------------------------------------
        if include_multimap_ppo:
            print(f"\n-> (2) Entrenando PPO Multi-Mapa [{name}] ({ppo_timesteps} pasos)...")
            m_ppo_model, m_ppo_log = multi_ppo.train_multimap(
                total_timesteps=ppo_timesteps,
                save_path=None,
                show_plots=False,
                reward_kwargs=r_kwargs,
                verbose=0,
            )
            std_env_multimap = multi_ppo.MultiMapPacmanEnv(
                TRAIN_MAPS,
                target_h=multi_ppo.MAX_H,
                target_w=multi_ppo.MAX_W,
                max_steps=multi_ppo.MAX_STEPS,
                ghost_chase_prob=multi_ppo.GHOST_CHASE_PROB,
                full_info_obs=False,
            )
            m_ppo_metrics = evaluate_policy_standardized(
                lambda obs, m=m_ppo_model: m.predict(obs, deterministic=False)[0],
                std_env_multimap,
                n_episodes=eval_episodes,
            )
            m_ppo_metrics["train_rewards"] = m_ppo_log.rewards
            m_ppo_metrics["train_wins"] = m_ppo_log.wins
            results["PPO_MultiMap"][name] = m_ppo_metrics
            print(
                f"   [PPO Multi-Mapa | {name}] Victorias: {m_ppo_metrics['win_rate']:5.1%} | "
                f"Capturas: {m_ppo_metrics['caught_rate']:5.1%} | "
                f"Bolitas comidas: {m_ppo_metrics['pellets_eaten_pct']:5.1%} | "
                f"Recompensa Estándar: {m_ppo_metrics['standard_reward']:6.2f}"
            )

        # -------------------------------------------------------------------
        # C) DQN con Red ReLU e Información Completa (train_dqn.py)
        # -------------------------------------------------------------------
        print(f"\n-> (3) Entrenando DQN Full-Info ({dqn_layers} ReLU) [{name}] ({dqn_timesteps} pasos)...")
        dqn_cfg = dqn_module.DQNConfig(
            total_timesteps=dqn_timesteps,
            hidden_layers=dqn_layers,
            activation="relu",
            full_info_obs=True,
            save_path="",
            eval_episodes=eval_episodes,
            show_plots=False,
            reward_kwargs=r_kwargs,
        )
        q_net, dqn_log = dqn_module.train_dqn(dqn_cfg)
        q_net.eval()

        std_env_dqn = PacmanEnv(
            MAPA,
            max_steps=dqn_cfg.max_steps,
            ghost_chase_prob=dqn_cfg.ghost_chase_prob,
            full_info_obs=True,
        )

        def dqn_predict(obs, net=q_net, env_ref=std_env_dqn):
            if np.random.random() < 0.02:
                return int(env_ref.action_space.sample())
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                return int(torch.argmax(net(obs_t), dim=1).item())

        dqn_metrics = evaluate_policy_standardized(
            dqn_predict,
            std_env_dqn,
            n_episodes=eval_episodes,
        )
        dqn_metrics["train_rewards"] = dqn_log["rewards"]
        dqn_metrics["train_wins"] = dqn_log["wins"]
        results["DQN_FullInfo"][name] = dqn_metrics
        print(
            f"   [DQN Full-Info  | {name}] Victorias: {dqn_metrics['win_rate']:5.1%} | "
            f"Capturas: {dqn_metrics['caught_rate']:5.1%} | "
            f"Bolitas comidas: {dqn_metrics['pellets_eaten_pct']:5.1%} | "
            f"Recompensa Estándar: {dqn_metrics['standard_reward']:6.2f}"
        )

    print_summary_table(schemes, results, include_multimap_ppo)
    plot_reward_comparisons(schemes, results, include_multimap_ppo, show_plots=show_plots)
    return results


def print_summary_table(schemes: List[Dict], results: dict, include_multimap_ppo: bool):
    print("\n" + "=" * 105)
    print("RESUMEN COMPARATIVO DE RECOMPENSAS (Métricas estandarizadas sobre entorno Baseline)")
    print("=" * 105)
    header = (
        f"{'Esquema de Recompensa':<24} | {'Categoría':<11} | "
        f"{'Algoritmo':<15} | {'Victorias':>9} | {'Capturas':>9} | "
        f"{'% Bolitas':>9} | {'Recomp. Estándar':>16}"
    )
    print(header)
    print("-" * 105)

    algos = ["PPO_SingleMap", "DQN_FullInfo"]
    if include_multimap_ppo:
        algos.insert(1, "PPO_MultiMap")

    for scheme in schemes:
        s_name = scheme["name"]
        s_cat = scheme["category"]
        for algo in algos:
            if s_name not in results[algo]:
                continue
            m = results[algo][s_name]
            print(
                f"{s_name:<24} | {s_cat:<11} | {algo:<15} | "
                f"{m['win_rate']:>8.1%} | {m['caught_rate']:>8.1%} | "
                f"{m['pellets_eaten_pct']:>8.1%} | {m['standard_reward']:>16.2f}"
            )
        print("-" * 105)


def plot_reward_comparisons(
    schemes: List[Dict],
    results: dict,
    include_multimap_ppo: bool,
    show_plots: bool = True,
    save_fig_path: str = "reward_schemes_comparison.png",
):
    scheme_names = [s["name"] for s in schemes]
    x = np.arange(len(scheme_names))

    algos = ["PPO_SingleMap", "DQN_FullInfo"]
    if include_multimap_ppo:
        algos.insert(1, "PPO_MultiMap")

    width = 0.8 / len(algos)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1) Gráfico de barras: Tasa de Victorias (%)
    ax = axes[0, 0]
    for i, algo in enumerate(algos):
        win_rates = [results[algo][s]["win_rate"] * 100.0 for s in scheme_names]
        ax.bar(x + (i - (len(algos) - 1) / 2) * width, win_rates, width, label=algo)
    ax.set_title("Tasa de Victorias (%) por Esquema de Recompensa")
    ax.set_xticks(x)
    ax.set_xticklabels(scheme_names, rotation=25, ha="right")
    ax.set_ylabel("Victorias (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2) Gráfico de barras: Porcentaje de Bolitas Comidas (%)
    ax = axes[0, 1]
    for i, algo in enumerate(algos):
        pellet_pcts = [
            results[algo][s]["pellets_eaten_pct"] * 100.0 for s in scheme_names
        ]
        ax.bar(x + (i - (len(algos) - 1) / 2) * width, pellet_pcts, width, label=algo)
    ax.set_title("Porcentaje Medio de Bolitas Comidas (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(scheme_names, rotation=25, ha="right")
    ax.set_ylabel("Bolitas comidas (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3) Curvas de aprendizaje (Tasa de victorias móvil) en PPO Mapa Único
    ax = axes[1, 0]
    for s in scheme_names:
        wins_curve = single_ppo.media_movil(
            results["PPO_SingleMap"][s]["train_wins"], k=50
        )
        ax.plot(wins_curve, label=s)
    ax.set_title("PPO Mapa Único: Evolución de Victorias durante Entrenamiento")
    ax.set_xlabel("episodios")
    ax.set_ylabel("tasa de victorias (media móvil)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4) Curvas de aprendizaje (Tasa de victorias móvil) en DQN Full-Info
    ax = axes[1, 1]
    for s in scheme_names:
        wins_curve = dqn_module.media_movil(
            results["DQN_FullInfo"][s]["train_wins"], k=50
        )
        ax.plot(wins_curve, label=s)
    ax.set_title("DQN Full-Info: Evolución de Victorias durante Entrenamiento")
    ax.set_xlabel("episodios")
    ax.set_ylabel("tasa de victorias (media móvil)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=150)
        print(f"\nGráfico comparativo guardado en '{save_fig_path}'")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar esquemas de recompensas (Sparse, Informative, Exaggerated) en PPO y DQN."
    )
    parser.add_argument(
        "--ppo-timesteps",
        type=int,
        default=80_000,
        help="Pasos de entrenamiento por esquema para PPO (default: 80000)",
    )
    parser.add_argument(
        "--dqn-timesteps",
        type=int,
        default=60_000,
        help="Pasos de entrenamiento por esquema para DQN (default: 60000)",
    )
    parser.add_argument(
        "--dqn-layers",
        type=int,
        nargs="+",
        default=dqn_module.DEFAULT_HIDDEN_LAYERS,
        help="Capas ocultas ReLU para DQN (ej: --dqn-layers 256 256 o 256 256 128)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Episodios de evaluación estandarizada por esquema (default: 100)",
    )
    parser.add_argument(
        "--skip-multimap",
        action="store_true",
        help="Omitir PPO Multi-Mapa para acelerar la comparación entre PPO y DQN en mapa único",
    )
    parser.add_argument(
        "--schemes",
        type=str,
        nargs="+",
        default=None,
        help="Filtrar qué esquemas correr por nombre (ej: --schemes baseline sparse_terminal informative_shaping)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir ventana interactiva de matplotlib (igual guarda el PNG)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_schemes = REWARD_SCHEMES
    if args.schemes:
        selected_schemes = [s for s in REWARD_SCHEMES if s["name"] in args.schemes]
        if not selected_schemes:
            raise ValueError(
                f"Ninguno de los esquemas {args.schemes} coincide con {[s['name'] for s in REWARD_SCHEMES]}"
            )

    run_reward_experiments(
        schemes=selected_schemes,
        ppo_timesteps=args.ppo_timesteps,
        dqn_timesteps=args.dqn_timesteps,
        dqn_layers=list(args.dqn_layers),
        eval_episodes=args.eval_episodes,
        include_multimap_ppo=not args.skip_multimap,
        show_plots=not args.no_show,
    )
