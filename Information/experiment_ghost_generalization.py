"""
Experimento: generalización a distintos tipos de fantasma (DQN vs PPO).

Pregunta
--------
Si entrenamos al agente SOLO contra el fantasma "chaser" (el que trae el entorno
por defecto en MAPA), ¿qué tan bien juega cuando el fantasma se comporta distinto?
¿Generaliza igual DQN que PPO?

Diseño (para que la comparación sea justa)
------------------------------------------
- Misma observación para ambos:  full_info_obs=True
- Mismo mapa (MAPA), mismos max_steps / ghost_chase_prob, misma recompensa base.
- Mismo presupuesto de entrenamiento (timesteps) para ambos.
- Entrenamiento: 1 fantasma "chaser" (MAPA tiene una sola 'G', y la política por
  defecto del primer fantasma es "chaser").
- Evaluación: mismos episodios (semillas 5000+i) contra "chaser", "ambusher",
  "patrol" y "random". Ambos algoritmos en modo greedy (PPO deterministic=True,
  DQN epsilon=0).
- N semillas de entrenamiento por algoritmo; se reporta media ± desvío entre semillas.

Métricas: % de episodios en que lo atrapan, % de bolitas comidas, % de victorias
y recompensa media. (Con presupuestos chicos las victorias pueden ser 0%, por eso
se reportan también las bolitas comidas.)

Uso
---
    python experiment_ghost_generalization.py --smoke                      # prueba rápida (~1 min)
    python experiment_ghost_generalization.py --timesteps 20000 --seeds 1 --eval-episodes 20   # medir tiempos
    python experiment_ghost_generalization.py --timesteps 150000 --seeds 3 --eval-episodes 100 # corrida final

Genera `ghost_generalization_results.json` y `ghost_generalization_comparison.png`.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no abre ventanas; solo guarda el PNG
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from maps import MAPA
from env import PacmanEnv
from train_dqn import DQNConfig, train_dqn

# ---------------------------------------------------------------------------
# Configuración común
# ---------------------------------------------------------------------------
MAX_STEPS = 100
GHOST_CHASE_PROB = 0.7
FULL_INFO_OBS = True
EVAL_POLICIES = ["chaser", "ambusher", "patrol", "random"]
EVAL_SEED_BASE = 5000
METRICS = ["caught", "pellets", "win", "reward"]
METRIC_LABELS = {
    "caught": "Capturas (%)",
    "pellets": "Bolitas comidas (%)",
    "win": "Victorias (%)",
    "reward": "Recompensa media",
}


def make_env(ghost_policies=None) -> PacmanEnv:
    return PacmanEnv(
        MAPA,
        max_steps=MAX_STEPS,
        ghost_chase_prob=GHOST_CHASE_PROB,
        ghost_policies=ghost_policies,
        full_info_obs=FULL_INFO_OBS,
    )


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def train_dqn_agent(seed: int, timesteps: int) -> Callable[[np.ndarray], int]:
    """Entrena DQN con la implementación del repo y devuelve una política greedy."""
    cfg = DQNConfig(
        seed=seed,
        max_steps=MAX_STEPS,
        ghost_chase_prob=GHOST_CHASE_PROB,
        full_info_obs=FULL_INFO_OBS,
        total_timesteps=timesteps,
        learning_starts=min(2_000, max(100, timesteps // 10)),
        save_path="",
        show_plots=False,
        eval_episodes=10,  # train_dqn hace una evaluación final propia; la dejamos corta
    )
    q_net, _ = train_dqn(cfg)
    q_net.eval()
    device = next(q_net.parameters()).device

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(torch.argmax(q_net(x), dim=1).item())

    return act


def train_ppo_agent(seed: int, timesteps: int) -> Callable[[np.ndarray], int]:
    """Entrena PPO (SB3) con los mismos hiperparámetros que train.py."""
    vec_env = make_vec_env(make_env, n_envs=8, seed=seed)
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[128, 128]),
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    vec_env.close()

    def act(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


# ---------------------------------------------------------------------------
# Evaluación (idéntica para ambos algoritmos)
# ---------------------------------------------------------------------------
def evaluate(act: Callable[[np.ndarray], int], ghost_policy: str, n_episodes: int) -> Dict[str, float]:
    env = make_env(ghost_policies=ghost_policy)
    total_pellets = env.total_pellets_init
    caught, win, pellets, rewards = [], [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        done, ep_r, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            ep_r += r
            done = term or trunc
        caught.append(float(info["caught"]))
        win.append(float(info["win"]))
        pellets.append(1.0 - info["pellets_left"] / total_pellets)
        rewards.append(ep_r)
    return {
        "caught": 100 * float(np.mean(caught)),
        "pellets": 100 * float(np.mean(pellets)),
        "win": 100 * float(np.mean(win)),
        "reward": float(np.mean(rewards)),
    }


# ---------------------------------------------------------------------------
# Resumen y gráfico
# ---------------------------------------------------------------------------
def summarize(results: dict) -> dict:
    """results[algo][policy] = lista (una por semilla) de dicts de métricas."""
    summary = {}
    for algo, by_policy in results.items():
        summary[algo] = {}
        for pol, runs in by_policy.items():
            summary[algo][pol] = {
                m: (float(np.mean([r[m] for r in runs])), float(np.std([r[m] for r in runs])))
                for m in METRICS
            }
    return summary


def print_table(summary: dict, n_seeds: int, n_eval: int) -> None:
    print("\n" + "=" * 86)
    print(f"RESULTADOS (media ± desvío entre {n_seeds} semilla(s); {n_eval} episodios por evaluación)")
    print("Entrenado solo contra 'chaser'. Evaluado contra cada tipo de fantasma.")
    print("=" * 86)
    for m in METRICS:
        print(f"\n{METRIC_LABELS[m]}")
        print(f"{'':8s}" + "".join(f"{p:>16s}" for p in EVAL_POLICIES))
        for algo in summary:
            row = "".join(
                f"{summary[algo][p][m][0]:>9.1f} ±{summary[algo][p][m][1]:>4.1f}" for p in EVAL_POLICIES
            )
            print(f"{algo:8s}{row}")
    print(
        f"\nOjo: con {n_eval} episodios el error estándar binomial es ~"
        f"{100 * 0.5 / np.sqrt(n_eval):.1f} pp; diferencias menores no son distinguibles."
    )


def plot(summary: dict, path: str, n_seeds: int) -> None:
    colors = {"DQN": "#D81F1F", "PPO": "#1723d3"}  # azul / naranja (validados: CVD ΔE>24)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    x = np.arange(len(EVAL_POLICIES))
    w = 0.36
    for ax, metric, title in zip(
        axes, ["pellets", "caught"], ["Bolitas comidas (%)", "Capturas (% de episodios)"]
    ):
        for k, algo in enumerate(summary):
            means = [summary[algo][p][metric][0] for p in EVAL_POLICIES]
            stds = [summary[algo][p][metric][1] for p in EVAL_POLICIES]
            bars = ax.bar(
                x + (k - 0.5) * w, means, w - 0.03,
                # los % no pasan de [0, 100]: se recortan las barras de error a ese rango
                yerr=[[min(s, m) for m, s in zip(means, stds)],
                      [min(s, 100 - m) for m, s in zip(means, stds)]] if n_seeds > 1 else None,
                color=colors.get(algo, "#888888"), label=algo, capsize=3,
                error_kw=dict(ecolor="#52514e", lw=1),
            )
            for b, v, s in zip(bars, means, stds):
                # el número va por encima del extremo superior de la barra de error
                top = v + (min(s, 100 - v) if n_seeds > 1 else 0.0)
                ax.text(b.get_x() + b.get_width() / 2, top + 1.5, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=8, color="#52514e")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n(entrenamiento)" if p == "chaser" else p for p in EVAL_POLICIES])
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        ax.set_ylim(0, 112)  # espacio extra arriba para barras de error + números
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    # Leyenda a nivel figura, fuera de los ejes (no tapa barras)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=len(labels),
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        f"Generalización a otros fantasmas (entrenado solo vs 'chaser'; {n_seeds} semilla(s), media ± desvío)",
        fontsize=11, x=0.01, y=0.99, ha="left",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Gráfico guardado en {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=150_000, help="pasos de entrenamiento por algoritmo y semilla")
    p.add_argument("--seeds", type=int, default=3, help="semillas de entrenamiento por algoritmo")
    p.add_argument("--eval-episodes", type=int, default=100, help="episodios de evaluación por política de fantasma")
    p.add_argument("--smoke", action="store_true", help="prueba rápida (3000 pasos, 1 semilla, 10 episodios)")
    p.add_argument("--out-prefix", default="ghost_generalization", help="prefijo de los archivos de salida")
    p.add_argument("--plot-only", action="store_true",
                   help="no entrena: regenera tabla y gráfico desde <out-prefix>_results.json")
    args = p.parse_args()

    if args.plot_only:
        with open(f"{args.out_prefix}_results.json") as f:
            saved = json.load(f)
        summary = summarize(saved["raw"])
        n_seeds, n_eval = saved["args"]["seeds"], saved["args"]["eval_episodes"]
        print_table(summary, n_seeds, n_eval)
        plot(summary, f"{args.out_prefix}_comparison.png", n_seeds)
        return
    if args.smoke:
        args.timesteps, args.seeds, args.eval_episodes = 3_000, 1, 10

    trainers = {"DQN": train_dqn_agent, "PPO": train_ppo_agent}
    results: Dict[str, Dict[str, List[dict]]] = {
        a: {pol: [] for pol in EVAL_POLICIES} for a in trainers
    }

    t0 = time.time()
    for seed in range(args.seeds):
        for algo, trainer in trainers.items():
            t1 = time.time()
            print(f"\n>>> {algo} | semilla {seed} | {args.timesteps} pasos")
            act = trainer(seed, args.timesteps)
            print(f"    entrenado en {time.time() - t1:.0f}s; evaluando...")
            for pol in EVAL_POLICIES:
                res = evaluate(act, pol, args.eval_episodes)
                results[algo][pol].append(res)
                print(
                    f"    vs {pol:9s} capturas {res['caught']:5.1f}% | bolitas {res['pellets']:5.1f}% | "
                    f"victorias {res['win']:5.1f}% | R {res['reward']:6.2f}"
                )

    summary = summarize(results)
    print_table(summary, args.seeds, args.eval_episodes)
    with open(f"{args.out_prefix}_results.json", "w") as f:
        json.dump({"args": vars(args), "raw": results, "summary": summary}, f, indent=2)
    plot(summary, f"{args.out_prefix}_comparison.png", args.seeds)
    print(f"\nTiempo total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()