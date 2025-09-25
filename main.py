# -*- coding: utf-8 -*-
"""
Changes:
- Безопасная оценка выражения (eval без __builtins__, разрешены math и numpy)
- Автоматическое определение размерности по максимальному индексу x[i]
- Исправлена вероятность мутации (теперь мутирует с вероятностью mut_rate)
- Полная потокобезопасность: фоновый поток считает, главный обновляет GUI через queue
- Встроенный график прогресса внутри окна (matplotlib, если установлен)
- Ранняя остановка по "patience", выбор максимизации/минимизации
- Инициализация из непрерывных диапазонов (общий min/max)
- Возможность сохранить историю в CSV
- Безопасное назначение иконки
- Улучшен layout: window resize, log & plot вынесены на вкладки
"""

from __future__ import annotations

import re
import math
import random
import threading
import queue
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

MATPLOTLIB_OK = True
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    MATPLOTLIB_OK = False

import numpy as np


# ==========================
# HELPERS
# ==========================

def safe_icon(window: tk.Tk | tk.Toplevel, path: str = "GAE.ico") -> None:
    try:
        window.iconbitmap(path)
    except Exception:
        pass


def detect_dim(expr: str) -> int:
    """finding rank by max x[i] in equation"""
    idx = [int(m) for m in re.findall(r"x\[(\d+)\]", expr)]
    return (max(idx) + 1) if idx else 1


ALLOWED_GLOBALS = {
    "np": np,
    "math": math,
}


def safe_eval(expr: str, x: np.ndarray | list[float]) -> float:
    """safely calculates expr, x — args vector"""
    return eval(expr, {"__builtins__": {}}, {**ALLOWED_GLOBALS, "x": x})


# ==========================
# GA CONFIG AND ALGO
# ==========================

@dataclass
class GAConfig:
    func_expr: str
    pop_size: int
    alpha: float
    mut_rate: float
    mut_dev: float
    n_iter: int
    bounds: Tuple[float, float] = (-500.0, 500.0)
    maximize: bool = True
    patience: int = 0            # 0
    seed: Optional[int] = None


class GeneticOptimizer:
    def __init__(self, cfg: GAConfig):
        self.cfg = cfg
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
        self.dim = detect_dim(cfg.func_expr)

        self.best_hist_i: List[int] = []
        self.best_hist_f: List[float] = []

        self._best_value: Optional[float] = None
        self._no_improve_ctr: int = 0

    # -------- Evaluation --------

    def evaluate(self, x_vec) -> float:
        return float(safe_eval(self.cfg.func_expr, x_vec))

    def fitness_list(self, population: List[List[float]]) -> List[float]:
        return [self.evaluate(p) for p in population]

    # -------- Rank selection --------

    def selection(self, population: List[List[float]]) -> List[Tuple[List[float], List[float]]]:
        scores = np.array(self.fitness_list(population))
        # ranking (1..N)
        ranks = scores.argsort().argsort() + 1 if self.cfg.maximize else (-scores).argsort().argsort() + 1
        p = ranks / ranks.sum()
        idx = np.random.choice(len(population), size=2 * len(population), p=p, replace=True)
        pairs = idx.reshape(len(population), 2)
        return [(population[i], population[j]) for i, j in pairs]

    # -------- BLX-like crossover (interval b1..b2) --------

    def crossover(self, parents: List[Tuple[List[float], List[float]]]) -> List[List[float]]:
        alpha = self.cfg.alpha
        p1 = np.array([a[0] for a in parents], dtype=float)
        p2 = np.array([a[1] for a in parents], dtype=float)
        b1 = p1 - alpha * (p2 - p1)
        b2 = p2 + alpha * (p2 - p1)
        kids = [np.random.uniform(low=b1[i], high=b2[i]).tolist() for i in range(len(parents))]
        return kids

    # -------- Mutation --------

    def mutate(self, population: List[List[float]]) -> List[List[float]]:
        arr = np.array(population, dtype=float)

        mask = np.random.rand(*arr.shape) < self.cfg.mut_rate
        noise = np.random.normal(0.0, self.cfg.mut_dev, size=arr.shape)
        arr = arr + mask * noise

        low, high = self.cfg.bounds
        arr = np.clip(arr, low, high)
        return arr.tolist()

    # -------- Elitism --------

    def elitism(self,
                old_pop: List[List[float]],
                new_pop: List[List[float]],
                scores_old: List[float],
                scores_new: List[float]) -> List[List[float]]:
        maximize = self.cfg.maximize
        best_idx_old = int(np.argmax(scores_old)) if maximize else int(np.argmin(scores_old))
        worst_idx_new = int(np.argmin(scores_new)) if maximize else int(np.argmax(scores_new))
        new_pop[worst_idx_new] = old_pop[best_idx_old]
        return new_pop

    # -------- Init population --------

    def init_population(self) -> List[List[float]]:
        low, high = self.cfg.bounds
        return [np.random.uniform(low, high, size=self.dim).tolist() for _ in range(self.cfg.pop_size)]

    # -------- Main loop --------

    def run(self, progress_cb=None, stop_event: threading.Event | None = None):
        pop = self.init_population()
        scores = self.fitness_list(pop)

        # starting best
        best_idx = int(np.argmax(scores)) if self.cfg.maximize else int(np.argmin(scores))
        best_x = pop[best_idx][:]
        best_f = scores[best_idx]
        self._best_value = best_f

        for it in range(self.cfg.n_iter):
            if stop_event is not None and stop_event.is_set():
                break

            parents = self.selection(pop)
            kids = self.crossover(parents)
            new_pop = self.mutate(kids)

            new_scores = self.fitness_list(new_pop)

            # elitism
            new_pop = self.elitism(pop, new_pop, scores, new_scores)
            scores = self.fitness_list(new_pop)
            pop = new_pop

            # current best
            cur_best_idx = int(np.argmax(scores)) if self.cfg.maximize else int(np.argmin(scores))
            cur_best_x = pop[cur_best_idx][:]
            cur_best_f = scores[cur_best_idx]

            # in history only when gets better
            improved = (cur_best_f > self._best_value) if self.cfg.maximize else (cur_best_f < self._best_value)
            if improved:
                self._best_value = cur_best_f
                best_x = cur_best_x
                best_f = cur_best_f
                self.best_hist_i.append(it)
                self.best_hist_f.append(best_f)
                if progress_cb:
                    progress_cb({
                        "iter": it,
                        "x": np.round(best_x, 6).tolist(),
                        "f": float(best_f),
                        "improved": True
                    })
                self._no_improve_ctr = 0
            else:
                self._no_improve_ctr += 1
                if progress_cb:
                    progress_cb({
                        "iter": it,
                        "x": np.round(cur_best_x, 6).tolist(),
                        "f": float(cur_best_f),
                        "improved": False
                    })

            # early stop
            if self.cfg.patience and self._no_improve_ctr >= self.cfg.patience:
                break

        return {
            "x_best": best_x,
            "f_best": float(best_f),
            "iters_done": it + 1
        }


# ==========================
# GUI
# ==========================

class GAApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Genetic Optimization of Continuous Functions")
        safe_icon(self.root)

        self.root.geometry("1024x720")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)

        self.q: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.track_i: List[int] = []
        self.track_f: List[float] = []

        params_frame = ttk.LabelFrame(self.root, text="Parameters")
        params_frame.pack(fill="x", padx=10, pady=10)

        self.var_func = tk.StringVar(value="-1*x[0]**2 - 100")
        self.var_pop = tk.IntVar(value=100)
        self.var_alpha = tk.DoubleVar(value=0.5)
        self.var_mut_rate = tk.DoubleVar(value=0.05)
        self.var_mut_dev = tk.DoubleVar(value=2.5)
        self.var_iters = tk.IntVar(value=10000)
        self.var_min = tk.DoubleVar(value=-500.0)
        self.var_max = tk.DoubleVar(value=500.0)
        self.var_maximize = tk.BooleanVar(value=True)
        self.var_patience = tk.IntVar(value=0)
        self.var_seed = tk.StringVar(value="")  # None

        row = 0
        ttk.Label(params_frame, text="f(x) =").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.var_func, width=70).grid(row=row, column=1, columnspan=7, sticky="we", padx=5, pady=5)

        row += 1
        ttk.Label(params_frame, text="Population").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(params_frame, textvariable=self.var_pop, width=10).grid(row=row, column=1, sticky="w", padx=5)

        ttk.Label(params_frame, text="Alpha").grid(row=row, column=2, sticky="w")
        ttk.Entry(params_frame, textvariable=self.var_alpha, width=10).grid(row=row, column=3, sticky="w", padx=5)

        ttk.Label(params_frame, text="Mut rate").grid(row=row, column=4, sticky="w")
        ttk.Entry(params_frame, textvariable=self.var_mut_rate, width=10).grid(row=row, column=5, sticky="w", padx=5)

        ttk.Label(params_frame, text="Mut dev").grid(row=row, column=6, sticky="w")
        ttk.Entry(params_frame, textvariable=self.var_mut_dev, width=10).grid(row=row, column=7, sticky="w", padx=5)

        row += 1
        ttk.Label(params_frame, text="Iterations").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(params_frame, textvariable=self.var_iters, width=10).grid(row=row, column=1, sticky="w", padx=5)

        ttk.Label(params_frame, text="Min").grid(row=row, column=2, sticky="w")
        ttk.Entry(params_frame, textvariable=self.var_min, width=10).grid(row=row, column=3, sticky="w", padx=5)

        ttk.Label(params_frame, text="Max").grid(row=row, column=4, sticky="w")
        ttk.Entry(params_frame, textvariable=self.var_max, width=10).grid(row=row, column=5, sticky="w", padx=5)

        ttk.Label(params_frame, text="Patience").grid(row=row, column=6, sticky="w")
        ttk.Entry(params_frame, textvariable=self.var_patience, width=10).grid(row=row, column=7, sticky="w", padx=5)

        row += 1
        ttk.Checkbutton(params_frame, text="Maximize (uncheck → minimize)", variable=self.var_maximize).grid(row=row, column=0, columnspan=3, sticky="w", padx=5)
        ttk.Label(params_frame, text="Seed").grid(row=row, column=3, sticky="e", padx=5)
        ttk.Entry(params_frame, textvariable=self.var_seed, width=12).grid(row=row, column=4, sticky="w", padx=5)

        for c in range(8):
            params_frame.grid_columnconfigure(c, weight=1)

        # Control buttons
        btns = ttk.Frame(self.root)
        btns.pack(fill="x", padx=10, pady=5)
        self.btn_start = ttk.Button(btns, text="START", command=self.on_start)
        self.btn_stop = ttk.Button(btns, text="STOP", command=self.on_stop, state="disabled")
        self.btn_save = ttk.Button(btns, text="Save CSV", command=self.on_save_csv, state="disabled")
        self.btn_start.pack(side="left", padx=5)
        self.btn_stop.pack(side="left", padx=5)
        self.btn_save.pack(side="left", padx=5)

        # INFO panel
        info = ttk.LabelFrame(self.root, text="Status")
        info.pack(fill="x", padx=10, pady=5)
        self.lbl_dim = ttk.Label(info, text="Dim: -")
        self.lbl_func = ttk.Label(info, text="f(x): -")
        self.lbl_bestx = ttk.Label(info, text="x*: -")
        self.lbl_bestf = ttk.Label(info, text="f(x*): -")
        self.lbl_iter = ttk.Label(info, text="Iteration: -")
        self.lbl_dim.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.lbl_func.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.lbl_bestx.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.lbl_bestf.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.lbl_iter.grid(row=3, column=0, sticky="w", padx=5, pady=2)
        info.grid_columnconfigure(1, weight=1)

        # Log/Plot
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Tab: Log
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="Log")
        log_frame = ttk.Frame(log_tab)
        log_frame.pack(fill="both", expand=True)
        self.txt = tk.Text(log_frame, wrap="word")
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=yscroll.set)
        self.txt.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        # Tab: Plot
        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="Plot")
        plot_frame = ttk.Frame(plot_tab)
        plot_frame.pack(fill="both", expand=True)

        if MATPLOTLIB_OK:
            self.fig = Figure(figsize=(7.5, 3.5))
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("GA Progress")
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("Best f(x)")
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self._plot_enabled = True
        else:
            lbl = ttk.Label(plot_frame, text="matplotlib is not installed — plot is unavailable.\nInstall: pip install matplotlib")
            lbl.pack(pady=10)
            self._plot_enabled = False

        self.root.after(50, self.poll_queue)

        self.update_info_dim_func()

    # ---------- Helpers ----------

    def update_info_dim_func(self):
        expr = self.var_func.get().strip()
        try:
            dim = detect_dim(expr)
        except Exception:
            dim = "?"
        self.lbl_dim.config(text=f"Dim: {dim}")
        self.lbl_func.config(text=f"f(x): {expr}")

    def on_start(self):
        try:
            expr = self.var_func.get().strip()
            detect_dim(expr)  # check if parses
            pop_size = int(self.var_pop.get())
            alpha = float(self.var_alpha.get())
            mut_rate = float(self.var_mut_rate.get())
            mut_dev = float(self.var_mut_dev.get())
            n_iter = int(self.var_iters.get())
            low, high = float(self.var_min.get()), float(self.var_max.get())
            maximize = bool(self.var_maximize.get())
            patience = int(self.var_patience.get())
            seed_str = self.var_seed.get().strip()
            seed = int(seed_str) if seed_str else None

            if pop_size <= 1 or n_iter <= 0:
                raise ValueError("Population must be > 1 and iterations > 0.")
            if not (low < high):
                raise ValueError("Bounds error: min < max is required.")
            if alpha < 0:
                raise ValueError("Alpha must be >= 0.")
            if not (0 <= mut_rate <= 1):
                raise ValueError("Mut rate must be in [0, 1].")
            if mut_dev < 0:
                raise ValueError("Mut dev must be >= 0.")

        except Exception as e:
            messagebox.showerror("Parameter error", str(e))
            return

        # Prev clear
        self.txt.delete("1.0", tk.END)
        self.track_i.clear()
        self.track_f.clear()
        if self._plot_enabled:
            self.ax.clear()
            self.ax.set_title("GA Progress")
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("Best f(x)")
            self.canvas.draw_idle()

        # cfg & optimization
        cfg = GAConfig(
            func_expr=expr,
            pop_size=pop_size,
            alpha=alpha,
            mut_rate=mut_rate,
            mut_dev=mut_dev,
            n_iter=n_iter,
            bounds=(low, high),
            maximize=maximize,
            patience=patience,
            seed=seed
        )
        self.optimizer = GeneticOptimizer(cfg)

        # btn block/unblock
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_save.config(state="disabled")
        self.stop_event.clear()

        self.lbl_dim.config(text=f"Dim: {self.optimizer.dim}")
        self.lbl_bestx.config(text="x*: -")
        self.lbl_bestf.config(text="f(x*): -")
        self.lbl_iter.config(text="Iteration: 0")

        # background thread
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def _run_worker(self):
        def progress_cb(msg: dict):
            self.q.put(msg)

        try:
            result = self.optimizer.run(progress_cb=progress_cb, stop_event=self.stop_event)
            self.q.put({"done": True, "result": result})
        except Exception as e:
            self.q.put({"error": str(e)})

    def on_stop(self):
        self.stop_event.set()

    def on_save_csv(self):
        if not self.track_i:
            messagebox.showinfo("Save", "No data to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save progress as CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["iteration", "best_f"])
                for i, fval in zip(self.track_i, self.track_f):
                    w.writerow([i, fval])
            messagebox.showinfo("Saved", f"File saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def poll_queue(self):
        """Периодический опрос очереди сообщений от фонового потока."""
        try:
            while True:
                msg = self.q.get_nowait()

                if "error" in msg:
                    messagebox.showerror("Computation error", msg["error"])
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.btn_save.config(state="disabled")
                    continue

                if msg.get("done"):
                    res = msg["result"]
                    x_best = np.round(res["x_best"], 6).tolist()
                    f_best = res["f_best"]
                    iters_done = res["iters_done"]
                    self.lbl_bestx.config(text=f"x*: {x_best}")
                    self.lbl_bestf.config(text=f"f(x*): {f_best:.6f}")
                    self.lbl_iter.config(text=f"Iteration: {iters_done}")
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.btn_save.config(state="normal")
                    continue

                it = msg["iter"]
                x = msg["x"]
                fval = msg["f"]
                improved = msg["improved"]

                self.lbl_iter.config(text=f"Iteration: {it}")
                self.lbl_bestx.config(text=f"x*: {x}")
                self.lbl_bestf.config(text=f"f(x*): {fval:.6f}")

                if improved:
                    self.txt.insert(tk.END, f"Iter {it}: improved → f(x) = {fval:.6f}, x* = {x}\n")
                    self.txt.see(tk.END)
                    self.track_i.append(it)
                    self.track_f.append(fval)
                    if self._plot_enabled:
                        self.update_plot()

        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.poll_queue)

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.track_i, self.track_f)
        self.ax.set_title("GA Progress")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Best f(x)")
        self.canvas.draw_idle()

    def run(self):
        safe_icon(self.root)
        self.root.mainloop()


# ==========================
# Entry point
# ==========================

if __name__ == "__main__":
    app = GAApp()
    app.run()