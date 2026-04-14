#!/usr/bin/env python3
"""
==========================================================================
Notch-Delta-ASC Lateral Inhibition Simulator  v2.0
==========================================================================
Stochastic (SDE / Euler-Maruyama) hexagonal lattice model of neuroblast
selection via Notch-Delta signaling with ASC proneural gene regulation.

ODE System per cell i:
  dD  = [lD·H+(ASC,K_ASC,nd) - dD·D - fD·D·Σj(Nj)]dt + σD·√D·dW
  dN  = [lN - dN·N - fN·N·Σj(Dj) - a·N/(b·D+N)]dt    + σN·√N·dW
  dA  = [-dA·A + fN·N·Σj(Dj) + a·N/(b·D+N)]dt          + σA·√A·dW
  dASC= [-dASC·ASC + lASC·H-(A,Kd,nASC)]dt               + σASC·√ASC·dW

Features: Stochastic SDE, Cell tracking, Phase diagram, Monte Carlo,
          Bifurcation, Noise sweep, tqdm, GPU support
==========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import os, time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm, trange

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def get_xp(use_gpu=False):
    return cp if (use_gpu and GPU_AVAILABLE) else np

# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Parameters:
    rows: int = 7;  cols: int = 7
    # Delta
    lambda_D: float = 5.0;  K_ASC: float = 0.3;  n_d: float = 5.0
    d_D: float = 0.2;  f_D: float = 0.05
    # Notch
    lambda_N: float = 1.5;  d_N: float = 0.2;  f_N: float = 0.8
    a_cis: float = 0.1;  b_cis: float = 2.0
    # Active Notch
    d_A: float = 0.3
    # ASC
    lambda_ASC: float = 5.0;  d_ASC: float = 0.15
    K_d: float = 0.5;  n_ASC: float = 5.0
    # Simulation
    t_end: float = 200.0;  dt: float = 0.01;  noise_sigma_ic: float = 0.15
    # Stochastic SDE
    stochastic: bool = True;  noise_type: str = "multiplicative"
    sigma_D: float = 0.05;  sigma_N: float = 0.05
    sigma_A: float = 0.03;  sigma_ASC: float = 0.05
    # Monte Carlo
    mc_runs: int = 100;  neuroblast_threshold: float = 5.0
    # Cell tracking
    track_cells: str = ""
    # GPU / Output
    use_gpu: bool = False;  output_dir: str = "results"
    save_prefix: str = "notch_delta"

    def get_tracked_cells(self) -> List[int]:
        if not self.track_cells: return []
        return [int(x.strip()) for x in self.track_cells.split(',') if x.strip()]

    def to_dict(self): return asdict(self)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Notch-Delta-ASC Simulator Parameters  v2.0\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            sections = {
                'Lattice': ['rows', 'cols'],
                'Delta': ['lambda_D', 'K_ASC', 'n_d', 'd_D', 'f_D'],
                'Notch': ['lambda_N', 'd_N', 'f_N', 'a_cis', 'b_cis'],
                'Active Notch': ['d_A'],
                'ASC': ['lambda_ASC', 'd_ASC', 'K_d', 'n_ASC'],
                'Simulation': ['t_end', 'dt', 'noise_sigma_ic'],
                'Stochastic': ['stochastic', 'noise_type',
                               'sigma_D', 'sigma_N', 'sigma_A', 'sigma_ASC'],
                'Monte Carlo': ['mc_runs', 'neuroblast_threshold'],
                'Tracking': ['track_cells'],
                'System': ['use_gpu', 'output_dir', 'save_prefix'],
            }
            d = self.to_dict()
            for sec, keys in sections.items():
                f.write(f"[{sec}]\n")
                for k in keys:
                    f.write(f"  {k} = {d[k]}\n")
                f.write("\n")

    @classmethod
    def load(cls, filepath):
        params = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith(('=', '#', '[')):
                    key, val = line.split('=', 1)
                    key, val = key.strip(), val.strip()
                    if hasattr(cls, key):
                        ft = type(getattr(cls(), key))
                        if ft == bool: params[key] = val.lower() in ('true','1','yes')
                        else: params[key] = ft(val)
        return cls(**params)

# ══════════════════════════════════════════════════════════════════════════
class HexLattice:
    def __init__(self, rows, cols, periodic=True):
        self.rows, self.cols = rows, cols
        self.n_cells = rows * cols
        self.periodic = periodic
        self._build_adjacency()
        self._compute_positions()

    def _build_adjacency(self):
        self.adj = [[] for _ in range(self.n_cells)]
        even_off = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1)]
        odd_off  = [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,1)]
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                for dr, dc in (even_off if c%2==0 else odd_off):
                    nr, nc = r+dr, c+dc
                    if self.periodic: nr, nc = nr%self.rows, nc%self.cols
                    if 0<=nr<self.rows and 0<=nc<self.cols:
                        nidx = nr*self.cols+nc
                        if nidx not in self.adj[idx]: self.adj[idx].append(nidx)

    def _compute_positions(self):
        self.positions = np.zeros((self.n_cells, 2))
        h = np.sqrt(3)/2
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r*self.cols+c
                x = c*0.75
                y = r*h + (h/2 if c%2==1 else 0)
                self.positions[idx] = [x, y]

    def adjacency_matrix(self):
        A = np.zeros((self.n_cells, self.n_cells))
        for i, nbrs in enumerate(self.adj):
            for j in nbrs: A[i,j] = 1.0
        return A

# ══════════════════════════════════════════════════════════════════════════
class NotchDeltaASCModel:
    def __init__(self, params, lattice):
        self.p, self.lattice = params, lattice
        self.n = lattice.n_cells
        self.adj_matrix = lattice.adjacency_matrix()
        self.xp = get_xp(params.use_gpu)
        self.adj_mx = cp.asarray(self.adj_matrix) if (params.use_gpu and GPU_AVAILABLE) else self.adj_matrix

    def _unpack(self, y):
        n = self.n
        return y[0:n], y[n:2*n], y[2*n:3*n], y[3*n:4*n]

    def drift(self, t, y):
        xp, p = self.xp, self.p
        D, N, A, ASC = self._unpack(y)
        eps = 1e-12
        sum_N = self.adj_mx @ N
        sum_D = self.adj_mx @ D
        h_asc = 1.0 / (1.0 + (p.K_ASC / (ASC+eps))**p.n_d)
        h_notch = 1.0 / (1.0 + (A / (p.K_d+eps))**p.n_ASC)
        trans = p.f_N * N * sum_D
        cis = p.a_cis * N / (p.b_cis * D + N + eps)
        dD   = p.lambda_D * h_asc - p.d_D*D - p.f_D*D*sum_N
        dN   = p.lambda_N - p.d_N*N - trans - cis
        dA   = -p.d_A*A + trans + cis
        dASC = -p.d_ASC*ASC + p.lambda_ASC*h_notch
        return xp.concatenate([dD, dN, dA, dASC])

    def diffusion(self, y):
        xp, p = self.xp, self.p
        D, N, A, ASC = self._unpack(y)
        if p.noise_type == "multiplicative":
            sq = lambda x, s: s * xp.sqrt(xp.maximum(x, 0)+1e-12)
            return xp.concatenate([sq(D,p.sigma_D), sq(N,p.sigma_N),
                                   sq(A,p.sigma_A), sq(ASC,p.sigma_ASC)])
        else:
            o = xp.ones(self.n)
            return xp.concatenate([p.sigma_D*o, p.sigma_N*o,
                                   p.sigma_A*o, p.sigma_ASC*o])

    def initial_conditions(self, noise_sigma, rng=None):
        if rng is None: rng = np.random.default_rng()
        n = self.n
        y0 = np.concatenate([
            np.maximum(1.0 + noise_sigma*rng.normal(size=n), 0.001),
            np.maximum(2.0 + noise_sigma*rng.normal(size=n), 0.001),
            np.maximum(0.1 + noise_sigma*rng.normal(size=n), 0.001),
            np.maximum(2.0 + noise_sigma*rng.normal(size=n), 0.001),
        ])
        if self.p.use_gpu and GPU_AVAILABLE: y0 = cp.asarray(y0)
        return y0

# ══════════════════════════════════════════════════════════════════════════
class EulerMaruyamaIntegrator:
    def __init__(self, model, dt): self.model, self.dt = model, dt

    def integrate(self, y0, t_span, save_every=10, rng=None, show_progress=True):
        xp = self.model.xp
        if rng is None: rng = np.random.default_rng()
        t0, tf = t_span
        dt, sqrt_dt = self.dt, np.sqrt(self.dt)
        n_steps = int((tf-t0)/dt)
        n_save = n_steps//save_every + 1
        dim = len(y0)
        y, t = y0.copy(), t0
        t_out, y_out = np.zeros(n_save), np.zeros((n_save, dim))
        t_out[0] = t0
        y_out[0] = y if xp==np else cp.asnumpy(y)
        si = 1
        it = range(1, n_steps+1)
        if show_progress:
            it = tqdm(it, desc="  SDE (E-M)", leave=False, ncols=80,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for step in it:
            f = self.model.drift(t, y)
            g = self.model.diffusion(y)
            dW = rng.normal(size=dim) * sqrt_dt
            if xp != np: dW = cp.asarray(dW)
            y = y + f*dt + g*dW
            y = xp.maximum(y, 0.0)
            t += dt
            if step % save_every == 0 and si < n_save:
                t_out[si] = t
                y_out[si] = y if xp==np else cp.asnumpy(y)
                si += 1
        return t_out[:si], y_out[:si]

class RK4Integrator:
    def __init__(self, model, dt): self.model, self.dt = model, dt

    def integrate(self, y0, t_span, save_every=10, rng=None, show_progress=True):
        xp = self.model.xp
        t0, tf = t_span; dt = self.dt
        n_steps = int((tf-t0)/dt)
        n_save = n_steps//save_every + 1
        y, t = y0.copy(), t0
        t_out, y_out = np.zeros(n_save), np.zeros((n_save, len(y0)))
        t_out[0] = t0
        y_out[0] = y if xp==np else cp.asnumpy(y)
        si = 1
        it = range(1, n_steps+1)
        if show_progress:
            it = tqdm(it, desc="  RK4", leave=False, ncols=80,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for step in it:
            k1 = self.model.drift(t, y)
            k2 = self.model.drift(t+.5*dt, y+.5*dt*k1)
            k3 = self.model.drift(t+.5*dt, y+.5*dt*k2)
            k4 = self.model.drift(t+dt, y+dt*k3)
            y = y + (dt/6)*(k1+2*k2+2*k3+k4)
            y = xp.maximum(y, 0.0); t += dt
            if step % save_every == 0 and si < n_save:
                t_out[si] = t
                y_out[si] = y if xp==np else cp.asnumpy(y)
                si += 1
        return t_out[:si], y_out[:si]

# ══════════════════════════════════════════════════════════════════════════
class Simulator:
    def __init__(self, params):
        self.params = params
        self.lattice = HexLattice(params.rows, params.cols)
        self.model = NotchDeltaASCModel(params, self.lattice)
        Integ = EulerMaruyamaIntegrator if params.stochastic else RK4Integrator
        self.integrator = Integ(self.model, params.dt)

    def run_single(self, seed=None, save_every=10, show_progress=True):
        rng = np.random.default_rng(seed)
        y0 = self.model.initial_conditions(self.params.noise_sigma_ic, rng)
        return self.integrator.integrate(y0, (0, self.params.t_end),
                                         save_every, rng, show_progress)

    def run_monte_carlo(self, n_runs=None):
        if n_runs is None: n_runs = self.params.mc_runs
        n = self.lattice.n_cells
        all_final_ASC = np.zeros((n_runs, n))
        all_nb = np.zeros((n_runs, n), dtype=bool)
        thr = self.params.neuroblast_threshold
        for i in trange(n_runs, desc="  Monte Carlo", ncols=80,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            t, y = self.run_single(seed=i, save_every=100, show_progress=False)
            fa = y[-1, 3*n:4*n]
            all_final_ASC[i] = fa
            all_nb[i] = fa > thr
        return {'final_ASC': all_final_ASC, 'neuroblast': all_nb,
                'prob_map': np.mean(all_nb, axis=0), 'n_runs': n_runs}

# ══════════════════════════════════════════════════════════════════════════
class StabilityAnalyzer:
    @staticmethod
    def classify_state(final_ASC, threshold):
        nb = np.sum(final_ASC > threshold)
        r = nb / len(final_ASC)
        if nb == 0: return "uniform_low"
        elif r > 0.8: return "uniform_high"
        elif r < 0.15: return "sparse_NB"
        elif r <= 0.45: return "salt_pepper"
        else: return "mixed"

    @staticmethod
    def phase_diagram(bp, p1n, p1r, p2n, p2r, n_runs=5):
        n1, n2 = len(p1r), len(p2r)
        phase_num = np.zeros((n1, n2))
        nb_map = np.zeros((n1, n2))
        p2i = {"uniform_low":0,"sparse_NB":1,"salt_pepper":2,"mixed":3,"uniform_high":4}
        pbar = tqdm(total=n1*n2, desc="  Phase diagram", ncols=80,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for i, v1 in enumerate(p1r):
            for j, v2 in enumerate(p2r):
                p = Parameters(**bp.to_dict())
                setattr(p, p1n, v1); setattr(p, p2n, v2)
                p.t_end = min(bp.t_end, 80)
                sim = Simulator(p); n = sim.lattice.n_cells
                states, nbs = [], []
                for s in range(n_runs):
                    t, y = sim.run_single(seed=s, save_every=100, show_progress=False)
                    fa = y[-1, 3*n:4*n]
                    states.append(StabilityAnalyzer.classify_state(fa, p.neuroblast_threshold))
                    nbs.append(np.sum(fa > p.neuroblast_threshold))
                maj = Counter(states).most_common(1)[0][0]
                phase_num[i,j] = p2i.get(maj, 3)
                nb_map[i,j] = np.mean(nbs)
                pbar.update(1)
        pbar.close()
        return phase_num, nb_map

    @staticmethod
    def bifurcation_1d(params, pname, prange, n_runs=10):
        results = []
        for val in tqdm(prange, desc="  Bifurcation", ncols=80,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            p = Parameters(**params.to_dict())
            setattr(p, pname, val); p.t_end = min(params.t_end, 80)
            sim = Simulator(p); n = sim.lattice.n_cells
            vals = []
            for s in range(n_runs):
                t, y = sim.run_single(seed=s, save_every=100, show_progress=False)
                fa = y[-1, 3*n:4*n]
                vals.extend([np.max(fa), np.min(fa), np.median(fa)])
            results.append({'param_val': val, 'asc_values': np.array(vals)})
        return results

    @staticmethod
    def noise_sweep(params, noise_range, n_runs=15):
        results = []
        for sigma in tqdm(noise_range, desc="  Noise sweep", ncols=80,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            p = Parameters(**params.to_dict())
            p.sigma_D = sigma; p.sigma_N = sigma
            p.sigma_A = sigma*0.6; p.sigma_ASC = sigma
            p.t_end = min(params.t_end, 80)
            sim = Simulator(p); n = sim.lattice.n_cells
            nbs = []
            for s in range(n_runs):
                t, y = sim.run_single(seed=s, save_every=100, show_progress=False)
                nbs.append(np.sum(y[-1, 3*n:4*n] > p.neuroblast_threshold))
            results.append({'noise_sigma': sigma, 'nb_counts': np.array(nbs),
                            'mean_nb': np.mean(nbs), 'std_nb': np.std(nbs)})
        return results

    @staticmethod
    def param_sweep_heatmap(bp, p1n, p1r, p2n, p2r, n_runs=5):
        n1, n2 = len(p1r), len(p2r)
        hm = np.zeros((n1, n2))
        pbar = tqdm(total=n1*n2, desc="  Heatmap", ncols=80,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for i, v1 in enumerate(p1r):
            for j, v2 in enumerate(p2r):
                p = Parameters(**bp.to_dict())
                setattr(p, p1n, v1); setattr(p, p2n, v2)
                p.t_end = min(bp.t_end, 80)
                sim = Simulator(p); n = sim.lattice.n_cells
                nbs = []
                for s in range(n_runs):
                    t, y = sim.run_single(seed=s, save_every=100, show_progress=False)
                    nbs.append(np.sum(y[-1, 3*n:4*n] > p.neuroblast_threshold))
                hm[i,j] = np.mean(nbs)
                pbar.update(1)
        pbar.close()
        return hm

# ══════════════════════════════════════════════════════════════════════════
class Visualizer:
    NB_CMAP = LinearSegmentedColormap.from_list('nb',
        ['#2b6cb0','#63b3ed','#faf089','#f6ad55','#e53e3e'], N=256)
    PROB_CMAP = LinearSegmentedColormap.from_list('prob',
        ['#edf2f7','#bee3f8','#4299e1','#2b6cb0','#1a365d'], N=256)

    @staticmethod
    def _hp(x, y, s=0.45):
        a = np.linspace(0, 2*np.pi, 7)
        return patches.Polygon([(x+s*np.cos(t), y+s*np.sin(t)) for t in a], closed=True)

    @classmethod
    def plot_lattice(cls, lat, vals, title, fp, cmap=None, vmin=None, vmax=None,
                     cbar_label="Value", highlight=None):
        if cmap is None: cmap = cls.NB_CMAP
        fig, ax = plt.subplots(figsize=(8,7))
        norm = Normalize(vmin=vmin if vmin is not None else np.min(vals),
                         vmax=vmax if vmax is not None else np.max(vals))
        hps, cols = [], []
        for i in range(lat.n_cells):
            hps.append(cls._hp(*lat.positions[i])); cols.append(vals[i])
        pc = PatchCollection(hps, cmap=cmap, norm=norm, edgecolors='#2d3748', linewidths=1.2)
        pc.set_array(np.array(cols)); ax.add_collection(pc)
        for i in range(lat.n_cells):
            x, y = lat.positions[i]
            fc, fw, fs = 'white', 'normal', 6
            if highlight and i in highlight:
                fc, fw, fs = '#ffd700', 'bold', 9
                ax.plot(x, y+0.35, '*', color='#ffd700', ms=10, mec='black', mew=0.5)
            ax.text(x, y, str(i), ha='center', va='center', fontsize=fs, color=fc, fontweight=fw, alpha=0.9)
        ax.set_xlim(lat.positions[:,0].min()-0.6, lat.positions[:,0].max()+0.6)
        ax.set_ylim(lat.positions[:,1].min()-0.7, lat.positions[:,1].max()+0.7)
        ax.set_aspect('equal'); ax.set_title(title, fontsize=14, fontweight='bold', pad=15); ax.axis('off')
        plt.colorbar(pc, ax=ax, shrink=0.7, pad=0.05).set_label(cbar_label, fontsize=11)
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_cell_tracking(cls, t, y, nc, tracked, fp):
        if not tracked: return
        nt = len(tracked)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        vn = ['Delta (D)', 'Inactive Notch (N)', 'Active Notch (A)', 'ASC']
        cm = plt.cm.Set1(np.linspace(0, 0.9, nt))
        for k, (ax, name) in enumerate(zip(axes.flat, vn)):
            for ci, c in enumerate(tracked):
                ax.plot(t, y[:, k*nc+c], color=cm[ci], lw=2, label=f'Cell {c}', alpha=0.9)
            ax.set_xlabel('Time'); ax.set_ylabel('Conc.')
            ax.set_title(name, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.suptitle(f'Tracked Cells: {tracked}', fontsize=15, fontweight='bold', y=1.02)
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_asc_tracking(cls, t, y, nc, tracked, fp, thr=5.0):
        if not tracked: return
        fig, ax = plt.subplots(figsize=(12, 6))
        cm = plt.cm.Set1(np.linspace(0, 0.9, len(tracked)))
        for ci, c in enumerate(tracked):
            ax.plot(t, y[:, 3*nc+c], color=cm[ci], lw=2.5, label=f'Cell {c}', alpha=0.9)
        ax.axhline(thr, color='#e53e3e', ls='--', lw=2, alpha=0.7, label=f'NB Threshold={thr}')
        ymax = max(ax.get_ylim()[1], thr*1.2)
        ax.fill_between(t, thr, ymax, color='#fed7d7', alpha=0.3, label='Neuroblast zone')
        ax.set_ylim(bottom=-0.5, top=ymax)
        ax.set_xlabel('Time', fontsize=13); ax.set_ylabel('ASC', fontsize=13)
        ax.set_title('ASC Level Tracking: Neuroblast Selection', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_timeseries(cls, t, y, nc, fp, n_show=None):
        if n_show is None: n_show = min(nc, 12)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        labs = ['Delta', 'Inactive Notch', 'Active Notch', 'ASC']
        cc = plt.cm.tab10(np.linspace(0, 1, n_show))
        for k, (ax, lab) in enumerate(zip(axes.flat, labs)):
            for ci in range(n_show):
                ax.plot(t, y[:, k*nc+ci], color=cc[ci], alpha=0.6, lw=0.8)
            ax.set_xlabel('Time'); ax.set_ylabel('Conc.')
            ax.set_title(lab, fontweight='bold'); ax.grid(True, alpha=0.3)
        fig.suptitle('All Variables (Stochastic)', fontsize=15, fontweight='bold', y=1.02)
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_mc_prob(cls, lat, pm, fp, hl=None):
        cls.plot_lattice(lat, pm, 'MC Neuroblast Probability', fp,
                         cmap=cls.PROB_CMAP, vmin=0, vmax=1, cbar_label='P(NB)', highlight=hl)

    @classmethod
    def plot_mc_hist(cls, mc, fp):
        nb = np.sum(mc['neuroblast'], axis=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(nb, bins=range(0, int(np.max(nb))+3), color='#4299e1', edgecolor='#2b6cb0',
                alpha=0.85, align='left', rwidth=0.85)
        ax.axvline(np.mean(nb), color='#e53e3e', ls='--', lw=2, label=f'Mean={np.mean(nb):.1f}')
        ax.set_xlabel('# Neuroblasts'); ax.set_ylabel('Freq')
        ax.set_title(f'NB Count (n={mc["n_runs"]})', fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_bifurcation(cls, res, pname, fp):
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in res:
            ax.scatter(np.full_like(r['asc_values'], r['param_val']),
                       r['asc_values'], c='#2b6cb0', s=5, alpha=0.4)
        ax.set_xlabel(pname, fontsize=13); ax.set_ylabel('Final ASC', fontsize=13)
        ax.set_title(f'Bifurcation: {pname}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_noise_sweep(cls, res, fp):
        sigs = [r['noise_sigma'] for r in res]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].errorbar(sigs, [r['mean_nb'] for r in res], yerr=[r['std_nb'] for r in res],
                         fmt='o-', color='#2b6cb0', capsize=4, lw=2, ms=6)
        axes[0].set_xlabel('SDE σ'); axes[0].set_ylabel('Mean NB')
        axes[0].set_title('Stochastic Noise vs Differentiation', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        bp = axes[1].boxplot([r['nb_counts'] for r in res], positions=range(len(sigs)), patch_artist=True)
        for p in bp['boxes']: p.set_facecolor('#bee3f8'); p.set_edgecolor('#2b6cb0')
        axes[1].set_xticklabels([f'{s:.3f}' for s in sigs], rotation=45)
        axes[1].set_xlabel('SDE σ'); axes[1].set_ylabel('NB Count')
        axes[1].set_title('Distribution per Noise', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_phase_diagram(cls, pnum, nbmap, p1n, p1r, p2n, p2r, fp):
        pnames = ['uniform_low','sparse_NB','salt_pepper','mixed','uniform_high']
        pcols = ['#a0aec0','#68d391','#4299e1','#ed8936','#e53e3e']
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        cm = LinearSegmentedColormap.from_list('ph', pcols, N=5)
        im = axes[0].imshow(pnum.T, origin='lower', aspect='auto', cmap=cm, vmin=-0.5, vmax=4.5,
                            extent=[p1r[0],p1r[-1],p2r[0],p2r[-1]])
        axes[0].set_xlabel(p1n); axes[0].set_ylabel(p2n)
        axes[0].set_title('Phase Diagram', fontsize=14, fontweight='bold')
        cb = plt.colorbar(im, ax=axes[0], ticks=[0,1,2,3,4], shrink=0.8)
        cb.set_ticklabels(pnames)
        im2 = axes[1].imshow(nbmap.T, origin='lower', aspect='auto', cmap='YlOrRd',
                             extent=[p1r[0],p1r[-1],p2r[0],p2r[-1]])
        axes[1].set_xlabel(p1n); axes[1].set_ylabel(p2n)
        axes[1].set_title('Mean NB Count', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=axes[1], shrink=0.8).set_label('Mean # NB')
        fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

    @classmethod
    def plot_dashboard(cls, lat, t, y, mc, params, fp, tracked=None):
        n = lat.n_cells
        fASC, fD = y[-1, 3*n:4*n], y[-1, :n]
        fig = plt.figure(figsize=(22, 16))
        gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

        # Row 0: lattices
        for ci, (data, cmap, ttl) in enumerate([
            (fASC, cls.NB_CMAP, 'Final ASC'), (fD, plt.cm.Greens, 'Final Delta'),
            (mc['prob_map'], cls.PROB_CMAP, 'NB Prob (MC)')]):
            ax = fig.add_subplot(gs[0, ci])
            norm = Normalize(vmin=0 if ci==2 else np.min(data), vmax=1 if ci==2 else np.max(data))
            for i in range(n):
                x, yp = lat.positions[i]
                h = cls._hp(x, yp)
                ax.add_patch(h); h.set_facecolor(cmap(norm(data[i])))
                h.set_edgecolor('#2d3748'); h.set_linewidth(1)
                if tracked and i in tracked:
                    ax.plot(x, yp+0.35, '*', color='#ffd700', ms=8, mec='black', mew=0.5)
            ax.set_xlim(lat.positions[:,0].min()-0.6, lat.positions[:,0].max()+0.6)
            ax.set_ylim(lat.positions[:,1].min()-0.7, lat.positions[:,1].max()+0.7)
            ax.set_aspect('equal'); ax.set_title(ttl, fontweight='bold'); ax.axis('off')

        # Row 1: time series with tracked highlights
        ns = min(n, 12); cc = plt.cm.tab10(np.linspace(0, 1, ns))
        for ci, (vo, yl, ttl) in enumerate([(3,'ASC','ASC'),(0,'Delta','Delta'),(2,'Active Notch','Act. Notch')]):
            ax = fig.add_subplot(gs[1, ci])
            for c in range(ns): ax.plot(t, y[:, vo*n+c], color=cc[c], alpha=0.5, lw=0.7)
            if tracked:
                tc = plt.cm.Set1(np.linspace(0, 0.9, len(tracked)))
                for ti, c in enumerate(tracked):
                    ax.plot(t, y[:, vo*n+c], color=tc[ti], lw=2.5, alpha=0.95, label=f'Cell {c}', zorder=10)
                ax.legend(fontsize=7)
            ax.set_xlabel('Time'); ax.set_ylabel(yl); ax.set_title(ttl, fontweight='bold'); ax.grid(True, alpha=0.3)

        # Row 2: stats
        ax7 = fig.add_subplot(gs[2, 0])
        nb = np.sum(mc['neuroblast'], axis=1)
        ax7.hist(nb, bins=range(0, int(np.max(nb))+3), color='#4299e1', edgecolor='#2b6cb0', alpha=0.85, align='left', rwidth=0.85)
        ax7.axvline(np.mean(nb), color='#e53e3e', ls='--', lw=2, label=f'Mean={np.mean(nb):.1f}')
        ax7.set_xlabel('# NB'); ax7.set_ylabel('Freq'); ax7.set_title('NB Distribution', fontweight='bold')
        ax7.legend(); ax7.grid(True, alpha=0.3, axis='y')

        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(mc['final_ASC'].flatten(), bins=50, color='#f6ad55', edgecolor='#c05621', alpha=0.85)
        ax8.axvline(params.neuroblast_threshold, color='#e53e3e', ls='--', lw=2, label=f'Thr={params.neuroblast_threshold}')
        ax8.set_xlabel('Final ASC'); ax8.set_ylabel('Freq'); ax8.set_title('ASC Distribution', fontweight='bold')
        ax8.legend(); ax8.grid(True, alpha=0.3, axis='y')

        ax9 = fig.add_subplot(gs[2, 2]); ax9.axis('off')
        ms = f"SDE (σ_D={params.sigma_D})" if params.stochastic else "ODE (RK4)"
        ts = str(tracked) if tracked else "None"
        info = (f"Mode: {ms}\nNoise: {params.noise_type}\n"
                f"Lattice: {params.rows}×{params.cols} ({n})\n"
                f"t={params.t_end}  dt={params.dt}\nIC σ={params.noise_sigma_ic}\n"
                f"MC runs={mc['n_runs']}  Thr={params.neuroblast_threshold}\n"
                f"Tracked: {ts}\n{'─'*30}\n"
                f"λD={params.lambda_D} K_ASC={params.K_ASC} nd={params.n_d}\n"
                f"dD={params.d_D} fD={params.f_D}\n"
                f"λN={params.lambda_N} dN={params.d_N} fN={params.f_N}\n"
                f"a_cis={params.a_cis} b_cis={params.b_cis} dA={params.d_A}\n"
                f"λASC={params.lambda_ASC} dASC={params.d_ASC}\n"
                f"Kd={params.K_d} nASC={params.n_ASC}\n"
                f"σD={params.sigma_D} σN={params.sigma_N} σA={params.sigma_A} σASC={params.sigma_ASC}")
        ax9.text(0.05, 0.95, info, transform=ax9.transAxes, fontsize=9, va='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#edf2f7', alpha=0.8))
        ax9.set_title('Parameters', fontweight='bold')

        fig.suptitle('Notch-Delta-ASC Stochastic Simulator v2.0', fontsize=18, fontweight='bold', y=1.01)
        fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"    ✓ {fp}")

# ══════════════════════════════════════════════════════════════════════════
def run_full_pipeline(params=None):
    if params is None: params = Parameters()
    outdir = params.output_dir; os.makedirs(outdir, exist_ok=True)
    pf = params.save_prefix
    mode = "STOCHASTIC (Euler-Maruyama)" if params.stochastic else "DETERMINISTIC (RK4)"

    print("═"*65)
    print("  Notch-Delta-ASC Simulator v2.0")
    print(f"  Mode: {mode}")
    print("═"*65)

    # [1] Parameters
    params.save(os.path.join(outdir, f"{pf}_parameters.txt"))
    print(f"\n[1/8] Parameters saved")

    # Tracking
    nc = params.rows * params.cols
    tracked = params.get_tracked_cells()
    if not tracked:
        ctr = nc//2
        tracked = sorted(set(c for c in [0, ctr, nc-1, params.cols//2, ctr+params.cols//2] if 0<=c<nc))[:5]
    tracked = [c for c in tracked if 0<=c<nc]
    print(f"       Tracked cells: {tracked}")

    # [2] Single run
    print(f"\n[2/8] Single simulation (t=0..{params.t_end})...")
    t0 = time.time()
    sim = Simulator(params)
    ta, ya = sim.run_single(seed=42, save_every=5)
    n = sim.lattice.n_cells
    fASC = ya[-1, 3*n:4*n]
    nb = np.sum(fASC > params.neuroblast_threshold)
    print(f"       {time.time()-t0:.1f}s — NB: {nb}/{n}  ASC: {np.min(fASC):.3f}–{np.max(fASC):.3f}")

    # [3] Lattice + timeseries
    print(f"\n[3/8] Lattice & timeseries plots...")
    Visualizer.plot_lattice(sim.lattice, fASC, 'Final ASC (Stochastic)',
                            os.path.join(outdir, f"{pf}_lattice_ASC.png"), highlight=tracked)
    Visualizer.plot_lattice(sim.lattice, ya[-1,:n], 'Final Delta',
                            os.path.join(outdir, f"{pf}_lattice_Delta.png"),
                            cmap=plt.cm.Greens, cbar_label='Delta', highlight=tracked)
    Visualizer.plot_timeseries(ta, ya, n, os.path.join(outdir, f"{pf}_timeseries.png"))

    # [4] Cell tracking
    print(f"\n[4/8] Cell tracking...")
    Visualizer.plot_cell_tracking(ta, ya, n, tracked, os.path.join(outdir, f"{pf}_cell_tracking.png"))
    Visualizer.plot_asc_tracking(ta, ya, n, tracked, os.path.join(outdir, f"{pf}_asc_tracking.png"),
                                 thr=params.neuroblast_threshold)

    # [5] Monte Carlo
    print(f"\n[5/8] Monte Carlo ({params.mc_runs} runs)...")
    t0 = time.time()
    mc = sim.run_monte_carlo()
    print(f"       {time.time()-t0:.1f}s")
    Visualizer.plot_mc_prob(sim.lattice, mc['prob_map'], os.path.join(outdir, f"{pf}_mc_prob.png"), hl=tracked)
    Visualizer.plot_mc_hist(mc, os.path.join(outdir, f"{pf}_mc_hist.png"))

    # [6] Noise sweep
    print(f"\n[6/8] SDE noise sweep...")
    t0 = time.time()
    nr = StabilityAnalyzer.noise_sweep(params, np.linspace(0, 0.15, 8), n_runs=10)
    Visualizer.plot_noise_sweep(nr, os.path.join(outdir, f"{pf}_noise_sweep.png"))
    print(f"       {time.time()-t0:.1f}s")

    # [7] Bifurcation
    print(f"\n[7/8] Bifurcation (lambda_D)...")
    t0 = time.time()
    br = StabilityAnalyzer.bifurcation_1d(params, 'lambda_D', np.linspace(0.5, 8.0, 12), n_runs=6)
    Visualizer.plot_bifurcation(br, 'lambda_D', os.path.join(outdir, f"{pf}_bifurcation.png"))
    print(f"       {time.time()-t0:.1f}s")

    # [8] Phase diagram
    print(f"\n[8/8] Phase diagram (f_N × sigma_D)...")
    t0 = time.time()
    p1r, p2r = np.linspace(0.1, 2.0, 8), np.linspace(0, 0.15, 7)
    pn, nbm = StabilityAnalyzer.phase_diagram(params, 'f_N', p1r, 'sigma_D', p2r, n_runs=4)
    Visualizer.plot_phase_diagram(pn, nbm, 'f_N', p1r, 'sigma_D', p2r,
                                  os.path.join(outdir, f"{pf}_phase_diagram.png"))
    print(f"       {time.time()-t0:.1f}s")

    # Dashboard
    print(f"\n[+] Dashboard...")
    Visualizer.plot_dashboard(sim.lattice, ta, ya, mc, params,
                              os.path.join(outdir, f"{pf}_dashboard.png"), tracked)

    print("\n" + "═"*65)
    print(f"  All outputs → {outdir}/  ({len(os.listdir(outdir))} files)")
    print("═"*65)
    return sim, ta, ya, mc

# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Notch-Delta-ASC v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python %(prog)s --track 0,12,24\n  python %(prog)s --no-stochastic\n  python %(prog)s --sigma_D 0.1 --sigma_ASC 0.1")
    parser.add_argument('--rows', type=int, default=7)
    parser.add_argument('--cols', type=int, default=7)
    parser.add_argument('--t_end', type=float, default=200.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--noise_ic', type=float, default=0.15, dest='noise_sigma_ic')
    parser.add_argument('--stochastic', action='store_true', default=True)
    parser.add_argument('--no-stochastic', action='store_false', dest='stochastic')
    parser.add_argument('--noise_type', type=str, default='multiplicative', choices=['multiplicative','additive'])
    parser.add_argument('--sigma_D', type=float, default=0.05)
    parser.add_argument('--sigma_N', type=float, default=0.05)
    parser.add_argument('--sigma_A', type=float, default=0.03)
    parser.add_argument('--sigma_ASC', type=float, default=0.05)
    parser.add_argument('--mc_runs', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=5.0, dest='neuroblast_threshold')
    parser.add_argument('--track', type=str, default='', dest='track_cells')
    parser.add_argument('--gpu', action='store_true', dest='use_gpu')
    parser.add_argument('--outdir', type=str, default='results', dest='output_dir')
    parser.add_argument('--load_params', type=str, default=None)
    parser.add_argument('--lambda_D', type=float, default=5.0)
    parser.add_argument('--K_ASC', type=float, default=0.3)
    parser.add_argument('--n_d', type=float, default=5.0)
    parser.add_argument('--d_D', type=float, default=0.2)
    parser.add_argument('--f_D', type=float, default=0.05)
    parser.add_argument('--lambda_N', type=float, default=1.5)
    parser.add_argument('--d_N', type=float, default=0.2)
    parser.add_argument('--f_N', type=float, default=0.8)
    parser.add_argument('--a_cis', type=float, default=0.1)
    parser.add_argument('--b_cis', type=float, default=2.0)
    parser.add_argument('--d_A', type=float, default=0.3)
    parser.add_argument('--lambda_ASC', type=float, default=5.0)
    parser.add_argument('--d_ASC', type=float, default=0.15)
    parser.add_argument('--K_d', type=float, default=0.5)
    parser.add_argument('--n_ASC', type=float, default=5.0)
    args = parser.parse_args()
    if args.load_params: params = Parameters.load(args.load_params)
    else: params = Parameters(**{k:v for k,v in vars(args).items() if k!='load_params'})
    run_full_pipeline(params)
