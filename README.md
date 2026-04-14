# Notch-Delta-ASC Lateral Inhibition Simulator

A stochastic simulation framework for modeling **neuroblast selection** via Notch-Delta signaling and Achaete-Scute Complex (ASC) regulation on a 2D hexagonal lattice.

This project implements a **custom-designed 4-variable ODE/SDE system** that extends classical lateral inhibition models (e.g., Collier et al., 1996) by explicitly incorporating:

- Separate **inactive** and **active (cleaved) Notch** dynamics  
- **Trans-activation** between adjacent cells  
- **Cis-inhibition** within the same cell  
- **ASC proneural gene** regulation with Hill-type repression by NICD  

---

## Mathematical Model

The following system of coupled ordinary/stochastic differential equations governs each cell $i$ on a hexagonal lattice. **This formulation was independently derived** by extending established Notch-Delta frameworks with explicit ASC dynamics and cis-inhibition kinetics.

### Deterministic Core (ODE)

$$\frac{dD_i}{dt} = \lambda_D \cdot \frac{1}{1 + \left(\frac{K_{ASC}}{ASC_i}\right)^{n_d}} - d_D \, D_i - f_D \, D_i \sum_{j \in \mathcal{N}(i)} N_j$$

$$\frac{dN_i}{dt} = \lambda_N - d_N \, N_i - f_N \, N_i \sum_{j \in \mathcal{N}(i)} D_j - \frac{a \, N_i}{b \, D_i + N_i}$$

$$\frac{dA_i}{dt} = -d_A \, A_i + f_N \, N_i \sum_{j \in \mathcal{N}(i)} D_j + \frac{a \, N_i}{b \, D_i + N_i}$$

$$\frac{dASC_i}{dt} = -d_{ASC} \, ASC_i + \lambda_{ASC} \cdot \frac{1}{1 + \left(\frac{A_i}{K_d}\right)^{n_{ASC}}}$$

where $\mathcal{N}(i)$ denotes the set of adjacent cells on the hexagonal lattice.

### Variable Descriptions

| Variable | Description |
|----------|-------------|
| $D_i$ | **Delta** ligand concentration in cell $i$ |
| $N_i$ | **Inactive Notch** (uncleaved receptor) in cell $i$ |
| $A_i$ | **Active Notch** (NICD, cleaved intracellular domain) in cell $i$ |
| $ASC_i$ | **Achaete-Scute Complex** proneural transcription factor in cell $i$ |

### Key Mechanistic Terms

**Delta production** — ASC activates Delta transcription via a Hill function:

$$\lambda_D \cdot \frac{1}{1 + (K_{ASC}/ASC_i)^{n_d}}$$

**Trans-activation** — Neighboring Delta cleaves Notch in cell $i$, converting inactive Notch ($N$) to active Notch ($A$). This term appears with opposite signs in $dN/dt$ and $dA/dt$:

$$f_N \, N_i \sum_{j \in \mathcal{N}(i)} D_j$$

**Cis-inhibition** — Same-cell Delta-Notch interaction modeled as a saturating function. In the current formulation, this term is designed so that cis-inhibition strength depends on the ratio of Notch to Delta within the same cell:

$$\frac{a \, N_i}{b \, D_i + N_i}$$

> **Design note:** This Michaelis-Menten-like form captures the competition between cis and trans pathways. When intracellular Delta ($D_i$) is high relative to Notch ($N_i$), cis-inhibition is attenuated — reflecting the idea that Delta is primarily engaged in trans-signaling to neighbors. Alternative formulations where cis-inhibition scales proportionally with $D_i$ (e.g., $a \cdot D_i \cdot N_i / (K + D_i \cdot N_i)$) may better capture direct cis-binding and could be explored in future work.

**ASC repression** — Active Notch (NICD) represses ASC transcription via HES/E(spl) pathway, modeled as an inhibitory Hill function:

$$\lambda_{ASC} \cdot \frac{1}{1 + (A_i / K_d)^{n_{ASC}}}$$

### Stochastic Extension (SDE)

Each equation is extended with multiplicative noise to model intrinsic biochemical fluctuations:

$$dX_i = f(X_i)\,dt + \sigma_X \sqrt{X_i}\,dW_i$$

where $dW_i$ are independent Wiener increments. The $\sqrt{X}$ scaling reflects birth-death process noise. Additive noise ($\sigma_X \cdot dW_i$) is also supported. Integration uses the **Euler-Maruyama** method.

### Feedback Loop Summary

The core lateral inhibition circuit operates as a **positive feedback loop** between adjacent cells:

```
Cell i: High ASC → High Delta → [trans] → Neighbor Notch activated
                                              ↓
Cell j: High Active Notch → HES/E(spl) → ASC repressed → Low Delta
                                              ↓
Cell i: Less Notch activation ← Low neighbor Delta → ASC stays high
```

Initial stochastic fluctuations break symmetry. The feedback amplifies small differences until a stable **salt-and-pepper pattern** emerges: isolated neuroblasts (high ASC) surrounded by epidermal cells (low ASC).

---

## Features

- **Hexagonal lattice** with periodic boundary conditions  
- **Stochastic SDE** (Euler-Maruyama) with multiplicative or additive noise  
- **Deterministic ODE** mode (RK4) as fallback  
- **Cell tracking** — monitor specific cells across all four variables over time  
- **Monte Carlo ensemble** — probability maps and neuroblast count distributions  
- **Phase diagram** — classify parameter space into distinct dynamical regimes  
- **Bifurcation analysis** — sweep individual parameters  
- **2D parameter heatmaps** — explore interactions between any two parameters  
- **Noise sweep** — quantify the effect of stochastic intensity on differentiation  
- **Parameter I/O** — save/load full parameter sets as `.txt`  
- **All figures exported as `.png`** (200 dpi)  
- **tqdm progress bars** for all computationally intensive steps  
- **Optional GPU acceleration** via CuPy  

---

## Installation

### Requirements

- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib
- tqdm

### Optional

- CuPy (for GPU acceleration)

```bash
pip install numpy scipy matplotlib tqdm
# Optional GPU support:
# pip install cupy-cuda12x
```

---

## Usage

### Basic Run (Stochastic)

```bash
python notch_delta_simulator.py
```

### Track Specific Cells

```bash
python notch_delta_simulator.py --track 0,12,24,6,18
```

### Deterministic Mode

```bash
python notch_delta_simulator.py --no-stochastic
```

### Custom Parameters

```bash
python notch_delta_simulator.py \
  --rows 9 --cols 9 \
  --t_end 300 --dt 0.005 \
  --sigma_D 0.1 --sigma_ASC 0.08 \
  --mc_runs 200 \
  --track 0,20,40
```

### Load from Parameter File

```bash
python notch_delta_simulator.py --load_params results/notch_delta_parameters.txt
```

### GPU Mode

```bash
python notch_delta_simulator.py --gpu
```

---

## Output Files

All outputs are saved to the `--outdir` directory (default: `results/`).

| File | Description |
|------|-------------|
| `*_parameters.txt` | Full parameter dump (reloadable) |
| `*_lattice_ASC.png` | Hex lattice colored by final ASC concentration |
| `*_lattice_Delta.png` | Hex lattice colored by final Delta concentration |
| `*_timeseries.png` | Time series of all four variables (D, N, A, ASC) |
| `*_cell_tracking.png` | Tracked cells: all four variables individually |
| `*_asc_tracking.png` | Tracked cells: ASC comparison with NB threshold |
| `*_mc_prob.png` | Monte Carlo neuroblast probability map |
| `*_mc_hist.png` | Neuroblast count distribution across MC runs |
| `*_noise_sweep.png` | SDE noise intensity vs. differentiation outcome |
| `*_bifurcation.png` | Bifurcation diagram (default: λ_D) |
| `*_phase_diagram.png` | Phase classification + NB count heatmap |
| `*_dashboard.png` | 9-panel summary dashboard |

---

## Parameters

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_D` | 5.0 | Delta max production rate |
| `K_ASC` | 0.3 | Hill threshold: ASC → Delta activation |
| `n_d` | 5.0 | Hill coefficient (Delta production) |
| `d_D` | 0.2 | Delta degradation rate |
| `f_D` | 0.05 | Delta consumption via trans-binding |
| `lambda_N` | 1.5 | Notch basal production rate |
| `d_N` | 0.2 | Notch degradation rate |
| `f_N` | 0.8 | Trans-activation rate |
| `a_cis` | 0.1 | Cis-inhibition strength |
| `b_cis` | 2.0 | Cis-inhibition saturation parameter |
| `d_A` | 0.3 | Active Notch (NICD) degradation |
| `lambda_ASC` | 5.0 | ASC max production rate |
| `d_ASC` | 0.15 | ASC degradation rate |
| `K_d` | 0.5 | Hill threshold: NICD → ASC repression |
| `n_ASC` | 5.0 | Hill coefficient (ASC repression) |
| `sigma_D` | 0.05 | SDE noise intensity (Delta) |
| `sigma_N` | 0.05 | SDE noise intensity (Notch) |
| `sigma_A` | 0.03 | SDE noise intensity (Active Notch) |
| `sigma_ASC` | 0.05 | SDE noise intensity (ASC) |

---

## Phase Classification

The simulator automatically classifies the final state of each simulation into one of five phases:

| Phase | Criteria | Interpretation |
|-------|----------|----------------|
| `uniform_low` | 0 neuroblasts | All cells adopt epidermal fate; no symmetry breaking |
| `sparse_NB` | < 15% neuroblasts | Weak lateral inhibition; few isolated neuroblasts |
| `salt_pepper` | 15–45% neuroblasts | Classic lateral inhibition pattern |
| `mixed` | 45–80% neuroblasts | Intermediate; potentially chaotic dynamics |
| `uniform_high` | > 80% neuroblasts | Pathological; inhibition failure |

---

## Biological Context

During *Drosophila* neurogenesis, **proneural clusters** of cells in the neuroectoderm all express Achaete-Scute Complex (ASC) genes. Through **Notch-Delta lateral inhibition**, a single cell within each cluster is selected as the **neuroblast** (neural precursor), while surrounding cells adopt the **epidermoblast** (epidermal) fate.

This simulator models this process at the molecular level, capturing:

1. **Proneural equivalence** — all cells begin with identical ASC expression  
2. **Stochastic symmetry breaking** — intrinsic noise creates initial asymmetry  
3. **Feedback amplification** — the Notch-Delta-ASC circuit amplifies differences  
4. **Stable pattern formation** — a salt-and-pepper pattern of neuroblasts emerges  

---

## Limitations & Future Directions

This model is a **work in progress** and has several areas for improvement:

- **Cis-inhibition formulation** — the current saturating form $a \cdot N/(b \cdot D + N)$ could be replaced with a direct binding term proportional to $D \cdot N$ to better reflect cis-interaction biochemistry.
- **HES/E(spl) intermediate** — the model currently collapses the NICD → HES → ASC repression cascade into a single Hill function. Adding HES as a fifth variable would improve mechanistic accuracy.
- **3D tissue geometry** — extending to 3D cell packing and realistic tissue morphology.
- **Gillespie algorithm** — for exact stochastic simulation at low molecule counts, replacing the Langevin SDE approximation.
- **Parameter estimation** — fitting to experimental data from live imaging of *Drosophila* neuroectoderm.
- **Lateral induction** — incorporating Notch-dependent upregulation of Delta (the "mutual activation" mode) that operates in some developmental contexts.
- **Cell division and growth** — dynamic lattice with proliferating cells.
- **Multi-gene proneural regulation** — distinguishing individual ASC genes (achaete, scute, lethal of scute, asense) and their distinct roles.

---

## Acknowledgments

This project was inspired by the **Developmental Neurobiology** course at the **University of Pennsylvania**. The mathematical model was independently formulated by extending classical Notch-Delta lateral inhibition frameworks with explicit ASC dynamics and cis-inhibition kinetics.

Development of the simulation code was assisted by **Claude** (Anthropic) and **ChatGPT** (OpenAI).

---

## License

This project is licensed under the **MIT License**. See below.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
