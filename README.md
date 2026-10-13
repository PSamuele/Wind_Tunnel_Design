# Wind Tunnel Design — open-circuit supersonic blowdown

Python tool that sizes an **axisymmetric open-circuit blowdown wind tunnel**
(pressurised tank → regulator → settling chamber → contraction → MOC nozzle →
test section → second throat → subsonic diffuser → atmosphere) and analyses it
in **steady state** and in **transient** conditions, with a dedicated study of
the **second throat**.

Inputs are declared in a Python file (`inputs.py`); results go to one folder
per distinct input, organised by topic: CSV tables with units, SVG figures,
ParaView files, a readable summary and an HTML/PDF report.

## Installation

Requires Python 3.11 or newer (the standard-library `tomllib` is used).

```bash
git clone https://github.com/PSamuele/Wind_Tunnel_Design.git
cd Wind_Tunnel_Design
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # numpy, scipy, matplotlib
pip install -e .[dev]       # + pytest, for the tests
```

## Quick start

```bash
python run.py                 # inputs from inputs.py
python run.py --quick         # coarse resolution, fast check (about 2 minutes)
python run.py examples/mach3.py
```

`run.py` also works without installing the package (it adds `src/` to the
path; numpy, scipy and matplotlib must be installed). Options:

| Option | Effect |
|---|---|
| `--results-root DIR` | root folder of the results (default `results`) |
| `--quick` | coarse resolution for a fast check |
| `--no-transient` | skip the Euler transients |
| `--no-second-throat` | skip the second-throat studies |
| `--no-optimisation` | skip the optimisation |
| `--no-sweep` | skip the Mach trade study |

A full run takes 4–7 minutes, mostly in the Euler transients, the
second-throat studies and the optimisation. At the end the console prints
`summary.txt` and the path of the results folder.

## Inputs

All parameters are declared in [`inputs.py`](inputs.py), grouped by component,
each with its unit and meaning:

```python
SPEC = TunnelSpec(
    design=DesignSpec(
        mach=2.0,                    # [-] test-section Mach number
        test_section_diameter=0.15,  # [m] test-section diameter
        ...
    ),
    model=ModelSpec(
        frontal_area=0.0015,         # [m^2] model frontal area, 0 = empty test section
        cd=0.0,                      # [-] drag coefficient, 0 = automatic (shock theory + base pressure)
        ...
    ),
    ...
)
```

Edit the values and run `python run.py`. To keep several cases, copy the file
(see `examples/mach3.py`) and pass it to `run.py`. Every run also writes
`inputs_used.py` into its results folder, so any run can be repeated exactly.
A TOML file with the same sections and keys is accepted as an alternative
(`examples/mach2.toml`, only the values that differ from the defaults).
Unknown parameter names are rejected.

## What it computes

| Area | Content |
|---|---|
| Sizing | throat, settling chamber, test section, second throat, diffuser; mass flow; test-section static state, unit Reynolds number, test rhombus length |
| Nozzle | axisymmetric contour by the method of characteristics (source-flow kernel + prescribed centreline Mach), uniform exit flow |
| Steady quasi-1D flow | Mach, static and total pressure, p/p0, temperature, velocity, density along the axis, for the starting and running conditions |
| Losses | honeycomb + screens, Fanno friction in constant-area ducts, **model drag**, oblique shocks in the convergent (option), normal shock, subsonic diffuser, exhaust dynamic pressure |
| Shock analysis | starting and minimum running pressure ratios, start/unstart hysteresis map, Kantrowitz blockage limit, oblique/conical (Taylor-Maccoll) shock on the model |
| **Second throat** | two-throat sonic compatibility and regimes R0–R3, variable geometry (open to start, closed to run), sensitivity to the throat area, ramp (oblique-shock) convergent, recommendations |
| Transient, Euler | unsteady quasi-1D Euler solver (HLLC, MUSCL, SSP-RK3): start, running, unstart; fixed throat and **variable throat with moving walls** |
| Transient, run | tank depletion with regulator, run time, T0 and Reynolds drift, start/unstart hysteresis, liquefaction over time |
| Optimisation | second-throat margin and length, diffuser half-angle and area ratio, minimising the running pressure ratio under length and tank-pressure constraints |
| Trade study | pressure ratios, run time, mass flow, liquefaction limit vs Mach number |

Mach range: 1.5–4 (accepted 1.2–5).

## Second throat

**Two sonic throats in series.** Through a sonic throat m = Γ p0 A* / √(R T0).
Between the nozzle throat (1) and the second throat (2) T0 is constant but p0
falls (friction, model drag, shocks), so both can be sonic only if
A*₂/A*₁ = p0₁/p0₂ > 1. With **equal areas the second throat chokes first and
the nozzle throat cannot be sonic**. The tool computes three critical areas
(friction and model drag included):

| Area | Meaning |
|---|---|
| A_sub | below it the second throat controls the mass flow: tunnel subsonic (R0) |
| A_run | smallest area through which the started flow passes the whole second throat (friction inside the throat included): running limit; below it the normal shock stays locked in the nozzle/test section (R1) |
| A_start | smallest area that swallows the starting shock: below it fixed walls cannot start (R2: variable walls needed); above it the tunnel starts with fixed walls (R3) |

It then reports, in `05_second_throat/`:

* **regime of the design and of equal areas**, with the reasons and
  recommendations (`summary.txt`, report);
* **variable second throat**: open to the fixed design area for the start,
  closed to A_run (1 + `second_throat.run_margin`) for the run. Gain in
  running pressure ratio, mass flow and run time, checked with an Euler
  transient in which the walls move (energy source −p ∂A/∂t);
* **sensitivity**: regime, PR_start, PR_run, throat Mach, locked-shock Mach
  and run time versus A_d2/A_t;
* **ramp convergent**: planar estimate of the oblique shock pair at the
  convergent entrance, compared with the isentropic convergent; it detects
  Mach reflection (a convergent too steep for the entrance Mach, e.g. when
  movable walls are closed over a short length) and gives the largest
  half-angle with a regular reflection.

**Model drag.** The drag D = Cd q A_m is a momentum sink; for the quasi-1D
total-pressure balance it equals a Fanno step with f L/D = Cd A_m/A_ts at the
model station (fully mixed-out loss). Cd is an input or, with `cd = 0`,
forebody pressure from cone/wedge shock theory plus base pressure
Cp_base = −1/M² (rough rule, about 70 % of the vacuum limit), skin friction
neglected.

## Results folder

```
results/M2.00_D150mm_T0-288K_pa101kPa_Vt2.00m3_pt20.0bar_<hash>/
├── README.txt              index: every file with a one-line description
├── summary.txt             key results, grouped, with units
├── summary.json            same content, machine-readable
├── inputs_used.py          all inputs of this run (re-runnable)
├── manifest.json           input SHA-256, timestamp, version, file list
├── 01_geometry/            wall profile (CSV for CAD), meridian and 3-D views
├── 02_steady_flow/         axial distributions (start, running), loss budget
├── 03_shock_analysis/      hysteresis map, start-up sequence, model shock
├── 04_nozzle_moc/          wall contour, characteristic net, Mach field
├── 05_second_throat/       critical areas, regimes, variable geometry, sensitivity, ramp
├── 06_transient_euler/     fixed and variable throat start-up/unstart
├── 07_transient_blowdown/  tank, regulator, run time
├── 08_optimisation/        variables, history, optimised profile
├── 09_mach_sweep/          trade study
├── paraview/               steady/, nozzle/, transient_fixed_throat/, transient_variable_throat/
└── report/                 report.html (figures inline), report.pdf
```

The folder name is built from the problem data plus the first 8 hex digits of
the SHA-256 of the full canonical input: the same input always maps to the
same folder (a re-run replaces it), any parameter change gives a new folder.
CSV headers carry units in brackets, e.g. `x [m]`, `static pressure [Pa]`.
`results/` is ignored by git.

## ParaView

| File | Content |
|---|---|
| `steady/steady_running.vts` | whole tunnel revolved around the axis, running condition; inside the nozzle the MOC Mach field |
| `steady/steady_starting.vts` | starting condition |
| `steady/steady_running_closed_throat.vts` | running condition with the variable throat closed |
| `nozzle/moc_nozzle.vts`, `nozzle/moc_characteristics.vtp` | MOC field; C+ and C− lines in the meridian plane |
| `transient_fixed_throat/frame.pvd` | Euler time series, fixed throat |
| `transient_variable_throat/frame.pvd` | Euler time series, the walls of the second throat move |

Open a `.pvd` and press Play. Arrays: `mach`, `pressure_Pa`, `temperature_K`,
`density_kg_m3`, `total_pressure_Pa`, `p_over_p0_supply`, `velocity`. Files
are VTK XML, zlib-compressed by default (`output.vtk_compress`).

## Method in short

**Sizing.** A* = A_ts/(A/A*)(M). The starting shock must sweep from the throat
to the end of the test section: the starting ratio is the largest p0/p_a over
those positions. Running: supersonic flow through the second throat, normal
shock in the diffuser where the exit static pressure equals p_a; PR_run is the
minimum over the admissible shock positions.

**Quasi-1D steady solver.** Isentropic area change on the current branch,
Fanno flow in constant-area ducts (Darcy factor from the Haaland formula at
the local Reynolds number), K q losses, model drag, shock pair (option),
normal shocks; losses carried through A*_local = A* p0_ref/p0_local.

**MOC nozzle.** Source-flow kernel up to a wall point on a cone; the
source-flow C− through it is integrated analytically to the axis and is the
initial line of the lattice. Downstream the centreline Mach is a quintic that
matches M, M′, M″ of the source flow and reaches the exit Mach with zero slope
and curvature. Predictor-corrector unit process with the axisymmetric
compatibility relations; the wall is the stream surface carrying the throat
mass flow.

**Euler transient.** Conservative quasi-1D Euler with p ∂A/∂x source, wall
friction in constant-area ducts, K q sinks for screens, model and diffuser;
moving walls add −p ∂A/∂t to the energy equation. Stagnation inlet with the
outgoing Riemann invariant, pressure outlet.

**Blowdown.** Polytropic tank, regulator holding p0, isenthalpic throttling
(T0 = T_tank), choked mass flow, RK4 in time, start/unstart hysteresis.

**Optimisation.** Sobol sampling of the design box followed by SLSQP.

## Repository layout

```
inputs.py                 input file: every parameter with unit and description
run.py                    launcher: python run.py [input file] [options]
examples/                 mach3.py (Python input), mach2.toml (optional TOML format)
src/windtunnel/
  config.py               parameters (dataclasses with units), Python/TOML loading, hashing
  gas.py                  perfect gas, Sutherland viscosity
  isentropic.py           isentropic relations, Prandtl-Meyer, mass flow
  shocks.py               normal, oblique, conical (Taylor-Maccoll) shocks; shock in nozzle; shock pair
  fanno.py                Fanno flow, friction factor
  moc.py                  axisymmetric method of characteristics (nozzle contour)
  geometry.py             tunnel profile r(x); variable second throat
  steady.py               quasi-1D steady solver, starting/running ratios, hysteresis
  design.py               sizing pipeline, model drag, loss budget
  second_throat.py        sonic compatibility, regimes, variable geometry, sensitivity, ramp
  euler1d.py              unsteady quasi-1D Euler solver, start-up scenarios, moving walls
  blowdown.py             tank and regulator, run time
  optimize.py             optimisation and Mach trade study
  losses.py               device and diffuser losses, liquefaction
  plots.py                SVG figures
  vtk_io.py, paraview.py  VTK XML writers and ParaView exports
  report.py               HTML and PDF report
  outputs.py              results folder, file index, summary text
  cli.py                  command line and run sequence
tests/                    pytest suite
```

## Development

```bash
pytest -q                 # 57 tests, about 1.5 minutes
python run.py --quick     # end-to-end smoke run
```

## Validation (tests)

`pytest -q` runs 57 tests (about 1.5 minutes), among them:

* isentropic, normal-shock, oblique-shock and Fanno values against NACA 1135;
* MOC: the lattice reproduces analytic source flow (|ΔM| < 10⁻³); exit area
  within 0.5 % of A/A*; monotone wall turning; grid convergence;
* steady solver: mass conservation to 10⁻⁶; frictionless starting ratio equal
  to the analytic normal-shock value; hysteresis branches;
* second throat: A_t < A_sub < A_run < A_start; equal areas fall in R0; the
  model loss equals the Fanno step f L/D = Cd A_m/A_ts; the model raises
  PR_start, PR_run and A_start; variable walls lower PR_run and lengthen the
  run; oblique shock pair vs normal shock and reflection limit;
* Euler: Sod shock tube against the exact Riemann solution; steady nozzle shock
  within two cells; start and unstart in the tunnel; moving walls compress a
  gas at rest isentropically (p ∝ ρ^γ, ρ ∝ 1/A);
* blowdown mass balance; isothermal tank gives a longer run than adiabatic;
* VTK files decoded back and read with the VTK library when installed;
* run folders: same input → same folder, any change → new folder; Python and
  TOML inputs round-trip; end-to-end run producing the full folder structure.

## Limitations

* Calorically perfect air.
* Quasi-1D: no boundary-layer displacement correction of the nozzle contour,
  no shock/boundary-layer interaction, no shock train (a single normal shock).
* Steady model: friction only in constant-area ducts.
* Model drag as a mixed-out momentum loss; automatic Cd is a rough estimate
  (shock theory + base-pressure rule, no skin friction).
* Oblique shocks in the convergent: planar shock-pair estimate; the Euler
  solver cannot represent oblique shocks.
* Liquefaction: Clausius-Clapeyron for the N2 and O2 partial pressures,
  without supersaturation (conservative).
* The subsonic diffuser correlation is incompressible (Idelchik-type); check
  it against data, or set `diffuser.K_override`.
* The throat region is approximated by the source flow, as in the classic
  kernel methods; a transonic throat solution is not included.
