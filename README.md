# Wind Tunnel Design

A Python tool for designing supersonic blowdown wind tunnels.

You give it a target Mach number, a test-section size and an air tank. It sizes every part of the tunnel, shapes the nozzle with the method of characteristics, finds the pressures needed to start and run it, simulates the start-up and the whole run, and saves everything in one folder: tables, figures, ParaView files and a PDF report.

## The tunnel

A blowdown tunnel stores air in a high-pressure tank and releases it through the test section into the atmosphere. On its way out, the air goes through:

```
tank → regulator → settling chamber → contraction → nozzle → test section → second throat → diffuser → atmosphere
```

The tunnel is axisymmetric (round cross-section) and open-circuit. It is meant for test-section Mach numbers between 1.5 and 4; values from 1.2 to 5 are accepted.

## Example

The default inputs describe a Mach 2 tunnel with a 150 mm test section, a 2 m³ tank at 20 bar and a 10° cone model. A full run gives, among many other results:

| Result | Value |
|---|---|
| Nozzle throat diameter | 115.5 mm |
| Total tunnel length | 2.67 m |
| Pressure ratio needed to start (p0/p_a) | 1.49 |
| Pressure ratio needed to keep running (p0/p_a) | 1.41 |
| Mass flow while running | 3.73 kg/s |
| Run time | 9.6 s |
| Run time with a variable second throat | 11.4 s |

## Getting started

You need Python 3.11 or newer.

```bash
git clone https://github.com/PSamuele/Wind_Tunnel_Design.git
cd Wind_Tunnel_Design
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e .                 # installs numpy, scipy and matplotlib
```

Then run it:

```bash
python run.py
```

The first thing it asks is a name for the simulation:

```
What would you like to name this simulation? mach2_baseline
```

The name becomes the results folder, here `results/mach2_baseline/`. Use letters, numbers and underscores; spaces are turned into underscores for you. If a simulation with that name already exists, you are asked whether to overwrite it.

When it finishes, the console prints a summary and the path of the results folder. A full run takes about 4 to 7 minutes, mostly spent on the transient simulations, the second-throat studies and the optimisation. For a fast check at coarse resolution (about 2 minutes):

```bash
python run.py --quick
```

`run.py` also works without installing the package, as long as numpy, scipy and matplotlib are available. After `pip install -e .` the same program can also be started with the `windtunnel` command.

| Option | What it does |
|---|---|
| `--name NAME` | name the simulation without being asked (an existing folder with that name is replaced) |
| `--quick` | coarse resolution, for a fast check |
| `--results-root DIR` | save results under `DIR` instead of `results` |
| `--no-transient` | skip the transient (Euler) simulations |
| `--no-second-throat` | skip the second-throat studies |
| `--no-optimisation` | skip the optimisation |
| `--no-sweep` | skip the Mach number trade study |

## Setting your own inputs

All inputs are in [`inputs.py`](inputs.py), grouped by tunnel component. Every parameter comes with its unit and a short note:

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

Change the values, save the file and run `python run.py`. To keep several designs side by side, make a copy of the file and pass it to the launcher:

```bash
python run.py examples/mach3.py
```

A few things worth knowing:

* every run saves its inputs as `inputs_used.py` in the results folder, so any run can be repeated exactly;
* a misspelled parameter name stops the run with an error, so a typo is never silently ignored;
* TOML works too: see [`examples/mach2.toml`](examples/mach2.toml), which uses the same sections and names and only lists the values that differ from the defaults.

## What you get

Each run gets its own folder inside `results/`:

```
results/<simulation name>/
├── README.txt              list of every file, one line each
├── summary.txt             key results with units
├── summary.json            the same, for scripts
├── inputs_used.py          the inputs of this run
├── manifest.json           simulation name, run id, input hash, date, version, file list
├── 01_geometry/            wall profile (CSV, ready for CAD), side and 3D views
├── 02_steady_flow/         flow along the axis (start and running), loss budget
├── 03_shock_analysis/      start/unstart map, start-up sequence, shock on the model
├── 04_nozzle_moc/          nozzle contour, characteristic net, Mach field
├── 05_second_throat/       critical areas, regimes, variable geometry, sensitivity, ramp
├── 06_transient_euler/     start-up and unstart, fixed and variable throat
├── 07_transient_blowdown/  tank pressure, regulator, run time
├── 08_optimisation/        design variables, history, optimised profile
├── 09_mach_sweep/          trade study over Mach number
├── paraview/               files to open in ParaView
└── report/                 report.html and report.pdf
```

The best places to start are `summary.txt` and `report/report.pdf`.

`summary.txt` and `manifest.json` also record a run id, made of the main design data plus the first 8 characters of a hash of all the inputs. Two runs with the same run id used exactly the same inputs, whatever name they were given. CSV column headers include units, for example `x [m]` or `static pressure [Pa]`. The `results/` folder is not tracked by git.

## What it calculates

| Topic | Details |
|---|---|
| Sizing | nozzle throat, settling chamber, test section, second throat and diffuser; mass flow; test-section conditions, unit Reynolds number, test rhombus length |
| Nozzle | axisymmetric contour from the method of characteristics, with uniform flow at the exit |
| Steady flow | Mach number, pressures, temperature, velocity and density along the tunnel axis, for start and running conditions |
| Losses | honeycomb and screens, wall friction, model drag, optional oblique shocks in the second-throat convergent, normal shock, subsonic diffuser, exhaust |
| Shocks | pressure ratios to start and to run, start/unstart hysteresis, Kantrowitz blockage limit, shock on the model (cone or wedge) |
| Second throat | throat compatibility and operating regimes, variable geometry, sensitivity to the throat area, ramp convergent (see below) |
| Start-up transient | quasi-1D unsteady Euler solver: start, running and unstart, with a fixed throat and with moving throat walls |
| Blowdown run | tank emptying through the regulator, run time, drift of stagnation temperature and Reynolds number, risk of air liquefaction |
| Optimisation | second-throat margin and length, diffuser angle and area ratio, chosen to lower the running pressure ratio within length and tank-pressure limits |
| Trade study | pressure ratios, run time, mass flow and liquefaction limit across a range of Mach numbers |

## The second throat

After the test section the flow passes through a second narrowing before the diffuser. Its size decides whether the tunnel can start at all and how much pressure it needs once it is running, so the tool studies it in detail.

The same mass flow passes through both throats, but total pressure drops between them because of friction, model drag and shocks. For both throats to be sonic, the second one must be larger than the first by the ratio of the total pressures (A₂/A₁ = p0₁/p0₂). If the two are the same size, the second throat chokes first, the nozzle throat never reaches sonic speed, and the test section stays subsonic.

The tool works out three critical areas for the second throat, with friction and model drag included:

| Area | Meaning |
|---|---|
| A_sub | below this, the second throat limits the mass flow and the tunnel stays subsonic |
| A_run | the smallest area the supersonic flow can pass through once the tunnel is running |
| A_start | the smallest area that lets the starting shock through with fixed walls |

These areas split the possible designs into four regimes:

| Regime | Second-throat area | What happens |
|---|---|---|
| R0 | below A_sub | the tunnel stays subsonic |
| R1 | between A_sub and A_run | the normal shock stays stuck in the nozzle or test section |
| R2 | between A_run and A_start | the tunnel can run but cannot start with fixed walls; movable walls are needed |
| R3 | above A_start | the tunnel starts and runs with fixed walls |

The results in `05_second_throat/` cover:

* the regime of your design and of the equal-area case, with recommendations (also in `summary.txt` and the report);
* a variable second throat: open at the design area to start, then closed to just above A_run for the run. It reports how much this lowers the running pressure ratio and the mass flow and how much longer the run lasts, and checks the result with a transient simulation in which the walls actually move;
* a sensitivity study: regime, pressure ratios, throat Mach number and run time as the throat area changes;
* a ramp convergent check: a planar estimate of the oblique shocks at the convergent entrance. It warns when the convergent is too steep and the shocks would form a Mach reflection (for example when movable walls close over a short length) and gives the largest safe half-angle.

### Model drag

The model's drag, D = Cd q A_m, is treated as a total-pressure loss equal to a friction step with f L/D = Cd A_m/A_ts at the model position. You can give Cd yourself, or set `cd = 0` to have it estimated from cone or wedge shock theory plus a base-pressure rule (Cp_base = −1/M², roughly 70 % of the vacuum limit). Skin friction on the model is not included.

## Viewing results in ParaView

| File | Content |
|---|---|
| `steady/steady_running.vts` | the whole tunnel, revolved around its axis, in running conditions (MOC Mach field inside the nozzle) |
| `steady/steady_starting.vts` | the same, in starting conditions |
| `steady/steady_running_closed_throat.vts` | running conditions with the variable throat closed |
| `nozzle/moc_nozzle.vts`, `nozzle/moc_characteristics.vtp` | nozzle Mach field and the characteristic lines |
| `transient_fixed_throat/frame.pvd` | start-up animation, fixed throat |
| `transient_variable_throat/frame.pvd` | start-up animation, moving throat walls |

Open a `.pvd` file and press Play. Available fields: `mach`, `pressure_Pa`, `temperature_K`, `density_kg_m3`, `total_pressure_Pa`, `p_over_p0_supply` and `velocity`. The files are compressed VTK XML; compression can be turned off with `output.vtk_compress`.

## How it works

**Sizing.** The nozzle throat comes from the isentropic area ratio, A* = A_ts / (A/A*)(M). To start, the starting shock has to travel from the nozzle throat to the end of the test section; the starting pressure ratio is the highest p0/p_a needed along that path. In running conditions the flow stays supersonic through the second throat and a normal shock sits in the diffuser, where the exit pressure matches ambient; the running pressure ratio is the lowest value over the possible shock positions.

**Steady flow.** Quasi-1D: isentropic flow through area changes, Fanno flow in constant-area ducts (Haaland friction factor at the local Reynolds number), K·q losses for devices, model drag, optional oblique shock pair, and normal shocks. Losses are carried through the local critical area, A*_local = A* p0_ref / p0_local.

**Nozzle.** Near the throat the flow is modelled as source flow, up to a wall point on a cone. The characteristic through that point, integrated analytically to the axis, is the starting line. Downstream, the centreline Mach number follows a fifth-degree polynomial that matches the source flow and reaches the exit Mach number with zero slope and zero curvature. The characteristic net is solved with a predictor-corrector scheme, and the wall is the streamline that carries the throat mass flow.

**Start-up transient.** Conservative quasi-1D Euler equations (HLLC flux, MUSCL reconstruction, SSP-RK3 time stepping) with the pressure-area source term, wall friction in constant-area ducts and K·q losses for screens, model and diffuser. Moving walls add a −p ∂A/∂t term to the energy equation. The inlet is a stagnation condition, the outlet a fixed pressure.

**Blowdown.** Polytropic tank, regulator holding p0 constant, isenthalpic throttling (T0 = T_tank), choked mass flow, RK4 time integration, with the start/unstart hysteresis taken into account.

**Optimisation.** Sobol sampling over the design space, followed by SLSQP.

## Tests

```bash
pip install -e .[dev]     # adds pytest
pytest -q                 # 75 tests, about 1.5 minutes
```

The tests check, among other things:

* isentropic, normal-shock, oblique-shock and Fanno values against NACA Report 1135;
* the nozzle: the characteristic net reproduces the exact source flow (Mach error below 10⁻³), the exit area is within 0.5 % of A/A*, the wall angle changes monotonically, and the result converges with grid refinement;
* the steady solver: mass is conserved to 10⁻⁶, the frictionless starting ratio matches the exact normal-shock value, and the hysteresis branches are correct;
* the second throat: A_t < A_sub < A_run < A_start; equal areas fall in R0; the model loss equals the matching friction step; a model raises the starting and running ratios and A_start; movable walls lower the running ratio and extend the run; oblique shocks are compared with the normal shock and the reflection limit;
* the Euler solver: Sod shock tube against the exact solution, a steady nozzle shock placed within two cells, tunnel start and unstart, and moving walls that compress still gas isentropically;
* the blowdown mass balance, and a longer run with an isothermal tank than with an adiabatic one;
* VTK files read back correctly (also with the VTK library, if installed);
* run ids: same inputs give the same id, any change gives a new one; simulation names are cleaned and checked, and the name question handles invalid names and existing folders; Python and TOML inputs load the same way; a full end-to-end run produces the complete folder.

## Limitations

* Air is treated as a calorically perfect gas.
* The flow model is quasi-1D: no boundary-layer correction of the nozzle contour, no shock/boundary-layer interaction, and a single normal shock instead of a shock train.
* In the steady model, friction is only applied in constant-area ducts.
* Model drag is treated as a fully mixed-out loss, and the automatic Cd is a rough estimate.
* Oblique shocks in the convergent use a planar estimate; the Euler solver cannot represent them.
* Liquefaction is checked with the Clausius-Clapeyron relation for nitrogen and oxygen, ignoring supersaturation, so the check is conservative.
* The subsonic diffuser loss uses an incompressible correlation (Idelchik type). Check it against real data, or set your own value with `diffuser.K_override`.
* Near the nozzle throat the flow is approximated by source flow, as in classic kernel methods; there is no transonic throat solution.

## Project layout

```
inputs.py                 input file: every parameter with unit and description
run.py                    launcher: python run.py [input file] [options]
examples/                 mach3.py (Python input), mach2.toml (TOML input)
src/windtunnel/
  config.py               parameters with units, Python/TOML loading, hashing
  gas.py                  perfect gas, Sutherland viscosity
  isentropic.py           isentropic relations, Prandtl-Meyer, mass flow
  shocks.py               normal, oblique and conical (Taylor-Maccoll) shocks
  fanno.py                Fanno flow, friction factor
  moc.py                  axisymmetric method of characteristics (nozzle contour)
  geometry.py             tunnel profile r(x), variable second throat
  steady.py               quasi-1D steady solver, starting/running ratios, hysteresis
  design.py               sizing, model drag, loss budget
  second_throat.py        throat compatibility, regimes, variable geometry, sensitivity, ramp
  euler1d.py              unsteady quasi-1D Euler solver, start-up, moving walls
  blowdown.py             tank and regulator, run time
  optimize.py             optimisation and Mach trade study
  losses.py               device and diffuser losses, liquefaction
  plots.py                SVG figures
  vtk_io.py, paraview.py  VTK writers and ParaView exports
  report.py               HTML and PDF report
  outputs.py              results folder, file index, summary
  cli.py                  command line and run sequence
tests/                    pytest suite
```
