"""Wind tunnel inputs.

Edit the values below and run:

    python run.py                  # uses this file
    python run.py --quick          # coarse resolution, fast check

Every parameter is listed with its unit and meaning. At the start you are
asked for a simulation name; results go to results/<name>/.
Other input files with the same layout can be passed to run.py, e.g.
    python run.py examples/mach3.py
"""

from windtunnel.config import (
    TunnelSpec, GasSpec, DesignSpec, SettlingSpec, ContractionSpec, NozzleSpec, TestSectionSpec,
    ModelSpec, DiffuserSpec, SecondThroatSpec, TankSpec, RunSpec, TransientSpec, OptimisationSpec,
    SweepSpec, OutputSpec,
)

SPEC = TunnelSpec(
    # Working gas (calorically perfect).
    gas=GasSpec(
        gamma=1.4,  # [-] ratio of specific heats
        R=287.05,   # [J/(kg K)] gas constant
    ),
    # Design point.
    design=DesignSpec(
        mach=2.0,                    # [-] test-section Mach number
        test_section_diameter=0.15,  # [m] test-section diameter
        T0=288.15,                   # [K] stagnation temperature (initial tank temperature)
        p_ambient=101325.0,          # [Pa] exhaust back pressure
        T_ambient=288.15,            # [K] ambient temperature (initial state of the tunnel)
    ),
    # Settling chamber with honeycomb and screens.
    settling=SettlingSpec(
        mach=0.05,          # [-] settling-chamber Mach number (sets the contraction ratio)
        length_factor=1.0,  # [-] length / settling-chamber diameter
        K_honeycomb=0.5,    # [-] honeycomb loss coefficient, dp0 = K q (typical value, use hardware data)
        K_screens=1.5,      # [-] sum of the screen loss coefficients (typical value, use hardware data)
    ),
    # Subsonic contraction.
    contraction=ContractionSpec(
        length_factor=1.0,  # [-] length / settling-chamber diameter
    ),
    # Supersonic nozzle (method of characteristics).
    nozzle=NozzleSpec(
        inflection_angle_deg=0.0,    # [deg] cone half-angle of the source-flow kernel, 0 = automatic
        throat_curvature_ratio=2.0,  # [-] throat wall radius of curvature / throat radius
        n_axis=120,                  # [-] MOC resolution: centreline points
        length_factor=1.0,           # [-] >= 1 stretches the centreline Mach distribution
        nu_fraction=0.75,            # [-] nu on the axis at the end of the source region / nu(exit Mach)
    ),
    # Test section.
    test_section=TestSectionSpec(
        length_factor=1.5,   # [-] length / test-section diameter
        wall_roughness=0.0,  # [m] equivalent sand roughness of the walls
    ),
    # Test model (blockage and drag).
    model=ModelSpec(
        frontal_area=0.0015,  # [m^2] model frontal area, 0 = empty test section
        shape="cone",         # [-] 'cone' or 'wedge' (model shock and automatic drag coefficient)
        half_angle_deg=10.0,  # [deg] cone or wedge half-angle
        cd=0.0,               # [-] drag coefficient on frontal area in supersonic flow, 0 = automatic (shock theory + base pressure)
        cd_subsonic=-1.0,     # [-] drag coefficient behind the starting shock, < 0 = same as cd
        position=0.4,         # [-] model station as a fraction of the test-section length
    ),
    # Second throat and subsonic diffuser (fixed geometry of the design).
    diffuser=DiffuserSpec(
        second_throat_margin=0.1,         # [-] fixed second throat: A_d2 = A_d2,start (1 + margin)
        convergent_half_angle_deg=4.0,    # [deg] half-angle of the second-throat convergent
        convergent_model="isentropic",    # [-] 'isentropic' or 'oblique_shocks' (shock pair at the convergent entrance, ramp diffuser)
        second_throat_length_factor=2.0,  # [-] second-throat length / its diameter
        subsonic_half_angle_deg=3.0,      # [deg] subsonic diffuser half-angle
        area_ratio=3.0,                   # [-] subsonic diffuser exit area / second-throat area
        K_override=-1.0,                  # [-] >= 0 replaces the subsonic diffuser loss correlation
    ),
    # Second-throat studies: sonic compatibility, variable geometry, sensitivity.
    second_throat=SecondThroatSpec(
        enabled=True,            # [-] run the second-throat studies
        variable_geometry=True,  # [-] study a variable second throat (open to start, closed to run)
        run_margin=0.05,         # [-] closed throat: A_d2 = A*_run (1 + run_margin)
        close_time=0.05,         # [s] Euler transient: closing time of the variable throat
        n_points=14,             # [-] sensitivity study: number of second-throat areas
        area_min_factor=0.9,     # [-] sensitivity study: smallest area / A*_run
        area_max_factor=1.3,     # [-] sensitivity study: largest area / A_d2,start
    ),
    # Pressure tank and regulator.
    tank=TankSpec(
        volume=2.0,            # [m^3] tank volume
        pressure=2000000.0,    # [Pa] initial tank pressure
        polytropic_n=1.4,      # [-] tank expansion exponent: 1.4 adiabatic, 1.0 isothermal
        regulator_margin=0.1,  # [-] minimum p_tank / p0 - 1 for regulation
    ),
    # Operating schedule of a blowdown run.
    run=RunSpec(
        start_margin=0.1,                  # [-] starting set point: p0 = PR_start p_a (1 + start_margin)
        run_margin=0.05,                   # [-] running set point: p0 = PR_run p_a (1 + run_margin)
        p0_schedule="reduce_after_start",  # [-] 'reduce_after_start' or 'constant'
        dt=0.01,                           # [s] time step of the blowdown integration
        t_max=600.0,                       # [s] maximum simulated time
    ),
    # Unsteady quasi-1D Euler start-up transient.
    transient=TransientSpec(
        enabled=True,         # [-] run the Euler transient
        n_cells=400,          # [-] number of cells
        cfl=0.7,              # [-] CFL number
        valve_time=0.04,      # [s] regulator opening time (linear p0 ramp)
        hold_time=0.08,       # [s] hold time at each set point
        unstart_demo=True,    # [-] final ramp below the running ratio (unstart)
        ramp_down_time=0.05,  # [s] duration of each p0 reduction
        n_snapshots=60,       # [-] number of saved snapshots
        friction=True,        # [-] wall friction in constant-area ducts
        limiter="minmod",     # [-] 'minmod' or 'van_leer'
    ),
    # Optimisation of the second throat and of the subsonic diffuser.
    optimisation=OptimisationSpec(
        enabled=True,                     # [-] run the optimisation
        max_total_length=3.2,             # [m] maximum tunnel length, 0 = no limit
        max_subsonic_half_angle_deg=8.0,  # [deg] upper bound of the diffuser half-angle
        min_second_throat_margin=0.05,    # [-] lower bound of the second-throat margin
        max_second_throat_margin=0.4,     # [-] upper bound of the second-throat margin
    ),
    # Mach-number trade study.
    sweep=SweepSpec(
        enabled=True,  # [-] run the Mach trade study
        mach_min=1.5,  # [-] smallest Mach number
        mach_max=4.0,  # [-] largest Mach number
        n=11,          # [-] number of Mach numbers
    ),
    # Output options.
    output=OutputSpec(
        vtk_n_theta=48,            # [-] ParaView revolved grids: points around the axis
        vtk_n_r=8,                 # [-] ParaView revolved grids: points from axis to wall
        vtk_transient_n_theta=24,  # [-] transient series: points around the axis
        vtk_transient_n_r=4,       # [-] transient series: points from axis to wall
        vtk_compress=True,         # [-] zlib compression of the VTK files
        pdf=True,                  # [-] write the PDF report
    ),
)
