V-10.30_alpha-0_RPMs-6700-5754-3705 .......... Case Name
1 ............................................ Case ID [0: Wind turbine; 1: Propeller]
0 ............................................ Solver ID [0: BEMT, 1: LLT]
# =================== XFOIL info ============
9.0 .......................................... Ncrit
16.0 ......................................... alpha lim [-x, x]
50000.0 ...................................... Re start
450000.0 ..................................... Re end
50000.0 ...................................... Re increment
# =================== ROTOR DATA ============
3 ............................................ Number of rotors
# ---- rotor 1 / forward hover rotor ----
0.1524 ....................................... Rotor radius [m]
2 ............................................ Number of blades [-]
0.16 ......................................... Blade start at [r/R]
0.75 ......................................... Blade reference pitch location [r/R]
0.0 .......................................... Blade pitch angle [deg]
0.0 .......................................... Rotor yaw angle [deg]
41 ........................................... Number of blade radial sections
12x55MR_bladeGeom.txt ........................ blade geometry file name
12x55MR_Cl_mesh.dat .......................... Cl mesh data from XFoil
12x55MR_Cd_mesh.dat .......................... Cd mesh data from XFoil
-6700.0 ...................................... Rotor RPM
left_handed .................................. Blade handedness [right_handed|left_handed] (optional)
-0.3274 ...................................... Rotor hub offset x from reference hub [m]
0.7033 ....................................... Rotor hub offset y from reference hub [m]
0.0265 ....................................... Rotor hub offset z from reference hub [m]
0.0 .......................................... Rotor axis x
0.0 .......................................... Rotor axis y
-1.0 ......................................... Rotor axis z
0.0 .......................................... Rotor e1 x
1.0 .......................................... Rotor e1 y
0.0 .......................................... Rotor e1 z
1.0 .......................................... Rotor e2 x
0.0 .......................................... Rotor e2 y
0.0 .......................................... Rotor e2 z
# ---- rotor 4 / aft hover rotor ----
0.1524 ....................................... Rotor radius [m]
2 ............................................ Number of blades [-]
0.16 ......................................... Blade start at [r/R]
0.75 ......................................... Blade reference pitch location [r/R]
0.0 .......................................... Blade pitch angle [deg]
0.0 .......................................... Rotor yaw angle [deg]
41 ........................................... Number of blade radial sections
12x55MR_bladeGeom.txt ........................ blade geometry file name
12x55MR_Cl_mesh.dat .......................... Cl mesh data from XFoil
12x55MR_Cd_mesh.dat .......................... Cd mesh data from XFoil
5754.0 ....................................... Rotor RPM
right_handed ................................. Blade handedness [right_handed|left_handed] (optional)
0.3524 ....................................... Rotor hub offset x from reference hub [m]
0.7033 ....................................... Rotor hub offset y from reference hub [m]
0.0363 ....................................... Rotor hub offset z from reference hub [m]
0.0 .......................................... Rotor axis x
0.0 .......................................... Rotor axis y
-1.0 ......................................... Rotor axis z
0.0 .......................................... Rotor e1 x
1.0 .......................................... Rotor e1 y
0.0 .......................................... Rotor e1 z
1.0 .......................................... Rotor e2 x
0.0 .......................................... Rotor e2 y
0.0 .......................................... Rotor e2 z
# ---- rotor 5 / pusher propeller ----
0.1905 ....................................... Rotor radius [m]
2 ............................................ Number of blades [-]
0.16 ......................................... Blade start at [r/R]
0.75 ......................................... Blade reference pitch location [r/R]
0.0 .......................................... Blade pitch angle [deg]
0.0 .......................................... Rotor yaw angle [deg]
41 ........................................... Number of blade radial sections
15x10E_bladeGeom.txt ......................... blade geometry file name
15x10E_Cl_mesh.dat ........................... Cl mesh data from XFoil
15x10E_Cd_mesh.dat ........................... Cd mesh data from XFoil
3705.0 ....................................... Rotor RPM
right_handed ................................. Blade handedness [right_handed|left_handed] (optional)
0.3910 ....................................... Rotor hub offset x from reference hub [m]
0.3513 ....................................... Rotor hub offset y from reference hub [m]
-0.0166 ...................................... Rotor hub offset z from reference hub [m]
1.0 .......................................... Rotor axis x
0.0 .......................................... Rotor axis y
0.0 .......................................... Rotor axis z
0.0 .......................................... Rotor e1 x
1.0 .......................................... Rotor e1 y
0.0 .......................................... Rotor e1 z
0.0 .......................................... Rotor e2 x
0.0 .......................................... Rotor e2 y
1.0 .......................................... Rotor e2 z
# =============== OPERATIONAL DATA ==========
10.30, 0.0, 0.0 .............................. Free-stream velocity vector [m/s] (x longitudinal, y lateral, z up; right-handed). |V|=10.3 at AoA=0.0 deg
# ============== ATMOSPHERIC DATA ===========
103064.21 .................................... Pressure [Pa]
293.15 ....................................... Temperature [K]
1.225 ........................................ Density [kg/m^3]
0.000018 ..................................... Dynamic viscosity [Pa*s]
70.0 ......................................... Relative humidity [%]
# ============== GPU LBM / ALM ==============
EVO-body.stl ................................. STL geometry file [use none for free rotor]
1 ............................................ Enable actuator line model [0/1]
1 ............................................ Use GPU voxelizer [0/1]
400 .......................................... Nx
301 .......................................... Ny
301 .......................................... Nz
0.1524 ....................................... Reference length for dx scaling [m]
26 ........................................... Reference length in lattice cells [-]
95.66149630180918 ............................ Reference velocity for dt scaling [m/s]
0.08 ......................................... Target lattice velocity [-]
1.1409 ....................................... Reference hub x-position [m]
0.7495 ....................................... Reference hub y-position [m]
1.1192 ....................................... Reference hub z-position [m]
1.0 .......................................... Reference rotor axis x
0.0 .......................................... Reference rotor axis y
0.0 .......................................... Reference rotor axis z
0.0 .......................................... Reference rotor e1 x
1.0 .......................................... Reference rotor e1 y
0.0 .......................................... Reference rotor e1 z
0.0 .......................................... Reference rotor e2 x
0.0 .......................................... Reference rotor e2 y
1.0 .......................................... Reference rotor e2 z
1200 ......................................... Total time steps
100 .......................................... Diagnostics interval [steps]
1 ............................................ Write flow visualization files [0/1]
100 .......................................... Output interval [steps]
100 .......................................... Output start step
0 ............................................ Start step
1000 ......................................... Force ramp steps
1.0 .......................................... ALM force scale
inverse_distance ............................. Force spread method [nearest|inverse_distance|gaussian]
0.25 ......................................... Gaussian epsilon [lattice]
2.0 .......................................... IDW beta
trilinear .................................... ALM velocity sampling mode
0.0 .......................................... ALM velocity sampling epsilon [lattice]
0.5 .......................................... ALM IDW radius factor x chord
1.0 .......................................... ALM IDW minimum radius [lattice]
0.6 .......................................... Density warning lower bound
1.4 .......................................... Density warning upper bound
0.2 .......................................... Density abort lower bound
2.2 .......................................... Density abort upper bound
0 ............................................ Generate Cl/Cd meshes with XFoil if missing [0/1]
inlet ........................................ Boundary mode for -x [inlet|outlet|periodic|pressure|open|pressure_sponge]
outlet ....................................... Boundary mode for +x [inlet|outlet|periodic|pressure|open|pressure_sponge]
pressure_sponge .............................. Boundary mode for -y [inlet|outlet|periodic|pressure|open|pressure_sponge]
pressure_sponge .............................. Boundary mode for +y [inlet|outlet|periodic|pressure|open|pressure_sponge]
pressure_sponge .............................. Boundary mode for -z [inlet|outlet|periodic|pressure|open|pressure_sponge]
pressure_sponge .............................. Boundary mode for +z [inlet|outlet|periodic|pressure|open|pressure_sponge]
12 ........................................... Lateral sponge width [cells]
0.08 ......................................... Lateral sponge strength [0..1]
# ============== FW-H DATA SURFACES =========
# Multi-rotor note: with 2R hub spacing, per-rotor FW-H surfaces larger than 1R overlap, and the current 144x144 lateral box is too small for outer surfaces.
0 ............................................ Enable FW-H surface sampling [0/1]
open_cylinder ................................ FW-H surface shape [sphere|cylinder|open_cylinder]
120 .......................................... FW-H surface azimuthal panels
150 .......................................... FW-H surface axial/polar panels
1.0 .......................................... FW-H cylinder half-length factor [R]
1 ............................................ FW-H sampling interval [steps]
0 ............................................ Write legacy SourceData.dat text [0/1]
FWH_multirotor ............................... FW-H output directory
2.0 .......................................... FW-H surface radius factor 1 [-]
2.4 .......................................... FW-H surface radius factor 2 [-]
2.8 .......................................... FW-H surface radius factor 3 [-]
equal_arc_cells .............................. FW-H azimuth resolution mode [legacy|equal_arc_cells]
1.25 ......................................... FW-H azimuth target arc length [cells]
5 ............................................ FW-H azimuth panel multiple [-]
3 ............................................ FW-H surface quadrature order [2|3|4]
2.0 .......................................... FW-H cylinder upstream length factor [R]
4.0 .......................................... FW-H cylinder downstream length factor [R]
# ========== INTEGRATED STATIONARY FD-FWH ===
0 ............................................ Enable integrated stationary FD-FWH acoustics [0/1]
mics_r4455_box256.txt ........................ Observer geometry file
340.75 ....................................... Acoustic sound speed [m/s]
0.0 .......................................... Mean-flow Mach x
0.0 .......................................... Mean-flow Mach y
0.0 .......................................... Mean-flow Mach z
1.0 .......................................... Acoustic recording start revolution [rev]
FD_FWH_multirotor ............................ Acoustic output directory
0 ............................................ Acoustic postprocess only [0/1]
none ......................................... Acoustic source data root [none or path]
auto ......................................... Acoustic replay density convention [auto|fluctuation|total]
8192 ......................................... Acoustic spectrum segment length N [0=auto]
0.5 .......................................... Acoustic spectrum shift fraction [0..1]
hann ......................................... Acoustic spectrum window [hann|hamm|bartlett|welch|blackman|triangular|rectangular]
1 ............................................ Acoustic spectrum mean subtract [0/1]
-1 ........................................... Acoustic startup guard samples [-1=auto]
20 ........................................... Native stationary FD harmonic count [0=auto, N=BPF harmonics]
# ============== FIXED WING =================
# A non-rotating lifting surface. This block is read by LABEL, not by position, so it can sit
# anywhere in the deck and a deck WITHOUT it parses exactly as before.
#
# Wing load model:
#   hs3d   Hess-Smith source+doublet solve with a Kutta condition at the trailing edge,
#          pressure-integrated. Default. The wing is the good case for this: a structured,
#          well-graded mesh, an unambiguous trailing edge, and it never moves -- so the
#          influence matrix factorises ONCE for the whole run.
#   polar  Blade-element strip loads from the Cl/Cd mesh files below, i.e. the same path the
#          rotors use in this driver. Use this with XFoil-generated tables when
#          LOPNOR_BEMTcpu is present; the tables shipped here are thin-airfoil theory on the
#          real camber lines plus a Viterna post-stall extension, good to roughly 5-10% on Cl
#          in the attached range and correlation-grade beyond it.
1 ............................................ Enable fixed wing [0/1]
wingGeom.txt ................................. Wing geometry file
120 .......................................... Wing chordwise points per section
hs3d ......................................... Wing load model [hs3d|polar]
wing_Cl_mesh.dat ............................. Wing Cl mesh [polar model only]
wing_Cd_mesh.dat ............................. Wing Cd mesh [polar model only]
# ============== AIRFRAME DISCRETISATION ====
# Loft the airframe from structured sections instead of reading the STL as triangles.
# Read by LABEL, so this can sit anywhere; "none" (or omitting the line) keeps the STL path.
#
# WHY: the STL gives one unstructured panel per triangle with i_span = -1, so there is no
# trailing edge to pair a Kutta condition on -- and on the shipped mesh three spanwise bands
# carried no lower-surface triangle at all within 12 mm of the wing TE. The sections file
# keeps point 0 of every section ON the trailing edge, so pairing is structural.
airframeSections.txt ......................... Airframe sections file [none to use the STL]
# ============== ROTOR LOAD MODEL ===========
# hs3d  = loads from the panel solve's own trailing-edge circulation jump (Kutta-Joukowski).
#         Requires enforce_kutta, which this driver had never run on the rotors before.
#         FIRST ATTEMPT BLEW UP: thrust ~1.9e5 N against a polar baseline of 7.6 N, i.e. mu
#         from the Kutta-constrained blade solve is diverging, not a sign error.
# polar = blade-element strips against 2D Cl/Cd tables. The panel solve still runs and still
#         shapes the wake; only the force integration differs.
hs3d ......................................... Rotor load model [hs3d|polar]
