# Hydrodynamic Function Matrix Generator

## Installation

1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Clone this git repo to your local machine
3. Enter project directory `cd stokeslet_integration`
4. Run `poetry install`

## Options

List all options by running `poetry run stokeslet_integration -h`:

```
Efficiently computes the hydrodynamic function matrix for a MEMS resonator.

options:
  -h, --help                   show this help message and exit
  -x PARTITIONS                horizontal partitions towards right edge, e.g., -x 3,5,3 means "create 3 equally sized
                               partitions, refine last partiton into 5 partitions, refine last partition (of those 5
                               partitions) further into 3 partitions". This results in the following unit lengths for
                               the mesh elements [15, 15, 3, 3, 3, 3, 1, 1, 1]. Single partition setting can also be
                               used, e.g., -x 101, although the size of the hydrodynamic function matrix will be greatly
                               increased. Note: all partition values must be ODD! (type: list[int])
  -y PARTITIONS                vertical partitions towards BOTH edges. Same principle as -x setting, except that the
                               setting gets mirrored onto both edges. Note: all partition values must be ODD! (type:
                               list[int])
  -l LENGTH                    length of the resonator plate [m] (type: float, default: 0.0005)
  -b WIDTH                     width of the resonator plate [m] (type: float, default: 5e-05)
  -t THRESHOLD                 relative error threshold for Stokeslet integrals (type: float, default: 1e-06)
  -f FREQUENCY                 resonator frequency [Hz] (type: float, default: 1000.0)
  --mu VISCOSITY               viscosity of the surrounding medium [Pa*s] (type: float, default: 0.00089)
  --rho DENSITY                density of the surrounding medium [kg/m^3] (type: float, default: 997)
  --arc POINTS                 # of points for defining the arc of the circle segment (more points -> smoother arc)
                               (type: int, default: 25)
  --segment-triangles INITIAL  initial # of triangles for meshing the circle segment (type: int, default: 100)
  --segment-quality FACTOR     each time integral convergence for a circle segment is not achieved, # of triangles will
                               be multiplied by this factor (type: float, default: 1.3)
  --mem-priority               if this flag is set, radial symmetry is used during hydrodynamic function matrix assembly
                               by mirroring indices (slower than first expanding to whole padded mesh by flipping +
                               concatenating horizontally and vertically, and using NumPy's fancy indexing on the full
                               padded mesh matrix, but uses less RAM) (type: bool, default: False)
  -p                           if this flag is set, v=Sp will be solved for p with v=1 (type: bool, default: False)
  --plot                       (type: bool, default: False)
```

## Getting started

For example, try running `poetry run stokeslet_integration -x 201 -y 25 --plot -p` or `poetry run stokeslet_integration -x 101,5 -y 25,5,3 --plot -p`.
