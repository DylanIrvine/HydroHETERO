# HydroHETERO

Solute transport through random hydraulic conductivity fields.

Current version: 0.1.0

A Streamlit app that makes random 2D hydraulic conductivity (K) fields, and
optionally runs solute transport and particle tracking through them. The fields
can be multi-Gaussian, or have their high or low values connected (Zinn and
Harvey 2003). It is a numerical model for teaching, and a companion to
HydroTRANSPORT, which holds the analytical solutions for uniform aquifers.

## Using it

1. **Fields.** Choose the structure, geometric mean K, standard deviation of
   ln K, correlation length, the domain and cell size, and how many fields to
   make. Generate them, look through them, and download them as one zip. This
   is a complete use of the app on its own.
2. **Transport through every field (optional).** Set the gradient, porosity,
   local dispersivities, a source on the left edge with a history that can
   change in steps, and up to four control planes. The result is the spread of
   breakthrough curves over the fields, with a workbook that holds the time
   series on its first sheet and the settings on its second.
3. **Particle tracking in one field (optional).** Choose a field, release
   particles from a point, a line or a rectangle, with or without local
   dispersion, and get the paths and travel times as figures and CSV files.

## Field formats

- **Node list** (`K_by_node.csv`): one row per cell with node number, cell-centre
  x and y, and one K column per field. Node 1 is the cell at the lowest x and y.
  Numbering runs along x first or along y first, as chosen.
- **Grid** (`grid/K_001.csv`, ...): one comma-separated file per field, laid out
  as on a map, with the highest y on the first line.
- **MODFLOW** (`modflow/k_001.dat`, ...): one space-separated array file per
  field in the same row order, which is the order MODFLOW reads a layer. A note
  in the zip gives the OPEN/CLOSE lines for MODFLOW 6 (NPF) and MODFLOW-2005
  (LPF). These files have not been run through MODFLOW.

## Method

- **Fields.** ln K is normal with covariance exp[-(r/l)^2], made by filtering
  white noise in Fourier space on a grid twice the size and cropping. Each field
  is rescaled to the exact mean and standard deviation asked for. The connected
  fields use the Zinn and Harvey (2003) transform, which keeps the distribution
  of K and moves either the lowest or the highest values onto the connected
  network. The starting field is made with a correlation length 1.7 times
  longer, which brings the final 1/e length to within about 5% of the value
  asked for (found numerically).
- **Flow.** Cell-centred finite differences, harmonic mean K between cells,
  fixed heads on the left and right edges, closed sides, one sparse direct solve.
- **Transport.** Particles are released on the source line in proportion to
  flow. Velocity inside a cell varies linearly between its faces (Pollock 1988).
  Each step moves a particle 0.4 of a cell along the flow by a midpoint rule,
  then adds a random step for local dispersion and diffusion. The time each
  particle first crosses a control plane gives the flux-averaged breakthrough
  curve, and a source that changes in steps is a sum of shifted copies of it.
- **Uniform aquifer line.** Ogata and Banks (1961) with K equal to the
  geometric mean and a dispersivity of the local value plus the first-order
  macrodispersivity, variance of ln K times the integral scale
  (l x sqrt(pi) / 2).

## Checks

- Uniform K: flow is exact, the particle breakthrough matches Ogata and
  Banks, including a source with a step change, to within sampling error, and
  a particle released at a point arrives at distance over velocity.
- The node list and grid files hold each value at the stated position.
- Layers along the flow give the arithmetic mean K, and every cell conserves
  water.
- The fields have the mean, standard deviation and correlation length asked
  for, and the connected transform keeps the normal distribution.
- Connected high K fields conduct more than the geometric mean and connected
  low K fields less, as Zinn and Harvey (2003) found.
- In a multi-Gaussian field with a small variance, the mean arrival time
  matches the geometric mean K and the arrival-time variance matches first-order
  macrodispersion theory to within 25%.

```bash
pip install pytest
pytest tests/
```

## Known limits

- 2D only. Connected paths form more easily in 3D.
- Not compared with an established transport code (for example MODFLOW with
  MODPATH or MT3D).
- On the default grid the effective K of multi-Gaussian fields averages about
  4% below the geometric mean that theory gives for 2D. This is grid error from
  the harmonic mean between cells.
- The random walk leaves out the drift term that depends on the gradient of the
  dispersion coefficient. Keep local dispersivities small compared with a cell.
- Solute cannot leave through the upstream edge or the sides.
- The covariance is smooth. Rougher covariances (exponential) need much finer
  grids to give the right effective K.
- No sorption or decay yet.
- Particles released from a line or region are spread evenly in space, not in
  proportion to flow.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository layout

```
HydroHETERO/
  app.py                    Streamlit front end
  hydrohetero/
    model.py                fields, flow, particle tracking, breakthrough curves
    export.py               node list, grid and MODFLOW array files
    plots.py                figures
    workbook.py             xlsx writer
    run_counter.py          shared run total (Upstash Redis over REST)
  logo/                     header logo, browser-tab icon, figure badge and svg source
  tests/                    pytest suite (no network needed)
```

## Licence

Copyright (c) 2026 Charles Darwin University. All rights reserved. See
`LICENSE`.

## References

Dagan, G. (1984). Solute transport in heterogeneous porous formations. Journal
of Fluid Mechanics 145, 151-177.

Gelhar, L.W., and Axness, C.L. (1983). Three-dimensional stochastic analysis of
macrodispersion in aquifers. Water Resources Research 19(1), 161-180.

Pollock, D.W. (1988). Semianalytical computation of path lines for
finite-difference models. Ground Water 26(6), 743-750.

Zinn, B., and Harvey, C.F. (2003). When good statistical models of aquifer
heterogeneity go bad: A comparison of flow, dispersion, and mass transfer in
connected and multivariate Gaussian hydraulic conductivity fields. Water
Resources Research 39(3), 1051.
