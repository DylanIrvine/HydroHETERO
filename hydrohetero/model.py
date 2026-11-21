# random K fields, steady flow and particle tracking on a 2D grid. Metres and days.
#
# The grid has nx columns along the mean flow (x) and ny rows across it (y), with square
# cells of size d. Heads are fixed on the left and right edges and the other two edges are
# closed. Solute is released on a line near the left edge and counted as it first crosses
# each control plane, which gives the flux-averaged breakthrough curve there.

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import erf, erfc, erfcx, erfinv

STRUCTURES = ['Multi-Gaussian', 'Connected high K', 'Connected low K']
# the Zinn and Harvey (2003) transform shortens the correlation length. The Gaussian field
# is generated with a length this many times longer so that the final field has close to
# the length asked for (found numerically for this covariance)
CONNECTED_STRETCH = 1.7


def gaussian_field(nx, ny, d, length, rng, length_y=None):
    # a standard normal field with the covariance exp(-(r / length)^2), so length is the
    # distance at which the correlation falls to 1/e. This covariance gives smooth fields that
    # a grid of a few cells per length resolves. Made by filtering white noise in Fourier
    # space on a grid twice the size, then cropping, so that opposite edges are not correlated.
    # length is the correlation scale along x. length_y, if given, is the scale along y, and
    # the covariance becomes exp(-(rx / length)^2 - (ry / length_y)^2)
    length_y = length if length_y is None else length_y
    mx, my = 2 * nx, 2 * ny
    kx = 2 * np.pi * np.fft.fftfreq(mx, d)
    ky = 2 * np.pi * np.fft.fftfreq(my, d)
    k2 = (kx[None, :] * length) ** 2 + (ky[:, None] * length_y) ** 2
    noise = np.fft.fft2(rng.standard_normal((my, mx)))
    field = np.real(np.fft.ifft2(noise * np.exp(-k2 / 8)))[:ny, :nx]
    return (field - field.mean()) / field.std()


def macrodispersivity(sigma, length):
    # long-distance longitudinal macrodispersivity from first-order theory (Gelhar and Axness
    # 1983; Dagan 1984): variance of ln K times the integral scale along the flow, which is
    # length * sqrt(pi) / 2 for this covariance, with length the correlation scale along x.
    # Valid for a multi-Gaussian field of small variance
    return sigma ** 2 * length * np.sqrt(np.pi) / 2


def connect(field):
    # Zinn and Harvey (2003): values near the mean of a Gaussian field form a connected
    # network. Taking the absolute value and mapping back to a normal distribution puts the
    # lowest values on that network. The result is still standard normal
    u = np.clip(erf(np.abs(field) / np.sqrt(2)), 1e-12, 1 - 1e-12)
    return np.sqrt(2) * erfinv(2 * u - 1)


def k_field(nx, ny, d, k_geometric, sigma, length, structure, rng, length_y=None):
    # K (m/d) with ln K normal: geometric mean k_geometric, standard deviation of ln K sigma.
    # length is the correlation scale along x and length_y the scale along y (the same if not given)
    length_y = length if length_y is None else length_y
    if structure == 'Multi-Gaussian':
        y = gaussian_field(nx, ny, d, length, rng, length_y)
    else:
        y = connect(gaussian_field(nx, ny, d, length * CONNECTED_STRETCH, rng, length_y * CONNECTED_STRETCH))
        y = (y - y.mean()) / y.std()
        if structure == 'Connected high K':
            y = -y
    return k_geometric * np.exp(sigma * y)


def integral_scale(field, d):
    # correlation length along x from the lag at which the correlation falls to 1/e
    f = field - field.mean()
    spectrum = np.abs(np.fft.fft(f, axis=1)) ** 2
    corr = np.real(np.fft.ifft(spectrum, axis=1)).mean(axis=0)
    corr = corr / corr[0]
    below = np.argmax(corr < np.exp(-1))
    return d * (below - 1 + (corr[below - 1] - np.exp(-1)) / (corr[below - 1] - corr[below]))


def flow(K, d, gradient):
    # steady heads and Darcy fluxes. Returns head (ny, nx), qx on the vertical cell faces
    # (ny, nx + 1) and qy on the horizontal faces (ny + 1, nx), in m/d. Flow is in +x
    ny, nx = K.shape
    cx = 2 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:])      # between columns
    cy = 2 * K[:-1, :] * K[1:, :] / (K[:-1, :] + K[1:, :])      # between rows
    h_left = gradient * nx * d
    index = np.arange(nx * ny).reshape(ny, nx)
    rows = np.concatenate([index[:, :-1].ravel(), index[:, 1:].ravel(), index[:-1, :].ravel(), index[1:, :].ravel()])
    cols = np.concatenate([index[:, 1:].ravel(), index[:, :-1].ravel(), index[1:, :].ravel(), index[:-1, :].ravel()])
    vals = np.concatenate([cx.ravel(), cx.ravel(), cy.ravel(), cy.ravel()])
    A = sparse.coo_matrix((-vals, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()
    diagonal = -np.asarray(A.sum(axis=1)).ravel()
    b = np.zeros(nx * ny)
    # fixed heads on the outer faces of the first and last columns (half a cell away)
    diagonal[index[:, 0]] += 2 * K[:, 0]
    diagonal[index[:, -1]] += 2 * K[:, -1]
    b[index[:, 0]] = 2 * K[:, 0] * h_left
    h = spsolve((A + sparse.diags(diagonal)).tocsc(), b).reshape(ny, nx)
    qx = np.empty((ny, nx + 1))
    qx[:, 1:-1] = cx * (h[:, :-1] - h[:, 1:]) / d
    qx[:, 0] = 2 * K[:, 0] * (h_left - h[:, 0]) / d
    qx[:, -1] = 2 * K[:, -1] * h[:, -1] / d
    qy = np.zeros((ny + 1, nx))
    qy[1:-1, :] = cy * (h[:-1, :] - h[1:, :]) / d
    return h, qx, qy


def effective_k(qx, d, gradient):
    # the K of a uniform aquifer that would carry the same total flow
    return qx[:, 0].mean() / gradient


def _velocity(x, y, vx, vy, d, nx, ny):
    # velocity at each particle: each component varies linearly across its cell between the
    # two faces it is known on (Pollock 1988), which conserves water within the cell
    i = np.clip((x / d).astype(int), 0, nx - 1)
    j = np.clip((y / d).astype(int), 0, ny - 1)
    fx = x / d - i
    fy = y / d - j
    return (1 - fx) * vx[j, i] + fx * vx[j, i + 1], (1 - fy) * vy[j, i] + fy * vy[j + 1, i]


def track(qx, qy, d, porosity, planes, n_particles, rng, alpha_l=0.0, alpha_t=0.0, diffusion=0.0,
          source=None, step=0.4, snapshots=(), paths=0, max_steps=20000, start=None):
    # move particles from a line just inside the left edge until they pass the last plane.
    # Each step moves a particle a fixed fraction of a cell along the flow (so slow zones do
    # not need more steps), by a midpoint rule, then adds a random step for local dispersion.
    # Returns the time each particle first crossed each plane (n_particles, n_planes), and
    # optionally particle positions at the snapshot times and the full paths of a few particles.
    # source is (y_low, y_high) for the release line, or None for the full width.
    # start, if given, is (x, y) arrays of starting positions and replaces the release line.
    # paths holds x, y and time at every step for the first few particles.
    # The random walk leaves out the drift term that depends on the gradient of the
    # dispersion coefficient, which matters only where dispersion is large next to sharp K changes
    ny, nx = qy.shape[0] - 1, qx.shape[1] - 1
    vx, vy = qx / porosity, qy / porosity
    width = ny * d
    if start is None:
        y_low, y_high = (0.0, width) if source is None else source
        # release in proportion to flow, so each particle carries the same solute flux
        rows = np.arange(ny)
        inside = np.clip(np.minimum((rows + 1) * d, y_high) - np.maximum(rows * d, y_low), 0, None)
        weight = qx[:, 0] * inside
        row = rng.choice(ny, n_particles, p=weight / weight.sum())
        low = np.maximum(row * d, y_low)
        high = np.minimum((row + 1) * d, y_high)
        y = low + (high - low) * rng.random(n_particles)
        x = np.full(n_particles, 1e-6 * d)
    else:
        x = np.clip(np.asarray(start[0], float).copy(), 1e-6 * d, nx * d * (1 - 1e-9))
        y = np.clip(np.asarray(start[1], float).copy(), 0, width * (1 - 1e-12))
        n_particles = x.size
    y_start = y.copy()
    t = np.zeros(n_particles)
    planes = np.asarray(planes, float)
    arrival = np.full((n_particles, planes.size), np.nan)
    snapshots = np.asarray(snapshots, float)
    snap = np.full((snapshots.size, n_particles, 2), np.nan)
    trail = [np.column_stack([x[:paths], y[:paths], t[:paths]])] if paths else None
    last = planes.max()
    active = np.arange(n_particles)
    ds = step * d
    for _ in range(max_steps):
        if active.size == 0:
            break
        xa, ya = x[active], y[active]
        u, w = _velocity(xa, ya, vx, vy, d, nx, ny)
        speed = np.hypot(u, w) + 1e-300
        dt = ds / speed
        if diffusion > 0:
            dt = np.minimum(dt, (0.5 * d) ** 2 / (4 * diffusion))
        um, wm = _velocity(np.clip(xa + 0.5 * u * dt, 0, nx * d * (1 - 1e-12)),
                           np.clip(ya + 0.5 * w * dt, 0, width * (1 - 1e-12)), vx, vy, d, nx, ny)
        speed = np.hypot(um, wm) + 1e-300
        along = np.sqrt(2 * (alpha_l * speed + diffusion) * dt) * rng.standard_normal(active.size)
        across = np.sqrt(2 * (alpha_t * speed + diffusion) * dt) * rng.standard_normal(active.size)
        xn = xa + um * dt + (along * um - across * wm) / speed
        yn = ya + wm * dt + (along * wm + across * um) / speed
        xn = np.abs(xn)                                    # closed to solute on the upstream side
        yn = np.abs(yn)
        yn = np.where(yn > width, 2 * width - yn, yn)
        yn = np.clip(yn, 0, width * (1 - 1e-12))
        tn = t[active] + dt
        for k, plane in enumerate(planes):
            crossed = (xa < plane) & (xn >= plane) & np.isnan(arrival[active, k])
            if crossed.any():
                fraction = (plane - xa[crossed]) / (xn[crossed] - xa[crossed])
                arrival[active[crossed], k] = t[active][crossed] + fraction * dt[crossed]
        for k, when in enumerate(snapshots):
            passing = (t[active] < when) & (tn >= when)
            if passing.any():
                fraction = ((when - t[active][passing]) / dt[passing])[:, None]
                start = np.column_stack([xa[passing], ya[passing]])
                end = np.column_stack([xn[passing], yn[passing]])
                snap[k, active[passing]] = start + fraction * (end - start)
        x[active], y[active], t[active] = np.minimum(xn, nx * d * (1 - 1e-12)), yn, tn
        if paths:
            trail.append(np.column_stack([x[:paths], y[:paths], t[:paths]]))
        active = active[xn < last]
    return dict(arrival=arrival, snapshots=snap, paths=None if not paths else np.array(trail), y0=y_start)


def breakthrough(arrival, times, ts, cs):
    # flux-averaged concentration at a plane through time, for a source that changes in
    # steps: start times ts and values cs. arrival holds the first-crossing time of each
    # particle. Particles that never arrived count as not yet arrived
    arrived = np.sort(arrival[np.isfinite(arrival)])
    n = arrival.size
    out = np.zeros(len(times))
    previous = 0.0
    for start, value in zip(ts, cs):
        out += (value - previous) * np.searchsorted(arrived, np.asarray(times) - start, side='right') / n
        previous = value
    return out


def uniform_breakthrough(x, times, ts, cs, v, D):
    # the same curve for a uniform aquifer with velocity v and dispersion coefficient D
    # (Ogata and Banks 1961), written so that exp() cannot overflow
    out = np.zeros(len(times))
    previous = 0.0
    for start, value in zip(ts, cs):
        t = np.asarray(times, float) - start
        on = t > 0
        s = 2 * np.sqrt(D * t[on])
        b = (x + v * t[on]) / s
        out[on] += (value - previous) * 0.5 * (erfc((x - v * t[on]) / s)
                                               + np.exp(v * x / D - b ** 2) * erfcx(b))
        previous = value
    return out


def transport(K, settings, rng, detail=False):
    # flow and particle arrivals for one K field. settings is a dict with d, gradient, porosity,
    # alpha_l, alpha_t, diffusion, planes, n_particles, source and (for detail) snapshots
    g = settings
    h, qx, qy = flow(K, g['d'], g['gradient'])
    planes = list(g['planes'])
    if detail:
        # carry the particles on to the right-hand edge, so the plotted paths cross the whole field
        planes.append(K.shape[1] * g['d'] * (1 - 1e-9))
    tracked = track(qx, qy, g['d'], g['porosity'], planes, g['n_particles'], rng, g['alpha_l'], g['alpha_t'],
                    g['diffusion'], g['source'], snapshots=g['snapshots'] if detail else (), paths=60 if detail else 0)
    out = dict(arrival=tracked['arrival'][:, :len(g['planes'])], k_effective=effective_k(qx, g['d'], g['gradient']))
    if detail:
        out.update(head=h, snapshots=tracked['snapshots'], paths=tracked['paths'])
    return out


def release_points(kind, n, rng, x1, y1, x2=None, y2=None):
    # starting positions for particle tracking: 'Point' (all at x1, y1), 'Line' (evenly spaced
    # from x1, y1 to x2, y2) or 'Region' (random within the rectangle with those corners)
    if kind == 'Point':
        return np.full(n, float(x1)), np.full(n, float(y1))
    if kind == 'Line':
        f = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
        return x1 + f * (x2 - x1), y1 + f * (y2 - y1)
    return (min(x1, x2) + abs(x2 - x1) * rng.random(n), min(y1, y2) + abs(y2 - y1) * rng.random(n))
