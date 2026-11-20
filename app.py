# HydroHETERO
# solute transport through random hydraulic conductivity fields, by particle tracking.
# Run with: streamlit run app.py

import datetime as dt
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from hydrohetero import VERSION as APP_VERSION
from hydrohetero import export, model, plots, run_counter, workbook

# %% page setup

APP_NAME = 'HydroHETERO'
# app address, used in the citation
APP_URL = 'https://hydrohetero.streamlit.app/'
LOGO = Path('logo/HydroHETERO_logo.png')
FAVICON = Path('logo/HydroHETERO_icon.png')
MAX_CELLS = 80000
MAX_VALUES = 8_000_000      # cells times fields held in memory
N_TIMES = 300

st.set_page_config(page_title=APP_NAME, page_icon=str(FAVICON) if FAVICON.exists() else None)

# %% header

head_text, head_logo = st.columns([3, 1], vertical_alignment='center')
with head_text:
    st.title(APP_NAME)
    st.subheader('Random hydraulic conductivity fields, and transport through them')
    st.write('Dylan Irvine, Charles Darwin University.')
    st.caption(f'Version {APP_VERSION}')
with head_logo:
    if LOGO.exists():
        st.image(str(LOGO), width='stretch')

st.markdown(
    'Hydraulic conductivity varies from place to place in every aquifer, and how the high and low '
    'values connect can matter as much as how much they vary. HydroHETERO does three things:\n'
    '1. **Makes random 2D conductivity fields** with the statistics you choose, to download as a '
    'node list, a grid, or MODFLOW array files.\n'
    '2. **Runs transport through every field** (optional), to show the range of breakthrough '
    'curves that the same statistics can produce.\n'
    '3. **Tracks particles through one field** (optional), from a point, a line or a region.\n\n'
    'This is a numerical model for teaching. It is 2D, the fields are idealised, and the flow '
    'and transport have not been compared with an established code.')

if 'run_count' not in st.session_state:
    st.session_state['run_count'] = run_counter.get_runs()
if st.session_state['run_count'] is not None:
    st.caption(f"This tool has generated fields {st.session_state['run_count']:,} times.")

with st.expander('**Privacy and data handling**'):
    st.markdown(
        '- Everything you enter is held in memory on the server for your session and discarded '
        'when it ends. Nothing is saved to a database or to permanent storage.\n'
        '- No login, and no record kept by this app of who you are or where you are.\n'
        '- The only thing recorded is a single running total: the number of times fields were generated.\n'
        '- Processing happens on the server that hosts the app. The hosting provider and the '
        'Streamlit framework may use their own cookies and keep their own server logs, under their '
        'own privacy policies.')

with st.expander('**Background: how it works**'):
    st.markdown(
        '**Fields.** The natural log of hydraulic conductivity, $\\ln K$, is normally distributed '
        'with a chosen geometric mean and standard deviation $\\sigma$. Its correlation falls off '
        'with distance $r$ as $\\exp[-(r/\\ell)^2]$, where $\\ell$ is the correlation length. Three '
        'structures are offered, all with the same distribution of $K$:\n'
        '- *Multi-Gaussian*: the standard model. The middle values form the connected network, '
        'and the extremes sit in isolated patches.\n'
        '- *Connected high K*: the highest values form connected channels.\n'
        '- *Connected low K*: the lowest values form connected barriers.\n\n'
        'The connected fields follow Zinn and Harvey (2003). For a standard normal field $Y$:')
    st.latex(r"Y' = \sqrt{2}\,\mathrm{erf}^{-1}\left[2\,\mathrm{erf}\left(\frac{|Y|}{\sqrt{2}}\right) - 1\right]")
    st.markdown(
        "$Y'$ is still standard normal, but its lowest values lie along the connected network. "
        "$\\ln K = \\mu + \\sigma Y'$ gives connected low K and $\\ln K = \\mu - \\sigma Y'$ gives "
        'connected high K. The transform shortens the correlation length, so the starting field is '
        'made with a longer one to compensate.\n\n'
        '**Flow.** Heads are fixed on the left and right edges to give the chosen mean gradient, '
        'and the other two edges are closed. Steady flow is solved by finite differences, one head '
        'per cell, with the harmonic mean of $K$ between neighbouring cells.\n\n'
        '**Transport.** Particles are released on the source line in proportion to the flow '
        'through it and moved along the flow, with a random step for local dispersion. The time '
        'each particle first crosses a control plane is recorded. The fraction that has arrived by '
        'time $t$ is the flux-averaged concentration at the plane for a constant source, and a '
        'source that changes in steps is the sum of shifted copies of that curve.\n\n'
        '**Uniform aquifer (theory).** The dashed line is the Ogata and Banks (1961) solution for a '
        'uniform aquifer with $K$ equal to the geometric mean and a dispersivity of '
        r'$\alpha_L + \sigma^2 \ell \sqrt{\pi}/2$. The second term is the macrodispersivity from '
        'first-order theory (Gelhar and Axness 1983; Dagan 1984), which holds far from the source '
        'in a multi-Gaussian field with $\\sigma^2$ below about 1.\n\n'
        '**Limits**\n'
        '- 2D only. Connected paths form more easily in 3D, so the results do not carry over '
        'directly.\n'
        '- The field needs at least four cells per correlation length, and the domain should be '
        'many correlation lengths wide.\n'
        '- With the default grid the effective K of multi-Gaussian fields comes out a few percent '
        'below the geometric mean, which theory gives for 2D. This is grid error.\n'
        '- The random walk leaves out a small drift term, so local dispersivities should be kept '
        'small compared with the cell size.\n'
        '- Solute cannot leave through the upstream edge or the sides.')

with st.expander('**How to cite and references**'):
    st.write(
        'This app:\n\n'
        f'Irvine, D.J. ({dt.date.today().year}). {APP_NAME} (version {APP_VERSION}) '
        f'[Web application]. Charles Darwin University. {APP_URL}\n\n'
        'References:\n\n'
        'Dagan, G. (1984). Solute transport in heterogeneous porous formations. Journal of Fluid '
        'Mechanics 145, 151-177.\n\n'
        'Gelhar, L.W., and Axness, C.L. (1983). Three-dimensional stochastic analysis of '
        'macrodispersion in aquifers. Water Resources Research 19(1), 161-180.\n\n'
        'Ogata, A., and Banks, R.B. (1961). A solution of the differential equation of longitudinal '
        'dispersion in porous media. U.S. Geological Survey Professional Paper 411-A.\n\n'
        'Pollock, D.W. (1988). Semianalytical computation of path lines for finite-difference '
        'models. Ground Water 26(6), 743-750.\n\n'
        'Zinn, B., and Harvey, C.F. (2003). When good statistical models of aquifer heterogeneity '
        'go bad: A comparison of flow, dispersion, and mass transfer in connected and multivariate '
        'Gaussian hydraulic conductivity fields. Water Resources Research 39(3), 1051.')

with st.expander('Plot colours'):
    st.caption('Change these to match your organisation. They apply to every plot.')
    colours = {}
    picker_columns = st.columns(3)
    for i, (role, default) in enumerate(plots.DEFAULT_COLOURS.items()):
        colours[role] = picker_columns[i % 3].color_picker(role, default)

problems = []
entered = {}


def number(label, value, column=st, **kwargs):
    # a number box whose value is also kept for the settings sheet
    result = column.number_input(label, value=value, format='%g', **kwargs)
    entered[label] = f'{result:g}'
    return result


# %% 1. conductivity fields

st.header('1. Conductivity fields')
structure = st.radio('Structure', model.STRUCTURES, horizontal=True)
entered['Structure'] = structure
col1, col2, col3 = st.columns(3)
k_geometric = number('Geometric mean K (m/d)', 10.0, col1, min_value=0.0)
sigma = number('Standard deviation of ln K', 1.5, col2, min_value=0.0, max_value=4.0)
length = number('Correlation length (m)', 6.0, col3, min_value=0.0)
col1, col2, col3 = st.columns(3)
domain_x = number('Domain length, x (m)', 200.0, col1, min_value=0.0)
domain_y = number('Domain width, y (m)', 100.0, col2, min_value=0.0)
d = number('Cell size (m)', 1.0, col3, min_value=0.0)
col1, col2 = st.columns(2)
n_fields = col1.slider('Number of K fields', 1, 200, 30)
seed = int(col2.number_input('Random seed', value=1, min_value=0, step=1,
                             help='The same seed and settings give the same fields.'))
nx = ny = 0
if min(k_geometric, length, domain_x, domain_y, d) <= 0:
    problems.append('The mean K, correlation length, domain size and cell size must be greater than zero.')
else:
    nx, ny = int(round(domain_x / d)), int(round(domain_y / d))
    st.caption(f'{nx} by {ny} cells. K spans roughly {k_geometric * np.exp(-2 * sigma):.3g} to '
               f'{k_geometric * np.exp(2 * sigma):.3g} m/d (95% of values).')
    if nx * ny > MAX_CELLS:
        problems.append(f'The grid has {nx * ny:,} cells. The limit is {MAX_CELLS:,}, so use a larger cell or a smaller domain.')
    elif nx * ny * n_fields > MAX_VALUES:
        problems.append(f'{n_fields} fields of {nx * ny:,} cells is too much to hold in memory. Use fewer fields or a coarser grid.')
    if nx < 20 or ny < 10:
        problems.append('The grid needs at least 20 cells along x and 10 along y.')
    if length < 4 * d:
        problems.append('Use a cell size of at most a quarter of the correlation length.')
    if length > domain_y / 6:
        st.caption('The domain is less than six correlation lengths wide, so single channels or '
                   'barriers can dominate and results will vary a lot between fields.')
for problem in problems:
    st.info(problem)
field_inputs = (structure, k_geometric, sigma, length, domain_x, domain_y, d, n_fields, seed)

if st.button('Generate fields', type='primary', disabled=bool(problems)):
    rng = np.random.default_rng(seed)
    bar = st.progress(0.0, text='Starting...')
    fields = []
    for i in range(n_fields):
        fields.append(model.k_field(nx, ny, d, k_geometric, sigma, length, structure, rng))
        bar.progress((i + 1) / n_fields, text=f'Field {i + 1} of {n_fields}')
    bar.empty()
    table = [('App', f'{APP_NAME} version {APP_VERSION}'), ('Run date', dt.date.today().isoformat())]
    table += list(entered.items()) + [('Grid', f'{nx} by {ny} cells'), ('Number of K fields', n_fields),
                                      ('Random seed', seed)]
    st.session_state['fields'] = dict(K=fields, d=d, nx=nx, ny=ny, k_geometric=k_geometric, sigma=sigma,
                                      length=length, structure=structure, seed=seed, settings=table,
                                      inputs=field_inputs)
    for stale in ('transport', 'tracking', 'field_zip'):
        st.session_state.pop(stale, None)
    new_total = run_counter.increment_runs()
    if new_total is not None:
        st.session_state['run_count'] = new_total

FIGURES = {}
FIGURE_DPI = 300


def show(fig, name):
    # render once, show it, keep the png for the zip
    plots.add_watermark(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    FIGURES[name] = buffer.getvalue()
    st.image(FIGURES[name], width='stretch')
    st.download_button(f'Download PNG ({FIGURE_DPI} dpi)', FIGURES[name], file_name=f'{name}.png',
                       mime='image/png', key=f'download_{name}', on_click='ignore')


def settings_text(rows):
    return '\n'.join(f'{name}: {value}' for name, value in rows) + '\n'


if 'fields' not in st.session_state:
    st.caption('Press Generate fields to make the fields. The download, transport and particle tracking steps appear below once they exist.')
    st.stop()

F = st.session_state['fields']
fields, fd = F['K'], F['d']
prefix = APP_NAME
size_x, size_y = F['nx'] * fd, F['ny'] * fd
FIGURE_DPI = st.radio('Figure download resolution (dpi)', [300, 600], horizontal=True)

st.subheader('The fields')
if F['inputs'] != field_inputs:
    st.warning('The inputs above have changed since these fields were made. Press Generate fields to apply '
               'the changes. Until then, everything below uses the earlier fields.')
st.caption(f"{len(fields)} fields, {F['structure']}, {F['nx']} by {F['ny']} cells. Change any input above "
           'and press Generate fields again to replace them.')
shown = st.number_input('Field to show', value=1, min_value=1, max_value=len(fields), step=1) - 1
show(plots.field(fields[shown], fd, c=colours, marks=False), f'{prefix}_field_{shown + 1}')

st.subheader('Download the fields')
st.caption('If the fields are all you need, this is the place to stop. Choose the formats and download '
           'one zip that holds every field.')
col1, col2, col3 = st.columns(3)
formats = set()
if col1.checkbox('Node list (CSV)', value=True, help='One row per cell: node number, x, y, and K in each field.'):
    formats.add('nodes')
if col2.checkbox('Grid (CSV)', value=True, help='One file per field, laid out as on a map.'):
    formats.add('grid')
if col3.checkbox('MODFLOW array files', value=True, help='One file per field, read with OPEN/CLOSE. A note in the zip gives the lines to use.'):
    formats.add('modflow')
order = st.radio('Node numbering', export.NODE_ORDERS, horizontal=True, disabled='nodes' not in formats,
                 help='Node 1 is the cell at the lowest x and y. Along x first: numbers run across a row of '
                      'cells, then move up to the next y. Along y first: numbers run up a column, then '
                      'move to the next x.')
if not formats:
    st.info('Choose at least one format.')
else:
    wanted = (tuple(sorted(formats)), order)
    if st.button('Prepare the download'):
        with st.spinner('Writing the files...'):
            st.session_state['field_zip'] = (wanted, export.archive(fields, fd, formats, order, settings_text(F['settings'])))
    if st.session_state.get('field_zip', (None,))[0] == wanted:
        st.download_button('Download the fields (.zip)', st.session_state['field_zip'][1],
                           file_name=f'{prefix}_fields.zip', mime='application/zip', type='primary', on_click='ignore')

# %% 2. transport through every field

st.header('2. Transport through every field (optional)')
st.caption('Flow runs from left to right. Solute enters on the left edge and is counted as it crosses '
           'each control plane.')
problems = []
entered = {}
col1, col2 = st.columns(2)
gradient = number('Mean hydraulic gradient', 0.01, col1, min_value=0.0)
porosity = number('Effective porosity', 0.3, col2, min_value=0.01, max_value=1.0)
col1, col2, col3 = st.columns(3)
alpha_l = number('Local longitudinal dispersivity (m)', 0.1, col1, min_value=0.0)
alpha_t = number('Local transverse dispersivity (m)', 0.01, col2, min_value=0.0)
diffusion = number('Effective diffusion coefficient (m²/d)', 0.0, col3, min_value=0.0)
if gradient <= 0:
    problems.append('The hydraulic gradient must be greater than zero.')
velocity = F['k_geometric'] * max(gradient, 1e-12) / porosity
st.caption(f'Velocity in a uniform aquifer with the geometric mean K: {velocity:.3g} m/d.')
if alpha_l > fd:
    st.caption('The local dispersivity is larger than a cell. The particle method is less accurate there.')

source_kind = st.radio('Source on the left edge', ['Full width', 'Part of the width'], horizontal=True)
source = None
if source_kind == 'Part of the width':
    col1, col2 = st.columns(2)
    y_low = number('Source from y (m)', 0.4 * size_y, col1, min_value=0.0)
    y_high = number('Source to y (m)', 0.6 * size_y, col2, min_value=0.0)
    source = (y_low, y_high)
    if not 0 <= y_low < y_high <= size_y:
        problems.append('The source must lie within the domain width, with the second value larger.')
    st.caption('The concentration reported at a control plane is then for the water that passed '
               'through the source. It is not diluted by the clean water that flows past the source.')
st.caption('Each row sets the source concentration from its start time until the next row. End with '
           'a row of zero to switch the source off.')
default_source = pd.DataFrame({'Start (days)': [0.0, 300.0], 'Concentration (mg/L)': [100.0, 0.0]})
source_table = st.data_editor(
    default_source, num_rows='dynamic', width='stretch', hide_index=True,
    column_config={'Start (days)': st.column_config.NumberColumn(min_value=0.0, required=True),
                   'Concentration (mg/L)': st.column_config.NumberColumn(min_value=0.0, required=True)})
rows = source_table.dropna()
order_t = np.argsort(rows.iloc[:, 0].to_numpy(float), kind='stable')
ts = rows.iloc[:, 0].to_numpy(float)[order_t]
cs = rows.iloc[:, 1].to_numpy(float)[order_t]
if ts.size == 0 or not (cs > 0).any():
    problems.append('Give the source history at least one row with a concentration above zero.')
if np.unique(ts).size < ts.size:
    problems.append('Give each row of the source history a different start time.')


def plane_list(text, label):
    # distances typed as a comma-separated list, checked against the domain
    try:
        values = sorted({float(part) for part in text.replace(';', ',').split(',') if part.strip()})
    except ValueError:
        problems.append(f'Enter the {label} as numbers separated by commas.')
        return []
    if len(values) > 4 or any(not 0 < x < size_x for x in values):
        problems.append(f'Enter up to four {label}, each inside the domain.')
        return []
    return values


planes = plane_list(st.text_input('Control planes: x (m)', value=f'{0.25 * size_x:g}, {0.75 * size_x:g}',
                                  help='Up to four distances, separated by commas.'), 'control planes')
if not planes and not problems:
    problems.append('Enter at least one control plane.')
col1, col2 = st.columns(2)
t_end = number('Length of the calculation (days)', float(f'{2.5 * size_x / velocity:.2g}'), col1, min_value=0.0)
n_particles = col2.select_slider('Particles per field', [500, 1000, 2000, 5000, 10000], value=2000)
if t_end <= 0:
    problems.append('The length of the calculation must be greater than zero.')
if not problems:
    seconds = len(fields) * (F['nx'] * F['ny'] / 20000 * 0.08 + n_particles * F['nx'] / 200 * 1e-4)
    st.caption(f'Rough run time: {max(1, round(seconds))} seconds on a typical server.')
for problem in problems:
    st.info(problem)
transport_inputs = (gradient, porosity, alpha_l, alpha_t, diffusion, source, tuple(ts), tuple(cs), tuple(planes),
                    t_end, n_particles)

if st.button('Run transport', type='primary', disabled=bool(problems)):
    try:
        times = np.linspace(0, t_end, N_TIMES)
        snapshots = [fraction * planes[-1] / velocity for fraction in (0.25, 0.6, 1.0)]
        setup = dict(d=fd, gradient=gradient, porosity=porosity, alpha_l=alpha_l, alpha_t=alpha_t,
                     diffusion=diffusion, planes=planes, n_particles=n_particles, source=source,
                     snapshots=snapshots)
        rng = np.random.default_rng(F['seed'] + 1)
        bar = st.progress(0.0, text='Starting...')
        curves = [[] for _ in planes]
        k_ratio, medians, first = [], [], None
        for i, K in enumerate(fields):
            one = model.transport(K, setup, rng, detail=i == 0)
            if i == 0:
                first = one
            for k in range(len(planes)):
                curves[k].append(model.breakthrough(one['arrival'][:, k], times, ts, cs))
            k_ratio.append(one['k_effective'] / F['k_geometric'])
            medians.append([np.nanmedian(one['arrival'][:, k]) if np.isfinite(one['arrival'][:, k]).any()
                            else np.nan for k in range(len(planes))])
            bar.progress((i + 1) / len(fields), text=f'Field {i + 1} of {len(fields)}')
        bar.empty()
        dispersivity = alpha_l + model.macrodispersivity(F['sigma'], F['length'])
        theory = [model.uniform_breakthrough(x, times, ts, cs, velocity, dispersivity * velocity + diffusion)
                  for x in planes]
        table = F['settings'] + list(entered.items())
        table += [(f'Source from {start:g} days', f'{value:g} mg/L') for start, value in zip(ts, cs)]
        table += [('Source', source_kind), ('Control planes (m)', ', '.join(f'{x:g}' for x in planes)),
                  ('Particles per field', n_particles),
                  ('Dispersivity used for the uniform aquifer (m)', f'{dispersivity:.4g}')]
        st.session_state['transport'] = dict(
            times=times, curves=[np.array(c) for c in curves], theory=theory, planes=planes, source=source,
            first=first, snapshots=snapshots, ts=ts, cs=cs, k_ratio=np.array(k_ratio), medians=np.array(medians),
            inputs=transport_inputs,
            settings=pd.DataFrame(table, columns=['Setting', 'Value']).astype(str))
    except Exception as error:
        st.error(f'Run failed: {error}')

if 'transport' in st.session_state:
    r = st.session_state['transport']
    if r['inputs'] != transport_inputs:
        st.warning('The transport inputs have changed since these results were made. Press Run transport to update them.')
    times, rplanes, first = r['times'], r['planes'], r['first']
    col1, col2, col3 = st.columns(3)
    col1.metric('Effective K / geometric mean', f"{np.mean(r['k_ratio']):.2f}")
    col2.metric('Median arrival at the last plane', f"{np.nanmedian(r['medians'][:, -1]):,.0f} days")
    col3.metric('Range between fields (5th to 95th)',
                f"{np.nanpercentile(r['medians'][:, -1], 5):,.0f} to {np.nanpercentile(r['medians'][:, -1], 95):,.0f} days")
    workbook_slot = st.container()

    st.subheader('Breakthrough at the control planes')
    source_curve = np.zeros(len(times))
    for start, value in zip(r['ts'], r['cs']):
        source_curve[times >= start] = value
    show(plots.breakthrough(times, r['curves'], r['theory'], rplanes, source_curve, colours), f'{prefix}_breakthrough')
    st.caption('The concentration is the flow-weighted average at each control plane, in the water '
               'that passed through the source. Thin grey lines are single fields (up to 60 shown). The '
               'dashed line is a uniform aquifer with the macrodispersivity from first-order theory.')

    st.subheader('Field 1')
    show(plots.field(fields[0], fd, first['paths'], rplanes, r['source'], colours), f'{prefix}_paths')
    st.caption('Hydraulic conductivity, with the paths of 60 particles in white. The source is the bar on '
               'the left edge and the dashed lines are the control planes.')
    show(plots.clouds(fields[0], fd, first['snapshots'], r['snapshots'], rplanes, r['source'], colours),
         f'{prefix}_plume')
    st.caption('Where solute released at time zero has reached at three times.')

    st.subheader('Variation between fields')
    show(plots.spread(r['k_ratio'], r['medians'], rplanes, colours), f'{prefix}_variation')
    st.caption('Left: the K of a uniform aquifer that would carry the same flow, relative to the '
               'geometric mean. Right: the time at which half the solute has passed each plane.')

    # workbook: the time series first, the input settings second, then the other tables
    series = pd.DataFrame({'Time (days)': times, 'Source (mg/L)': source_curve})
    sheets = {}
    for k, x in enumerate(rplanes):
        low, mid, high = np.percentile(r['curves'][k], [5, 50, 95], axis=0)
        series[f'x = {x:g} m, median (mg/L)'] = mid
        series[f'x = {x:g} m, 5th percentile (mg/L)'] = low
        series[f'x = {x:g} m, 95th percentile (mg/L)'] = high
        series[f'x = {x:g} m, uniform aquifer (mg/L)'] = r['theory'][k]
        every = pd.DataFrame(r['curves'][k].T, columns=[f'Field {i + 1}' for i in range(len(fields))])
        every.insert(0, 'Time (days)', times)
        sheets[f'All fields x = {x:g} m'] = every
    per_field = pd.DataFrame({'Field': np.arange(1, len(fields) + 1), 'Effective K / geometric mean': r['k_ratio']})
    for k, x in enumerate(rplanes):
        per_field[f'Median arrival at x = {x:g} m (days)'] = r['medians'][:, k]
    xlsx = workbook.build({'Time series': series, 'Settings': r['settings'], 'Fields': per_field, **sheets})
    with workbook_slot:
        st.download_button('Download the transport workbook (.xlsx)', xlsx, file_name=f'{prefix}_transport.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           type='primary', on_click='ignore')

# %% 3. particle tracking in one field

st.header('3. Particle tracking in one field (optional)')
st.caption('Release particles anywhere in one field and follow them to the right-hand edge. This uses '
           'the gradient, porosity and dispersion values from step 2.')
problems = []
col1, col2, col3 = st.columns(3)
chosen = int(col1.number_input('Field', value=1, min_value=1, max_value=len(fields), step=1)) - 1
release = col2.radio('Release from', ['Point', 'Line', 'Region'], horizontal=True)
n_track = int(col3.number_input('Number of particles', value=50, min_value=1, max_value=2000, step=10))
col1, col2, col3, col4 = st.columns(4)
x1 = col1.number_input('x (m)' if release == 'Point' else 'First x (m)', value=0.05 * size_x, format='%g')
y1 = col2.number_input('y (m)' if release == 'Point' else 'First y (m)', value=(0.5 if release == 'Point' else 0.2) * size_y, format='%g')
x2 = y2 = None
if release != 'Point':
    x2 = col3.number_input('Second x (m)', value=(0.05 if release == 'Line' else 0.15) * size_x, format='%g')
    y2 = col4.number_input('Second y (m)', value=0.8 * size_y, format='%g')
    st.caption('The two points are the ends of the line.' if release == 'Line' else
               'The two points are opposite corners of a rectangle. Particles start at random inside it.')
inside = [0 <= x1 < size_x, 0 <= y1 <= size_y] + ([0 <= x2 < size_x, 0 <= y2 <= size_y] if release != 'Point' else [])
if not all(inside):
    problems.append('The release points must lie inside the domain.')
with_dispersion = st.checkbox('Include local dispersion', value=False,
                              help='Off: particles follow the flow exactly, so particles from one point share one '
                                   'path. On: each step adds the random spreading set in step 2.')
track_planes = plane_list(st.text_input('Report the travel time to x (m)', value='',
                                        help='Optional. Up to four distances, separated by commas. The time to '
                                             'the right-hand edge is always reported.'), 'travel-time distances')
if gradient <= 0:
    problems.append('Set a hydraulic gradient above zero in step 2.')
for problem in problems:
    st.info(problem)

if st.button('Track particles', type='primary', disabled=bool(problems)):
    try:
        with st.spinner('Tracking...'):
            rng = np.random.default_rng(F['seed'] + 2)
            starts = model.release_points(release, n_track, rng, x1, y1, x2, y2)
            edge = size_x * (1 - 1e-9)
            h, qx, qy = model.flow(fields[chosen], fd, gradient)
            out = model.track(qx, qy, fd, porosity, track_planes + [edge], n_track, rng,
                              alpha_l if with_dispersion else 0.0, alpha_t if with_dispersion else 0.0,
                              diffusion if with_dispersion else 0.0, paths=n_track, start=starts)
            st.session_state['tracking'] = dict(field=chosen, starts=starts, planes=track_planes, paths=out['paths'],
                                                arrival=out['arrival'], dispersion=with_dispersion)
    except Exception as error:
        st.error(f'Tracking failed: {error}')

if 'tracking' in st.session_state:
    k = st.session_state['tracking']
    show(plots.field(fields[k['field']], fd, k['paths'], k['planes'], None, colours, marks=False, starts=k['starts']),
         f'{prefix}_tracking')
    labels = [f'x = {x:g} m' for x in k['planes']] + ['Right-hand edge']
    reached = np.isfinite(k['arrival'][:, -1]).mean() * 100
    st.caption(f"Field {k['field'] + 1}. Dots are the starting points and white lines are the paths"
               + (', with local dispersion.' if k['dispersion'] else ', following the flow only.')
               + f' {reached:.0f}% of particles reached the right-hand edge.')
    show(plots.travel_times(k['arrival'], labels, colours), f'{prefix}_travel_times')
    summary = pd.DataFrame({'Particle': np.arange(1, k['arrival'].shape[0] + 1), 'Start x (m)': k['starts'][0],
                            'Start y (m)': k['starts'][1]})
    for column, label in enumerate(labels):
        summary[f'Travel time to {label[0].lower() + label[1:]} (days)'] = k['arrival'][:, column]
    st.dataframe(summary.describe().loc[['min', '50%', 'max']].drop(columns='Particle').rename(
        index={'min': 'Shortest', '50%': 'Median', 'max': 'Longest'}), width='stretch')
    steps, count, _ = k['paths'].shape
    path_table = pd.DataFrame({'Particle': np.tile(np.arange(1, count + 1), steps),
                               'Step': np.repeat(np.arange(steps), count),
                               'x (m)': k['paths'][:, :, 0].ravel(), 'y (m)': k['paths'][:, :, 1].ravel(),
                               'Time (days)': k['paths'][:, :, 2].ravel()})
    path_table = path_table.sort_values(['Particle', 'Step']).drop_duplicates(['Particle', 'x (m)', 'y (m)', 'Time (days)'])
    col1, col2 = st.columns(2)
    col1.download_button('Download travel times (.csv)', summary.to_csv(index=False), on_click='ignore',
                         file_name=f'{prefix}_travel_times.csv', mime='text/csv')
    col2.download_button('Download paths (.csv)', path_table.to_csv(index=False, float_format='%.6g'), on_click='ignore',
                         file_name=f'{prefix}_paths.csv', mime='text/csv')

# %% everything

if FIGURES:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in FIGURES.items():
            archive.writestr(f'{name}.png', data)
    st.download_button('Download every figure on this page (.zip)', zip_buffer.getvalue(),
                       file_name=f'{prefix}_figures.zip', mime='application/zip', on_click='ignore')
