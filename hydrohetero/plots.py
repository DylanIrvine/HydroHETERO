# figures

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mimage
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
import numpy as np
from pathlib import Path

# default colours. The app lets the user change them
DEFAULT_COLOURS = {
    'Median and particles': '#CC004C',
    'Range': '#0089D0',
    'Uniform aquifer': '#F37021',
    'Source': '#6460AA',
    'Control planes': '#0DB14B',
    'Extra': '#FCB711',
}
# colour maps offered for the K fields. The first is the default
COLOUR_MAPS = ['cividis', 'viridis', 'cubehelix', 'gist_earth', 'Purples', 'Greys', 'YlOrBr', 'BuGn', 'BuPu', 'YlGnBu']
WATERMARK = 'Dylan Irvine, Charles Darwin University'
BADGE_ICON = Path(__file__).resolve().parents[1] / 'logo' / 'HydroHETERO_badge.png'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 9,
    'font.weight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

BADGE_PT = 14       # height of the app icon in the 'Made using' badge, in points


def add_badge(fig, x, y, transform):
    # 'Made using HydroHETERO' with the app icon, right-aligned at (x, y). Skipped if the icon is missing
    try:
        img = mimage.imread(BADGE_ICON)
    except (FileNotFoundError, OSError):
        return
    packed = HPacker(children=[TextArea('Made using HydroHETERO', textprops=dict(fontsize=6, color='0.5')),
                               OffsetImage(img, zoom=BADGE_PT / img.shape[0])],
                     align='center', pad=0, sep=3)
    fig.add_artist(AnnotationBbox(packed, (x, y), xycoords=transform, box_alignment=(1, 0.5),
                                  frameon=False, pad=0, annotation_clip=False))


def add_watermark(fig):
    # credit line bottom left and the 'Made using' badge bottom right, in a strip below the
    # plot so both can be cropped off
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    below = fig.transFigure + mtransforms.ScaledTranslation(0, -(BADGE_PT / 2 + 4) / 72, fig.dpi_scale_trans)
    fig.text(box.x0 / width, box.y0 / height, WATERMARK, transform=below,
             ha='left', va='center', fontsize=6, color='0.5')
    add_badge(fig, box.x1 / width, box.y0 / height, below)


def cm(width, height):
    return (width / 2.54, height / 2.54)


def legend_above(ax, ncols=3):
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=ncols, borderaxespad=0.2,
              handlelength=1.5, columnspacing=1.2)


def path_style(cmap):
    # line colour for particle paths over a colour map: white on a dark map, near black on a
    # light one, with a thin outline in the other tone so paths stay visible where the map
    # passes through the line colour
    r, g, b = matplotlib.colormaps[cmap](np.linspace(0, 1, 64))[:, :3].T
    light = (0.299 * r + 0.587 * g + 0.114 * b).mean() > 0.55
    line, outline = ('0.1', 'white') if light else ('white', '0.1')
    return dict(color=line, lw=0.6, alpha=0.9,
                path_effects=[patheffects.withStroke(linewidth=1.3, foreground=outline, alpha=0.35)])


def _k_map(ax, K, d, limits=None, alpha=1.0, cmap='cividis'):
    ny, nx = K.shape
    limits = limits or (K.min(), K.max())
    return ax.imshow(K, origin='lower', extent=(0, nx * d, 0, ny * d), cmap=cmap, alpha=alpha,
                     norm=LogNorm(*limits), aspect='equal', interpolation='nearest')


def _marks(ax, planes, source, width, c):
    low, high = (0, width) if source is None else source
    ax.plot([0, 0], [low, high], color=c['Source'], lw=4, solid_capstyle='butt', clip_on=False, zorder=5)
    for plane in planes:
        ax.axvline(plane, color=c['Control planes'], lw=1.2, ls='--', zorder=4)


def field(K, d, paths=None, planes=(), source=None, c=DEFAULT_COLOURS, marks=True, starts=None, cmap='cividis'):
    # the K field of one realisation, optionally with particle paths, the source line and
    # control planes (marks), or the starting points of tracked particles (starts)
    ny, nx = K.shape
    fig, ax = plt.subplots(figsize=cm(16, 3.5 + 12.5 * ny / nx), layout='constrained')
    image = _k_map(ax, K, d, cmap=cmap)
    if paths is not None:
        ax.plot(paths[:, :, 0], paths[:, :, 1], **path_style(cmap))
    if marks:
        _marks(ax, planes, source, ny * d, c)
    if starts is not None:
        ax.plot(starts[0], starts[1], 'o', color=c['Median and particles'], mec='white', mew=0.4, ms=3.5, zorder=6)
        for plane in planes:
            ax.axvline(plane, color=c['Control planes'], lw=1.2, ls='--', zorder=4)
    ax.set_xlabel('x, distance along the mean flow (m)')
    ax.set_ylabel('y (m)')
    fig.colorbar(image, ax=ax, pad=0.02, shrink=0.9).set_label('Hydraulic conductivity (m/d)')
    return fig


def clouds(K, d, snapshots, times, planes, source, c, cmap='cividis'):
    # where the particles of one realisation are at each chosen time, over a faint K field
    ny, nx = K.shape
    fig, axes = plt.subplots(len(times), 1, figsize=cm(16, 1.5 + len(times) * (1.2 + 12.5 * ny / nx)),
                             sharex=True, squeeze=False, layout='constrained')
    for ax, cloud, t in zip(axes[:, 0], snapshots, times):
        _k_map(ax, K, d, alpha=0.35, cmap=cmap)
        ax.plot(cloud[:, 0], cloud[:, 1], '.', color=c['Median and particles'], ms=1.5, alpha=0.6)
        _marks(ax, planes, source, ny * d, c)
        ax.set_ylabel('y (m)')
        ax.set_title(f'{t:g} days', loc='left', fontsize=9)
    axes[-1, 0].set_xlabel('x, distance along the mean flow (m)')
    return fig


def breakthrough(times, curves, theory, planes, source_curve, c, source_unit='mg/L'):
    # curves[k] is an array (realisations, times) for plane k. Thin lines are single
    # realisations, the band is their 5th to 95th percentile, the heavy line is their median
    n = len(planes)
    fig, axes = plt.subplots(n + 1, 1, figsize=cm(16, 4 + 5.5 * n), sharex=True,
                             height_ratios=[1] + [3] * n, layout='constrained')
    axes[0].step(times, source_curve, where='post', color=c['Source'], lw=1.6)
    axes[0].set_ylabel(f'Source\n({source_unit})')
    axes[0].set_ylim(0, None)
    for ax, plane, values, reference in zip(axes[1:], planes, curves, theory):
        for single in values[:60]:
            ax.plot(times, single, color='0.6', lw=0.4, alpha=0.6)
        low, mid, high = np.percentile(values, [5, 50, 95], axis=0)
        ax.fill_between(times, low, high, color=c['Range'], alpha=0.25, lw=0, label='5th to 95th percentile')
        ax.plot(times, mid, color=c['Median and particles'], lw=1.8, label='Median')
        ax.plot(times, reference, color=c['Uniform aquifer'], lw=1.4, ls='--', label='Uniform aquifer (theory)')
        ax.plot([], [], color='0.6', lw=0.6, label='Single fields')
        ax.set_ylabel('Concentration (mg/L)')
        ax.set_ylim(0, None)
        ax.set_title(f'Control plane at x = {plane:g} m', loc='left', fontsize=9)
    axes[1].legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Time (days)')
    axes[-1].set_xlim(times[0], times[-1])
    return fig


def spread(k_ratio, arrivals, planes, c):
    # how effective K and the median arrival time vary between fields
    fig, (left, right) = plt.subplots(1, 2, figsize=cm(16, 7), layout='constrained')
    left.hist(k_ratio, bins=min(20, max(5, len(k_ratio) // 3)), color=c['Range'], alpha=0.8)
    left.axvline(1, color='0.3', lw=0.8, ls=':')
    left.set_xlabel('Effective K / geometric mean K')
    left.set_ylabel('Number of fields')
    palette = [c['Median and particles'], c['Control planes'], c['Uniform aquifer'], c['Extra']]
    for k, plane in enumerate(planes):
        values = arrivals[:, k][np.isfinite(arrivals[:, k])]
        right.hist(values, bins=min(20, max(5, len(values) // 3)), color=palette[k % 4], alpha=0.6,
                   label=f'x = {plane:g} m')
    right.set_xlabel('Median arrival time (days)')
    legend_above(right, ncols=4)
    return fig


def travel_times(times, labels, c):
    # how long the tracked particles took to reach each plane. times is (particles, planes)
    fig, ax = plt.subplots(figsize=cm(16, 7), layout='constrained')
    palette = [c['Median and particles'], c['Control planes'], c['Uniform aquifer'], c['Extra'], c['Range']]
    for k, label in enumerate(labels):
        values = np.sort(times[:, k][np.isfinite(times[:, k])])
        if values.size:
            ax.step(values, np.arange(1, values.size + 1) / times.shape[0] * 100, where='post',
                    color=palette[k % 5], lw=1.6, label=label)
    ax.set_xlabel('Travel time (days)')
    ax.set_ylabel('Particles arrived (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, None)
    legend_above(ax, ncols=5)
    return fig
