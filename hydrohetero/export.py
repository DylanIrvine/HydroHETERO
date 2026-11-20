# K fields written out in the formats other software reads

import io
import zipfile

import numpy as np
import pandas as pd

NODE_ORDERS = ['Along x first', 'Along y first']


def node_table(fields, d, order):
    # one row per cell: node number, cell-centre x and y, and K in each field.
    # Node 1 is the cell at the lowest x and y. 'Along x first' numbers across a row of cells
    # (x increasing) before moving up to the next y; 'Along y first' numbers up a column first
    ny, nx = fields[0].shape
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * d, (np.arange(ny) + 0.5) * d)
    flat = (lambda a: a.ravel(order='C')) if order == NODE_ORDERS[0] else (lambda a: a.ravel(order='F'))
    table = pd.DataFrame({'Node': np.arange(1, nx * ny + 1), 'x (m)': flat(X), 'y (m)': flat(Y)})
    values = pd.DataFrame({f'K field {i + 1} (m/d)': flat(K) for i, K in enumerate(fields)})
    return pd.concat([table, values], axis=1)


def grid_text(K, delimiter=','):
    # the field as rows of text, laid out as on a map: the first line is the row with the
    # highest y, and x increases along each line. This is also the row order MODFLOW uses
    buffer = io.StringIO()
    np.savetxt(buffer, K[::-1], fmt='%.6e', delimiter=delimiter)
    return buffer.getvalue()


MODFLOW_NOTE = '''K fields from HydroHETERO, as MODFLOW array files
=================================================

Each file k_NNN.dat holds one field: {ny} rows of {nx} values, separated by spaces.
The first line is row 1 (the row with the highest y) and values run along x, which is
the order MODFLOW reads a layer. Values are hydraulic conductivity in m/d. Cells are
{d:g} m square. Use the same length and time units in the rest of the model, or scale
the values.

A grid that matches: 1 layer, {ny} rows, {nx} columns, DELR = DELC = {d:g}.

MODFLOW 6, in the NPF package:

    BEGIN griddata
      icelltype
        CONSTANT 0
      k
        OPEN/CLOSE k_001.dat
    END griddata

MODFLOW-2005, as the HK array of the LPF package (or HY in BCF):

    OPEN/CLOSE k_001.dat 1.0 (FREE) -1

These files have not been run through MODFLOW by the app. Check the first model you
build against the map of the field.
'''


def archive(fields, d, formats, order, settings_text):
    # a zip of the fields in each chosen format, with a settings file.
    # formats is a set drawn from 'nodes', 'grid' and 'modflow'
    ny, nx = fields[0].shape
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('settings.txt', settings_text)
        if 'nodes' in formats:
            z.writestr('K_by_node.csv', node_table(fields, d, order).to_csv(index=False, float_format='%.6e'))
        for i, K in enumerate(fields):
            if 'grid' in formats:
                z.writestr(f'grid/K_{i + 1:03d}.csv', grid_text(K, ','))
            if 'modflow' in formats:
                z.writestr(f'modflow/k_{i + 1:03d}.dat', grid_text(K, ' '))
        if 'modflow' in formats:
            z.writestr('modflow/README.txt', MODFLOW_NOTE.format(nx=nx, ny=ny, d=d))
    return buffer.getvalue()
