#!/usr/bin/env python3
"""A simple gradio app for drawing a Bezier's curve."""
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt


def calc_bezier_pt(pts, t):
    """Calculate a Bezier's point at the current t (deCasteljau's algo).

    Parameters
    ----------
    pts : numpy.ndarray
        Control points of a bezier curve.
    t : float
        T-value of Bezier's curve [0; 1].

    Returns
    -------
    point
        A point of a Bezier's curve at t value.

    """
    for _ in range(len(pts) - 1):
        pts = (1 - t) * pts[:-1] + t * pts[1:]

    return pts[0]


def calc_bezier_curve(pts, n=100):
    """Calculate points n points of a Bezier's curve."""
    return [calc_bezier_pt(pts, t) for t in np.linspace(0, 1, n + 2)]


def calc_bezier_subdivision_pts(pts):
    """Calculate left/right points for a single subdivision iteration."""
    pts = np.array(pts)
    t = 0.5

    left_pts = [pts[0]] * (len(pts))
    right_pts = [pts[-1]] * (len(pts))
    for i in range(1, len(pts)):
        pts = (1 - t) * pts[:-1] + t * pts[1:]
        left_pts[i] = pts[0]
        right_pts[-i - 1] = pts[-1]

    return left_pts, right_pts


def calc_bezier_curve_subdivision(pts, n_iter=5):
    """Apply subdivision recursively and return resulted Bezier's curve."""
    if n_iter == 0:
        return pts

    left_pts, right_pts = calc_bezier_subdivision_pts(pts)
    return (calc_bezier_curve_subdivision(left_pts, n_iter - 1) +
            calc_bezier_curve_subdivision(right_pts, n_iter - 1))


def draw_bezier_curve(control_pts, n_steps=100, n_subdivisions=5):
    # prepare input data
    if n_steps < 0:
        raise gr.Error('n_steps should be >= 0.', duration=5)
    if n_subdivisions < 0:
        raise gr.Error('n_subdivisions should be >= 0.', duration=5)

    try:
        mask_x = pd.to_numeric(control_pts['x'], errors='coerce').notnull()
        mask_y = pd.to_numeric(control_pts['y'], errors='coerce').notnull()

        control_pts = control_pts[mask_x & mask_y]
        control_pts_np = control_pts.values.astype(np.float32)

        if len(control_pts) < 1:
            raise gr.Error('pass at least 1 control point.', duration=5)

    except:
        raise gr.Error('something wrong with input points.', duration=5)

    # prepare plt.figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_lim = control_pts_np.min() - 0.5
    max_lim = control_pts_np.max() + 0.5
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    ax.set_title('Bezier\'s curve (with control points)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_aspect('equal')

    # prepare bezier points
    bezier_pts = calc_bezier_curve(control_pts_np, n_steps)

    # draw curve
    ax.plot(*np.array(bezier_pts).T, color='b', label='naive')

    # draw control points
    for idx, (x, y) in enumerate(control_pts_np):
        ax.plot(x, y, marker='o', color='r', ls='')
        ax.annotate(f'P{idx + 1}', (x, y))

    # approximate curve using subdivision
    bezier_pts = calc_bezier_curve_subdivision(control_pts_np, n_subdivisions)

    # draw subdivided curve
    ax.plot(*np.array(bezier_pts).T, color='g', label='subdivision')
    ax.legend()

    return fig


# gradio app
demo = gr.Interface(
    fn=draw_bezier_curve,
    inputs=[gr.Dataframe(
        value=pd.DataFrame({'x': [0, 0, 1, 1], 'y': [0, 1, 1, 0]}),
        headers=['x', 'y'],
        datatype=['number', 'number'],
        row_count=1,
        col_count=(2, 'fixed')
    ), gr.Number(100), gr.Number(5)],
    outputs=[gr.Plot()],
    allow_flagging='never'
)
demo.launch()
