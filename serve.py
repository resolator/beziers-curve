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
    assert len(pts) > 0
    assert 0 <= t <= 1

    for _ in range(len(pts) - 1):
        pts = (1 - t) * pts[:-1] + t * pts[1:]

    return pts[0]


def calc_bezier_curve(pts, n=100):
    """Calculate points n points of a Bezier's curve."""
    return [calc_bezier_pt(pts, t) for t in np.linspace(0, 1, n + 2)]


def draw_bezier_curve(control_pts, n_steps=100):
    # prepare input data
    if n_steps < 0:
        raise gr.Error('n_steps should be >= 0.', duration=5)

    try:
        mask_x = pd.to_numeric(control_pts['x'], errors='coerce').notnull()
        mask_y = pd.to_numeric(control_pts['y'], errors='coerce').notnull()

        control_pts = control_pts[mask_x & mask_y]
        control_pts_np = control_pts.values.astype(np.float32)
    except:
        raise gr.Error('something wrong with input points.', duration=5)

    # prepare bezier points
    bezier_pts = calc_bezier_curve(control_pts_np, n_steps)
    bezier_x, bezier_y  = np.array(bezier_pts).T

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

    # draw curve
    ax.plot(bezier_x, bezier_y, color='b')

    # drav control points
    for idx, (x, y) in enumerate(control_pts_np):
        ax.plot(x, y, marker='o', color='r', ls='')
        ax.annotate(f'P{idx + 1}', (x, y))

    return fig


# gradio app
demo = gr.Interface(
    fn=draw_bezier_curve,
    inputs=[gr.Dataframe(
        value=pd.DataFrame({'x': [0, 0, 1], 'y': [0, 1, 1]}),
        headers=['x', 'y'],
        datatype=['number', 'number'],
        row_count=1,
        col_count=(2, 'fixed')
    ), gr.Number(5)],
    outputs=[gr.Plot()],
    allow_flagging='never'
)
demo.launch()
