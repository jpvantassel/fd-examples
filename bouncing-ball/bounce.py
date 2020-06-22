"""Finite difference solution to bouncing ball."""

"""
Problem definition:

A ball falls in a vaccum from an initial height h(t=0). It impacts the
ground at h(t)=0 at which time its velocity is decreased by a factor
Cr (the material's coefficient of restitution) and bounces back in the
direction its started from.

Governing equation:
    h''(t) = -g, where g is acceleration due to gravity.

    or 

    v'(t) = -g
    h'(t) = v(t)

Boundary conditions:
    When the ball is close to zero two things may occur

    1) The ball bounces back with a reduced velocity.
    2) The ball comes to rest.

Discrete equation:
    h''(t) = [h'(t+dt) - h'(t)]/dt = -g -> Forward FD
    h''(t) = {[h(t+dt) - h(t+dt-dt)] - [h(t) - h(t-dt)]}/dt**2 = -g -> Backward FD
           = [h(t+dt) - 2*h(t) + h(t-dt)]/dt**2 = -g
    therefore,
    h(t+dt) = -g*dt**2 + 2*h(t) - h(t-dt)

Discrete initial conditions:
    h(t=0) = h0
    h(0) = h0

    h'(t=0) = v0

    h'(t=0) = [h(dt) - h(-dt)]/2*dt = v0
    h(-dt) = h(dt) - 2*dt*v0
    h(dt) = -g*dt**2 + 2*h(0) - h(-dt)

    2*h(-dt) = -g*dt**2 + 2*h(0) - 2*dt*v0
    h(-dt) = -0.5*g*dt**2 + h(0) - dt*v0

Discrete boundary condition:
    if h<EPS_H:
        v /= -cor
    
    v(t) = h'(t) = [h(t+dt) - h(t-dt)]/2*dt

    h(t+dt) = ?, so that v*(t) = v(t)/-cor

    v*(t) = [h(t+dt) - h(t-dt)]/-2*dt*cor

    [b - h(t-dt)] = [h(t+dt) - h(t-dt)]/-cor

    b = [h(t+dt) - h(t-dt)]/-cor + h(t-dt)

    if v<eps_v:
        v = 0
        break

"""

import numpy as np
import matplotlib.pyplot as plt


def bounce(h0, v0=0, cor=1, dt=0.01, nsteps=1000, g=9.81):
    h = np.empty(nsteps+1)

    a = -g*dt*dt
    h[0] = 0.5*a + h0 - dt*v0
    h[1] = h0

    EPS_H = 1E-3

    for n in range(2, nsteps+1):
        h[n] = a + 2*h[n-1] - h[n-2]

        if h[n] < EPS_H:
            h[n] = (h[n] - h[n-1])/-cor + h[n-1]

    return h[1:]


def draw_circle(ax, y, x=0, r=3):
    rads = np.linspace(0, 360, 100)*180/np.pi

    xs = x + np.cos(rads)*r
    ys = y + np.sin(rads)*r

    ax.plot(xs, ys, color="k")


if __name__ == "__main__":

    h = bounce(h0=100, v0=0, cor=1, dt=0.01, nsteps=1500)

    for n, h in enumerate(h[::20]):
        fig, ax = plt.subplots(figsize=(3,3), dpi=100)
        draw_circle(ax, h)
        ax.set_xlim(-55, 55)
        ax.set_xticks([])
        ax.set_ylim(0, 110)
        ax.set_yticks([])
        plt.savefig(str(n).zfill(3)+".png", dpi=100)
        plt.close()