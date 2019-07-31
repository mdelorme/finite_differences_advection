### Toro Section 5.2 -> Finite differences to a PDE
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys, os
import shutil

# Arguments
parser = argparse.ArgumentParser(description='Finite-differences using various ICs and schemes')
parser.add_argument('--xmin', default=-5.0,   type=np.float)
parser.add_argument('--xmax', default= 5.0,   type=np.float)
parser.add_argument('--Nx',   default= 1001,  type=np.int)
parser.add_argument('--tmax', default= 100.0, type=np.float)
parser.add_argument('--dt',   default=  0.1,  type=np.float)
parser.add_argument('--IC',   default=    0,  type=np.int, help=('0, 2, 4 : a=0.1; 1, 3, 5 : a=-0.1\n0, 1 : IC=Gaussian; 2, 3 : IC=Top hat; 4, 5 : IC=Triangle'))
parser.add_argument('--int' , default='CIR', type=str, help=('CIR, LF, LW, WB, FR'))

args = parser.parse_args()
x_min  = args.xmin
x_max  = args.xmax
t_max  = args.tmax
dt     = args.dt
Nx     = args.Nx     # Number of points
itc    = args.IC     # Test case ID
id_int = args.int    # Integrator ID
dx     = (x_max - x_min) / (Nx-1)

# Mesh creation
x = np.linspace(x_min, x_max, Nx)
u_old = np.zeros((Nx, ))
u_new = np.zeros((Nx, ))

# The three initial conditions
def ic_gaussian(u):
    sigma2 = 0.3
    u[:] = np.exp(-x**2.0/(2.0*sigma2)) / np.sqrt(2.0 * np.pi * sigma2)

def ic_tophat(u):
    mask = (x > -2.0) & (x < 2.0)
    u[mask] = 1.0

def ic_triangle(u):
    u[:] = 1.0 - 0.5 * np.abs(x)
    u[u < 0.0] = 0.0

# Test cases : Each IC, even itc -> advection to the right; odd itc -> advection to the left
test_cases = [{'a' :  0.1, 'ic' : ic_gaussian},
              {'a' : -0.1, 'ic' : ic_gaussian},
              {'a' :  0.1, 'ic' : ic_tophat},
              {'a' : -0.1, 'ic' : ic_tophat},
              {'a' :  0.1, 'ic' : ic_triangle},
              {'a' : -0.1, 'ic' : ic_triangle}]
tc = test_cases[itc]

# ICs
tc['ic'](u_old)
a = tc['a']

# Courang Number
c = a * dt / dx

# Building the list of integrators (requires c)
ap = max(a, 0.0)
am = min(a, 0.0)
cp = dt * ap / dx
cm = dt * am / dx

# Courant-Issacson-Rees
CIR = ((-1, cp),
       ( 0, 1.0-abs(c)),
       ( 1, -cm))

# Lax-Friedrichs
LF = ((-1, 0.5*(1.0+c)),
      ( 1, 0.5*(1.0-c)))

# Lax-Wendroff
LW = ((-1,  0.5*c*(1.0+c)),
      ( 0,  1.0-c*c),
      ( 1, -0.5*c*(1.0-c)))

# Warming and Beam
WB = ((-2, 0.5*c*(c-1.0)),
      (-1,     c*(2.0-c)),
      ( 0, 0.5*(c-1.0)*(c-2.0)))

# Fromm
FR = ((-2, -0.25*c*(1.0-c)),
      (-1,  0.25*c*(5.0-c)),
      ( 0,  0.25*(1.0-c)*(4.0+c)),
      ( 1, -0.25*c*(1.0-c)))

integrators = {'CIR': CIR,
               'LF' : LF,
               'LW' : LW,
               'WB' : WB,
               'FR' : FR}

# Current integrator
integrator = integrators[id_int]

# Building the results folder
if not os.path.exists('results'):
    os.mkdir('results')
    
prefix = 'results/{}_{}'.format(id_int, itc)
if os.path.exists(prefix):
    shutil.rmtree(prefix)
os.mkdir(prefix)

# Displaying the Courant number
# TODO : Check that c is in the stability range of the integrator
print('Courant number c={}'.format(c))

# Plotting the current state
def plot_state():
    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, u_old, '-+r')
    ax.set_ylim(0.0, 1.1)
    ax.set_title('TC = {}\nIntegrator = {}\nt = {:.3f}'.format(itc, id_int, t))
    plt.savefig(os.path.join(prefix, 'img_{:05}.png'.format(ite)))


# x-array id points for vectorisation of the stencil
ix = np.linspace(0, Nx-1, Nx, dtype=np.int)

t = 0.0
ite = 0
Nt = int(t_max / dt)

# Initial state
plot_state()

# Time loop
while t < t_max:
    sys.stderr.write('\rIteration {}/{}'.format(ite+1, Nt))
    ite += 1
    
    u_new[:] = 0.0

    for ui, b in integrator:
        u_new += u_old[(ix+ui)%Nx] * b

    t += dt
    u_old = u_new.copy()

    plot_state()




