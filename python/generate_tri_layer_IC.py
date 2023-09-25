import numpy as np
import matplotlib.pyplot as plt

Nx = int(input('Nx : '))
Ny = int(input('Ny : '))
Ar = float(input('Aspect ratio : '))
ymax = float(input('Total height of the box : '))
y1   = float(input('First transition height : '))
y2   = float(input('Second transition height : '))
S    = float(input('Stiffness'))
theta1 = float(input('Theta1 : '))
filename_out = input('Output filename : ')

m1     = 1.0
kappa1 = 0.07
gamma0 = 5.0/3.0

if y1 > ymax:
  print(f'Error y1 > ymax : {y1}, {ymax}')
  exit(0)
if y2 > ymax:
  print(f'Error y2 > ymax : {y2}, {ymax}')
  exit(0)

xmin = 0.0
xmax = Ar*ymax
ymin = 0.0

y = np.linspace(ymin, ymax, Ny)

mad = 1.0 / (gamma0-1.0)
m2  = S * (mad-m1) + mad
kappa2 = (m2 + 1.0) / (m1+1.0) * kappa1
theta2 = kappa1 / kappa2 * theta1

gval = (m1+1.0) * theta1

# Setting threshold values
T0   = 1.0
rho0 = 1.0
p0   = rho0*T0

T1   = T0 + theta2*y1
rho1 = rho0 * (T1/T0)**m2
p1   = p0   * (T1/T0)**(m2+1.0)

T2   = T1 + theta1*(y2-y1)
rho2 = rho1 * (T2/T1)**m1
p2   = p1   * (T2/T1)**(m1+1.0)

rho = np.empty_like(y)
T   = np.empty_like(y)
prs = np.empty_like(y)

zone1 = y<y1
zone3 = y>y2
zone2 = ~zone1 & ~zone3

T[zone1]   = T0 + theta2 * y[zone1]
rho[zone1] = rho0 * (T[zone1]/T0)**m2
prs[zone1] = p0   * (T[zone1]/T0)**(m2+1)

T[zone2]   = T1 + theta1 * (y[zone2]-y1)
rho[zone2] = rho1 * (T[zone2]/T1)**m1
prs[zone2] = p1   * (T[zone2]/T1)**(m1+1)

T[zone3]   = T2 + theta2 * (y[zone3]-y2)
rho[zone3] = rho2 * (T[zone3]/T2)**m2
prs[zone3] = p2   * (T[zone3]/T2)**(m2+1)

dy = y[1]-y[0]
hse = (prs[2:] - prs[:-2]) / (2.0*dy) - rho[1:-1]*20.0

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0,0].plot(y, T, '-k')
ax[0,1].plot(y, rho, '-k')
ax[1,0].plot(y, prs, '-k')
ax[1,1].plot(y[1:-1], hse, '-k')
ax[1,0].set_xlabel('y')
ax[1,1].set_xlabel('y')
ax[0,0].set_ylabel('Temperature')
ax[0,1].set_ylabel('Density')
ax[1,0].set_ylabel('Pressure')
ax[1,1].set_ylabel('HSE')

plt.tight_layout()
plt.show()


data_file = f'''[mesh]
Nx={Nx}
Ny={Ny}
xmin=0.0
xmax={xmax}
ymin=0.0
ymax={ymax}

[run]
tend=200.0
save_freq=1.0

boundaries_x=reflecting
boundaries_y=reflecting



[solvers]
riemann_solver=hllc
reconstruction=plm
CFL=0.1

[physics]
gamma0={gamma0}
problem=C91
gravity=true
g={gval}
well_balanced_flux_at_y_bc=true

[tri_layer]
perturbation=1.0e-3 
kappa1={kappa1/0.07}
kappa2={kappa2/0.07}
y1={y1}
y2={y2}

[polytrope]
theta1={theta1}
m1={m1}
theta2={theta2}
m2={m2}

[thermal_conduction]
active=true
kappa=0.07

conductivity_mode=tri_layer

bc_ymin=no_flux
bc_ymax=no_flux

[viscosity]
active=true
mu=0.028

[heating]
active=true
mode=C2020
log_total_heating=false


[misc]
log_frequency=10
log_energy_contributions=true
log_energy_frequency=10000'''

f_out = open(filename_out, 'w')
f_out.write(data_file)
f_out.close()