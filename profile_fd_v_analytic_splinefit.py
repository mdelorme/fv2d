import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sys
import warnings
import math

def get_d_coef(N):
  #diag coefs
  dc = lambda n: np.diag(np.ones(N-abs(n)),  n)

  order = 2
  order_uncen = 5
  coef = lambda p, n: (-1)**(p+1) * math.factorial(n)**2/(p*math.factorial(n-p)*math.factorial(n+p))
  coef_uncen = lambda list_i: np.linalg.inv(np.matrix([[i**n/math.factorial(n) for n in range(len(list_i))] for i in list_i]))[1,:]
  if order_uncen <= 1:
    order_uncen = 2; coef_uncen = lambda dummy: [-1,1]
  d_coef = np.zeros((N,N))

  for i in range(1,order+1):
    c = coef(i, order)
    d_coef += c*dc(i) - c*dc(-i)

  d_coef[[0,-1],:] = 0
  d_coef[0,:order_uncen] = coef_uncen(range(order_uncen))
  d_coef[-1,:] = -d_coef[0,:][::-1]
  
  for i in range(1,order):
    d_coef[i,:] = 0
    d_coef[i,:order_uncen] = coef_uncen(range(-i, order_uncen-i))
  d_coef[-order-1:-1,:] = -d_coef[1:order+1,:][::-1, ::-1]
  return d_coef

def tr(f,g, rc=0.70, s=0.05):
  a = 0.5 * (1.0 + np.tanh((r-rc)/s))
  return (1-a)*f + a*g


N = 256
gamma = 5/3
Cp = gamma/(gamma-1) # R := Cp-Cv = 1

P0 = 1.0
rho0 = 1.0
g0 = 5
dSdr_LEFT   = -0.1
dSdr_RIGHT  = -0.1
# dSdr_RIGHT = -5.2e-2

dr = 1/(N-1)
r = np.linspace(0,1,N)
# r = np.linspace(0.5*dr,1-0.5*dr,N) # converge moins bien, à adapter
r_cut_dSdr = 0.2
r_cut_grav = 0.2

# dSdr = tr(dSdr_LEFT * np.ones(N), dSdr_RIGHT *np.ones(N), r_cut_dSdr, 0.025)
dSdr = np.ones(N) * dSdr_LEFT
S = sc.integrate.cumulative_simpson(dSdr, x=r, initial=0.)
g = r/r_cut_grav/(2+(r/r_cut_grav)**3) * 3*g0

d_coef = get_d_coef(N) / dr

Gam = 2-gamma
c1 = 1/Cp * dSdr
c2 = 1/gamma * g * np.exp(-gamma * S/Cp)

# todo: solution avec c1 une fonction de r
ksi_h = np.exp(-(1-Gam) * c1 * r) * rho0**(1-Gam)
ksi_p = -(1-Gam) * np.exp(-(1-Gam) * c1 * r) * sc.integrate.cumulative_simpson(c2 * np.exp((1-Gam) * c1 * r), x=r, initial=0.)
rho = (ksi_h + ksi_p)**(1/(1-Gam))

prs = sc.integrate.cumulative_simpson(-rho*g, x=r, initial=P0)
if len(prs[prs<0]) > 0: 
  print(f"negative pressure. (number negative: {len(prs[prs<0])})")
  sys.exit(0)

T = prs/rho
dTdr = np.gradient(T, r)
# dTdr = d_coef@T

## calcul luminosité
luminosity=0.01
n = 1.5
L_rad_factor = 0.7
# tr_factors = ((0.3, 0.1), (0.8, 0.05))
tr_factors = ((0.3, 0.1), (0.8, 0.05))
if True:
# if False:
  L_tot = sc.integrate.cumulative_simpson(rho * T**n, x=r, initial=0.)
  epsilon = 1/L_tot[-1] * luminosity
  L_tot *= epsilon
  # L_rad = tr(L_rad_factor*L_tot, L_tot, 0.8, 0.05)
  L_rad = tr(L_tot, tr(L_rad_factor*L_tot, L_tot, *tr_factors[1]), *tr_factors[0])

  # L_rad = tr(0, tr(L_rad_factor*L_tot, L_tot, *tr_factors[1]), *tr_factors[0])
  # L_rad = tr(0, tr(L_rad_factor*L_tot, L_tot, 0.65, 0.1), 0.15, 0.05)

  kappa = -L_rad / (dTdr)
  # kappa = -L_tot / (dTdr)
  # kappa[0] = 2*kappa[1] - kappa[2]  # remove nan

  F = L_tot
  F[0] = 2*F[1] - F[2]  # remove nan
  # kappa[0] = -F[0]/dTdr[0]
  kappa[0] = 0
else:
  L_tot = sc.integrate.cumulative_simpson(rho * T**n * r**2, x=r, initial=0.)
  epsilon = 1/L_tot[-1] * luminosity
  L_tot *= epsilon
  # L_rad = tr(L_rad_factor*L_tot, L_tot, 0.65, 0.1)
  L_rad = tr(L_tot, tr(L_rad_factor*L_tot, L_tot, *tr_factors[1]), *tr_factors[0])
  # L_rad = tr(0, tr(L_rad_factor*L_tot, L_tot, 0.65, 0.1), 0.075, 0.1)
  # L_rad = tr(0, tr(L_rad_factor*L_tot, L_tot, 0.65, 0.1), 0.15, 0.05)

  kappa = -L_rad / (dTdr * r**2)
  # kappa = -L_tot / (r**2 * dTdr)
  # kappa[0] = 2*kappa[1] - kappa[2]  # remove nan

  F = L_tot/r**2
  F[0] = 2*F[1] - F[2]  # remove nan
  kappa[0] = -F[0]/dTdr[0]

L_conv = L_tot - L_rad
heating = epsilon * T**n

# plt.plot(r, L_conv)
# plt.plot(r, L_rad)
plt.plot(r, L_conv/L_rad)
plt.show()
# sys.exit(0)


# print(F[0], F[-1])
# plt.plot(r, F)
# plt.plot(r, -kappa * dTdr)
# I1 = sc.integrate.cumulative_simpson(F + kappa * dTdr, x=r, initial=0.)
# plt.plot(r, I1)
# print("tot_flux_diff: ", I1[-1])
# print("flux_end ? : ", F[-1] - I1[-1])

heat_diff = np.gradient(F+kappa * dTdr, r)/rho
# plt.plot(r, heat_diff)
# plt.plot(r, heating)
# plt.plot(r, heating + np.gradient(F+kappa * dTdr, r))
# heating = heating + heat_diff

plt.plot(r, kappa )
plt.show()
plt.plot(r, np.gradient(-kappa * dTdr, r))
# plt.plot(r, d_coef@(-kappa * dTdr))
plt.plot(r, heating * rho)
# plt.plot(r, d_coef@(-kappa * dTdr) - heating * rho)
# plt.plot(r, -sc.integrate.cumulative_simpson(d_coef@(-kappa * dTdr) - heating * rho, x=r, initial=0.))

print("diff: ", *(d_coef@(-kappa * dTdr) - heating * rho)[[0,-1]])
print("F: ", *(d_coef@(-kappa * dTdr))[[0,-1]])
print("heat: ", *(heating * rho)[[0,-1]])
# plt.plot(r, -kappa * dTdr)
print((-kappa * dTdr)[0])
print(L_rad[0])
# plt.plot(r, sc.integrate.cumulative_simpson(heating * rho, x=r, initial=0.))


# plt.plot(r, L_tot)


plt.show()
# sys.exit(0)
# print(heating)

"""
# heating = np.gradient(L_rad, r) / (rho * r**2)
# kappa = -1/(dTdr * r**2) * L_rad

# heating[0] = 2*heating[1] - heating[2]  # remove nan
# kappa[0] = 2*kappa[1] - kappa[2]  # remove nan
"""

# plt.plot(r, heating)
# plt.plot(r, kappa)
# print(heating)
# print(kappa)
# plt.show()
# sys.exit(0)

"""
L_conv = L_tot-L_rad
# kappa = np.ones(N) * 0.01
# kappa = -epsilon * L_rad / dTdr 
kappa = -L_rad / (dTdr * 4*np.pi * r**2)
# kappa = -1 / dTdr # / (4*np.pi*r**2)
# kappa[0] = 0               # remove nan
# kappa[0] = kappa[1]               # remove nan
kappa[0] = 2*kappa[1] - kappa[2]  # remove nan

# heating = np.gradient(kappa * dTdr, r)
heating = -np.gradient(L_rad / (4*np.pi * r**2), r)
heating[1] = 2*heating[2] - heating[3]  # remove nan
heating[0] = 2*heating[1] - heating[2]  # remove nan
# print(heating)
print(kappa)
# print(L_rad)

# heating = epsilon * T**n
# kappa2 = -1/dTdr
# kappa2[kappa2>2] = 2
"""

# kappa = -1.0/dTdr
# print( kappa)

plt.plot(r, rho, label='rho')
plt.plot(r, g/g0, label='g/g0')
plt.plot(r, prs, label='prs')
# plt.plot(r, T, label='T')
# plt.plot(r, -dTdr, label="-dTdr")
# plt.plot(r, -rho*g, label='-rho*g')
# plt.plot(r, d_coef@prs, label='dprs/dr')

plt.plot(r, L_tot, label='L_tot')
plt.plot(r, L_rad, label='L_rad')
# plt.plot(r, L_conv/epsilon, label='L_conv')
# plt.plot(r, 1/kappa, label='inv_kappa')
# plt.plot(r, 1/heating, label='inv_heating')
# plt.plot(r, kappa/epsilon, label='kappa')
# plt.plot(r, heating, label='heating')

# tmp = (-np.gradient(L_rad, r)/(4*np.pi*r**2) + L_rad/(2*np.pi*r**3)) / rho
# tmp[0] = 2*tmp[1] - tmp[2]  # remove nan
# plt.plot(r, 1/tmp, label='inv_heating2')

S_eq = Cp/gamma * np.log(prs) - Cp * np.log(rho)
# plt.plot(r, S_eq, label='S2')
# plt.plot(r, S, label='S')
# plt.plot(r, np.gradient(S_eq, r), label='dSdr')
# plt.plot(r, np.gradient(S, r), label='dSdr')
# plt.plot(r, d_coef@(S_eq), label='dSdr')
# plt.plot(r, d_coef@(S), label='dSdr')

# factor_kappa = 100
factor_kappa = 1
limit_npoints = 1
with open("polyfit2.h", "w") as f:
  def dump_c_arr(name, arr, lp=limit_npoints):
    # poly = np.polynomial.polynomial.Polynomial.fit(r, arr, points-1).convert().coef
    # poly = np.polyfit(r, arr, points-1)[::-1]

    rr =  r[::lp]
    ar = arr[::lp]
    spl = sc.interpolate.CubicSpline(rr, ar)
    
    lines =  ''
    lines += 'KOKKOS_INLINE_FUNCTION\n'
    lines += f'real_t get_{name}(real_t x) {{\n'
    lines += '  static const real_t xs[] = {' + ', '.join([f'{float(c).hex()}' for c in spl.x]) + '};\n'
    lines += '  static const real_t coef[][4] = ' + '{{' + '}, {'.join([', '.join([f'{float(c).hex()}' for c in col[::-1]]) for col in spl.c.T]) + '}};\n'
    lines += '  const size_t n = sizeof(coef)/sizeof(*coef);\n'
    lines += '  size_t id = x*n;\n'
    lines += '  id = (id >= n) ? n-1 : (id < 0) ? 0 : id;\n'
    lines += '  const real_t xi = x-xs[id];\n'
    lines += '  const real_t *c = coef[id];\n'
    lines += '  return c[0] + xi * c[1] + xi*xi * c[2] + xi*xi*xi * c[3];\n'
    lines += '}\n\n'

    f.write(lines)

    # line = f'const real_t {name}_coef[][] = {{' + ', '.join([f'{c.hex()}' for i,c in enumerate(spl.c)])+ ' };\n'
    # print(', '.join([f'{float(c).hex()}' for c in spl.c[:,0]]))


    # print(line)
    # sys.exit(0)
    # line = f'const __device__ real_t {name}_coef[] = {{' + ', '.join([f'{c.hex()}' for i,c in enumerate(poly)])+ ' };\n'
    # line = f'const real_t {name}_coef[] = {{' + ', '.join([f'{c.hex()}' for i,c in enumerate(poly)])+ ' };\n'
    # print(line)
  dump_params = "// {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}\n// {:<20} {:<20} {:<20} {:<20} {:<20}\n\n".format(f"{N = }", f"{gamma = :.6f}", f"{Cp = }", f"{P0 = }", f"{rho0 = }", f"{g0 = }", f"{dSdr_LEFT = }", f"{dSdr_RIGHT = }", f"{r_cut_dSdr = }", f"{r_cut_grav = }", f"{dr = :.3e}")
  f.write(dump_params)
  print(dump_params)
  dump_c_arr('rho', rho)
  dump_c_arr('prs', prs)
  dump_c_arr('g', -g)
  # dump_c_arr('dTdr', dTdr)
  # dump_c_arr('inv_kappa', 1./kappa)
  dump_c_arr('kappa', kappa)
  dump_c_arr('heating', heating)
  # dump_c_arr('inv_heating', factor_kappa/heating)

# warnings.simplefilter('ignore', np.exceptions.RankWarning)
# def plotfit(name, arr, points=npoints):
#   # poly = np.polynomial.polynomial.Polynomial.fit(r, arr, points-1).convert().coef
#   poly = np.polyfit(r, arr, points-1)[::-1]
#   tmp = sum([r**i * c for i,c in enumerate(poly)])
#   plt.plot(r, tmp, '--'  ) ###, label=f'{name}_fit')
#   return tmp
# plotfit('g/g0', g/g0)
# plotfit('rho', rho)
# plotfit('prs', prs)
# # # plotfit('-dTdr', -dTdr)
# plotfit('inv_kappa', 1/kappa)
# # plotfit('inv_heating', 1/heating)
# # plotfit('kappa', kappa)
# # plotfit('heating', heating)



# print(f"""
# real_t poly_eval(real_t r, real_t *coef, size_t len){{
#   real_t ret = 0; 
#   real_t poly_r = 1.;
#   for(size_t i=0; i<len; i++){{
#     ret += poly_r * coef[i];
#     poly_r *= r;
#   }}
# }}""")

print('')
print("N_rho =", np.log(rho[0]/rho[-1]))
print(f"hydrostatic equilibrium RMS = {(np.mean((d_coef@prs + rho*g)**2))**0.5:e}", 
      f"\tmax = {np.max(np.abs(d_coef@prs + rho*g)):e}")
# print(f"hydrostatic equilibrium RMS = {(np.mean((np.gradient(prs, r) + rho*g)**2))**0.5:e}", 
#       f"\tmax = {np.max(np.abs(np.gradient(prs, r) + rho*g)):e}")

print(f"\n{'name':<10}:  {'inside':^12}\t{'outside':^12}\t   {'mag':^12}")
for arr,name in ((T, "T"), (dTdr, "dTdr"), (rho, "rho"), (prs, "prs"), (np.gradient(rho, r), "drhodr"), (np.gradient(prs, r), "dpdr"), 
                 (heating, "heating"), (kappa, "kappa"), (-kappa * dTdr, "flux th")):
  print(f"{name:<10}:  {arr[0]: 3.8e}\t{arr[-1]: 3.8e}\t   {np.log(abs(arr[0]/arr[-1])): 3.3e}")


if len(kappa[kappa<0]) > 0: 
  print(f"negative kappa. (number negative: {len(kappa[kappa<0])})")

plt.legend()
# plt.xlim((0.5,1))
# plt.ylim((-0.05,1.2))
plt.show()


# drho/dr + g/gamma * rho**(2-gamma) * E**(-gamma*S/Cp) + rho/Cp dS/dr = 0


