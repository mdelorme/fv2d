import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
# from d_coef import get_d_coef
import sys
import os
import warnings
import signal
import inspect
import struct
from array import array
from datetime import datetime

signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))

#
filename_data = sys.argv[1]
Nrho_target   = float(sys.argv[2]) if len(sys.argv) > 2 else None # enter targeted N_rho 
#

# load
with open(filename_data, "rb") as f:
  header_str = '@IIIIIIIddddddd'
  sizeof_header = struct.calcsize(header_str)
  len_spline, _, _, _, _, _, _, rmin, dr, r_cz, rsun, Cp, R, gamma = struct.unpack(header_str, f.read(sizeof_header))
  r = np.array([rmin + dr * i for i in range(len_spline+1)])
  
  def load_next_array():
    spl = sc.interpolate.CubicSpline(r, np.zeros_like(r))
    ar = np.array(array('d', f.read(4*8*len_spline))).reshape(len_spline, 4)
    spl.c = ar.T[::-1]
    return spl(r), spl

  rho    , spl_rho     =  load_next_array()
  prs    , spl_prs     =  load_next_array()
  g      , spl_g       =  load_next_array()
  kappa  , spl_kappa   =  load_next_array()
  heating, spl_heating =  load_next_array()
  N_rho  , spl_N_rho   =  load_next_array()

# print(r[[0,-1]])
# print(spl_rho(0))
# print(spl_rho(rcut))
# print(np.log(spl_rho(0)/spl_rho(rcut)))
# print(np.log(spl_rho(0)/spl_rho(r[-1])))

# def get_ar(ar): return ar/max(ar)
def get_ar(ar): return ar
# index_cv=np.searchsorted(r, r_cz)
# index_cut=np.searchsorted(r, rcut)

# N_rho=np.log(rho[index_cv]/rho)
S_eq = Cp/gamma * np.log(prs) - Cp * np.log(rho)
dSdr = np.gradient(S_eq, r)

index_cz  = np.searchsorted(r, r_cz)
index_cut = np.searchsorted(N_rho, Nrho_target) if Nrho_target else -1
rmax = r[index_cut]
H_rho = -1/np.gradient(np.log(rho), r)
Hrho_surface = H_rho[index_cut]
msun=1.989e33
masse_tot = 4*np.pi*sc.integrate.simpson(rho*r**2, x=r)
masse_cut = 4*np.pi*sc.integrate.simpson(rho[:index_cut]*r[:index_cut]**2, x=r[:index_cut])

print(f'        File: {filename_data}')
print(f'    Nrho_max: {N_rho[-1]}')
print(f' Nrho_target: {Nrho_target}')
print(f'Hrho_surface: {Hrho_surface:.12e}    (r=rmax)')
print(f'        rsun: {rsun:.12e}')
print(f'        rmax: {rmax:.12e}    (Nrho = {N_rho[index_cut]:.5})')
print(f'        r_cz: {r_cz:.12e}    (r_cz/rsun = {r_cz/rsun:.4})')
print(f'       m_tot: {masse_tot:.12e}    (m_tot/msun = {masse_tot/msun:.4})')
# print(f'       m_cut: {masse_cut:.12e}    (m_cut/msun = {masse_cut/msun:.4})')
print(f'          Cp: {Cp:.12e}')
print(f'           R: {R:.12e}')
print(f'       gamma: {gamma:.12e}')
# print(f'N_rho: {N_rho[index_cut]:f}')

# plt.plot(r/rsun, H_rho)
# plt.xlim(0.7, 1.01)
# plt.ylim(0.1e7, 1e10)

# # Hsinv = dSdr[-1]/Cp - 1/gamma * g[-1] * np.exp(-S_eq[-1]/Cp)/(rho[-1] * np.exp(S_eq[-1]/Cp))**(gamma-1)
# # Hsinv = dSdr[-1]/Cp - 1/gamma * g[-1]/rho[-1]**(gamma-1) * np.exp(-gamma*S_eq[-1]/Cp)
# # print(1/Hsinv)
# # print(H_rho[-1])

# # plt.show()
# # sys.exit(0)

print(f"\n{'name':<10}:  {'inside':^12}\t{'outside':^12}\t  {'r_cz':^12}\t {'rmax':^12}\t   {'ratio':^12}\t  {'max':^12}\t    {'min':^12}")
fields = [("rho", rho), ("prs", prs), ("T", prs/rho/R), ("g", -g), ("kappa", kappa), ("heating", heating)]
with np.errstate(divide='ignore', invalid='ignore'):
  for name, arr in fields:
    print(f"{name:<10}:  {arr[0]: 3.8e}\t{arr[-1]: 3.8e}\t{arr[index_cz]: 3.8e}\t  {arr[index_cut]: 3.3e}\t   {arr[index_cz]/arr[index_cut]: 3.3e}\t{max(arr): 3.8e}\t  {min(arr): 3.8e}")
rho = get_ar(rho); prs = get_ar(prs); g = get_ar(g); kappa = get_ar(kappa); heating = get_ar(heating)

# plots
fig, axs = plt.subplots(3,3)
axs = axs.flatten()
fields.append(("$N_\\rho$", N_rho))
fields.append(("$S$", S_eq))
fields.append(("$\\frac{dS}{dr}$", dSdr))

for i, f in enumerate(fields):
  ax = axs[i]
  name, field = f
  ax.grid()
  ax.set_title(name)
  # ax.set_xlabel('r')
  # ax.set_ylabel(name)
  ax.axvline(x=rmax/rsun, c='g', lw=0.7)
  ax.axvline(x=r_cz/rsun, c='r', lw=0.7)
  ax.plot(r/rsun, field, lw=0.8)

axs[-1].set_ylim(-1e-3, 0.02)
# axs[6].axhline(y=Nrho_target, c='g', lw=0.7)

# S_eq = Cp/gamma * np.log(prs) - Cp * np.log(rho)
# dSdr = np.gradient(S_eq, r)
# T = prs/rho /R
# # plt.plot(r, S_eq,     label="S")
# # plt.plot(r, dSdr,     label="dSdr")
# # print(f"{min(r):e}, {max(r):e}")

# # plt.plot(r, rho,     label="rho")
# # r = np.linspace(0, 6.8880350398044205e10, 1024)
# # plt.plot(r, spl_rho(r),     label="rho")
# # plt.plot(r, prs,     label="prs")
# axs[0].plot(r, T, label="T")
# # plt.plot(r, g,       label="g")
# # plt.plot(r, kappa,   label="kappa")
# # plt.plot(r, heating, label="heating")

# # plt.plot(r, g/max(g),       label="g")
# # plt.plot(r, rho/max(rho),       label="rho")
# # plt.plot(r, prs/max(prs),       label="prs")
# # plt.plot(r, T/max(T),       label="T")

# plt.legend()
plt.show()
