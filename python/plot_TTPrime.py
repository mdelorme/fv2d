import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

import matplotlib as mpl
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

COLOR = 'k'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#set_up
t_unit = 60 # 1 minute
cs_unit = 7100 # photosphere sound speed m/s
d_unit = t_unit*cs_unit
a_unit = d_unit/t_unit**2
solar_g = 274/a_unit
print('g = '+str(round(solar_g,1)))
solar_10Mm = 10e6/d_unit
print('10Mm = '+str(round(solar_10Mm,1)))



class fv2d_output():
    def __init__(self,dir,y1,y2):

        self.filelist = sorted(glob.glob(dir+'/*.h5'))
        self.Nf = len(self.filelist)
        f = h5py.File(self.filelist[0], 'r')
        self.x = np.unique(np.array(f['x']))[1:]
        self.y = np.unique(np.array(f['y']))[1:]
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.y1 = y1
        self.y2 = y2
        self.y_depth = (self.y-self.y1)*d_unit/1e6
        self.x_width = (self.x)*d_unit/1e6

    def returnVariable(self, snap, var):
        if var == 'T':
            prs = self.returnVariable(snap,'prs')
            rho = self.returnVariable(snap,'rho')
            return prs / rho
        elif var == 'mach':
            v = self.returnVariable(snap,'v')
            u = self.returnVariable(snap,'u')
            cs = np.sqrt(self.returnVariable(snap,'T'))
            return np.sqrt(v*v+u*u) / cs
        else:
            f = h5py.File(self.filelist[snap], 'r')
            return np.array(f[var]).reshape((self.Ny, self.Nx))

    def returnTime(self, snap):
        return h5py.File(self.filelist[snap], 'r').attrs['time']




ystart = 0
yend = 13.6820142565237
y1 = 1.0
y2 = 9.755395581941308
RHO0 = 100
dir = '/local/home/lb281911/runs/THREE_LAYER/fivewaves_MHD/'


file = fv2d_output(dir=dir,y1=y1,y2=y2)


snap = file.Nf-1

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(12,3))
ax1.plot(file.y_depth,np.mean(file.returnVariable(0,'T'),axis=1)*5800,c='k')
ax1.plot(file.y_depth,np.mean(file.returnVariable(snap,'T'),axis=1)*5800,c='r')
ax2.plot(file.y_depth,np.mean(file.returnVariable(0,'rho'),axis=1)/100,c='k')
ax2.plot(file.y_depth,np.mean(file.returnVariable(snap,'rho'),axis=1)/100,c='r')
ax3.plot(file.y_depth,np.mean(file.returnVariable(0,'prs'),axis=1)/100,c='k')
ax3.plot(file.y_depth,np.mean(file.returnVariable(snap,'prs'),axis=1)/100,c='r')
ax4.plot(file.y_depth,np.mean(file.returnVariable(0,'mach'),axis=1),c='k')
ax4.plot(file.y_depth,np.mean(file.returnVariable(snap,'mach'),axis=1),c='r')
ax1.text(0.01,0.15, 't={:.1f}'.format(file.returnTime(0)),ha='left',va='top',fontsize=12,transform=ax1.transAxes,c='k')
ax1.text(0.01,0.08, 't={:.1f}'.format(file.returnTime(snap)),ha='left',va='top',fontsize=12,transform=ax1.transAxes,c='r')
ax1.text(0.5,1.01, 'Temperature [K]' ,ha='center',va='bottom',fontsize=12,transform=ax1.transAxes,c='k')
ax2.text(0.5,1.01, 'Density [surface]' ,ha='center',va='bottom',fontsize=12,transform=ax2.transAxes,c='k')
ax3.text(0.5,1.01, 'Pressure [surface]' ,ha='center',va='bottom',fontsize=12,transform=ax3.transAxes,c='k')
ax4.text(0.5,1.01, 'Mach' ,ha='center',va='bottom',fontsize=12,transform=ax4.transAxes,c='k')
for ax in [ax1,ax2,ax3]:
    ax.semilogy()
for ax in fig.axes:
    ax.set_xlim((yend-y1)*d_unit/1e6,(ystart-y1)*d_unit/1e6)
    ax.axvline(x=0*d_unit/1e6,c='k',ls='--',lw=1)
    ax.axvline(x=y2*d_unit/1e6,c='k',ls='--',lw=1)
    ax.set_xlabel('Depth [Mm]',fontsize=12)
plt.tight_layout()
plt.savefig(dir+'/VerticalProfiles.png',bbox_inches=0)


for snap in tqdm(np.arange(file.Nf-1)):

    ext = [file.xmin*d_unit/1e6, file.xmax*d_unit/1e6, (file.ymin-file.y1)*d_unit/1e6, (file.ymax-file.y1)*d_unit/1e6]

    snap4 = '{:04}'.format(snap)
    fig, ax = plt.subplot_mosaic('''C''', figsize=(8, 6))
    ax['C'].text(0.01,1.01,'T-<T>',c='k',fontsize=10,ha='left',va='bottom',transform=ax['C'].transAxes)
    time = file.returnTime(snap)
    ax['C'].text(0.99,1.01,'t='+str(int(time))+' mins',c='k',fontsize=10,ha='right',va='bottom',transform=ax['C'].transAxes)
    T = file.returnVariable(snap,'T')
    averageT = np.tile(np.nanmean(file.returnVariable(snap,'T'),axis=1),(file.Nx, 1)).T
    VAR = 0.01
    C = ax['C'].imshow((T-averageT),extent=ext, vmin=-VAR,vmax=VAR,cmap='afmhot',origin='lower')
    ax['C'].contour(file.returnVariable(snap,'rho'),extent=ext,levels=[RHO0],colors=['w'],linewidths=[0.5],alpha=0.5)
    ax['C'].set_aspect('equal')
    tickspace = int((np.ceil(file.xmax)-np.floor(file.xmin))*d_unit/1e6/8)
    ax['C'].set_xticks(np.arange(np.floor(file.xmin*d_unit/1e6),np.ceil(file.xmax*d_unit/1e6)+0*tickspace,tickspace))
    ytickspace = int((np.ceil(file.ymax)-file.y1)*d_unit/1e6/4)
    ax['C'].set_yticks(np.arange(0,np.ceil(yend)*d_unit/1e6,ytickspace))
    ax['C'].set_ylim((yend-file.y1)*d_unit/1e6,-file.y1*d_unit/1e6)
    ax['C'].axhline(y=file.y2*d_unit/1e6,c='w',ls='--',lw=0.5)
    ax['C'].axhline(y=0*d_unit/1e6,c='w',ls='--',lw=0.5)
    ax['C'].set_xlabel('Width [Mm]',fontsize=12)
    ax['C'].set_ylabel('Depth [Mm]',fontsize=12)
    divider = make_axes_locatable(ax['C'])
    cax = divider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(C, cax=cax)
    plt.tight_layout()
    ax['C'].set_facecolor('lightgrey')
    plt.savefig(dir+'/img_{:04}.png'.format(snap))
    plt.close('all')
if os.path.exists(dir+'/fv2d_avT.mp4')==True:
    os.system('rm -r ' +dir+ '/fv2d_avT.mp4')
os.system('ffmpeg -r 24 -pattern_type glob -i "'+dir+'/img_*.png" -s:v 1890x970 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p '+dir+'/fv2d_avT.mp4')
os.system('rm -r ' +dir+ '/img_*.png')



v_unit=d_unit/t_unit
for snap in np.arange(file.Nf-1):

    ext = [file.xmin*d_unit/1e6, file.xmax*d_unit/1e6, (file.ymin-file.y1)*d_unit/1e6, (file.ymax-file.y1)*d_unit/1e6]

    snap4 = '{:04}'.format(snap)
    fig, ax = plt.subplot_mosaic('''C''', figsize=(8, 6))
    v = -file.returnVariable(snap,'v')
    C = ax['C'].imshow(np.flipud(v)*v_unit/1e3,extent=ext, cmap='bwr',vmin=-0.8,vmax=0.8)
    X,Y = np.meshgrid(file.x_width,file.y_depth)
    Bx = file.returnVariable(snap,'bx')
    By = -file.returnVariable(snap,'by')
    ax['C'].streamplot(X,Y,Bx,By,density=1,color='k',linewidth=0.5)
    ax['C'].contour(file.returnVariable(snap,'rho'),extent=ext,levels=[RHO0],colors=['k'],linewidths=[0.5],alpha=0.5)
    ax['C'].text(0.01,1.01,'vertical velocity km/s',c='k',fontsize=10,ha='left',va='bottom',transform=ax['C'].transAxes)
    time = file.returnTime(snap)
    ax['C'].text(0.99,1.01,'t='+str(int(time))+' mins',c='k',fontsize=10,ha='right',va='bottom',transform=ax['C'].transAxes)
    ax['C'].set_aspect('equal')
    tickspace = int((np.ceil(file.xmax)-np.floor(file.xmin))*d_unit/1e6/8)
    ax['C'].set_xticks(np.arange(np.floor(file.xmin*d_unit/1e6),np.ceil(file.xmax*d_unit/1e6)+0*tickspace,tickspace))
    ax['C'].set_xlim(file.xmin*d_unit/1e6,file.xmax*d_unit/1e6)
    ytickspace = int((np.ceil(file.ymax)-file.y1)*d_unit/1e6/4)
    ax['C'].set_yticks(np.arange(0,np.ceil(yend)*d_unit/1e6,ytickspace))
    ax['C'].set_ylim((yend-file.y1)*d_unit/1e6,-file.y1*d_unit/1e6)
    ax['C'].axhline(y=file.y2*d_unit/1e6,c='k',ls='--',lw=0.5)
    ax['C'].axhline(y=0*d_unit/1e6,c='k',ls='--',lw=0.5)
    ax['C'].set_xlabel('Width [Mm]',fontsize=12)
    ax['C'].set_ylabel('Depth [Mm]',fontsize=12)
    divider = make_axes_locatable(ax['C'])
    cax = divider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(C, cax=cax)
    plt.tight_layout()
    ax['C'].set_facecolor('lightgrey')
    plt.savefig(dir+'/img_{:04}.png'.format(snap))
    plt.close('all')
if os.path.exists(dir+'/fv2d_vz.mp4')==True:
    os.system('rm -r ' +dir+ '/fv2d_vz.mp4')
os.system('ffmpeg -r 24 -pattern_type glob -i "'+dir+'/img_*.png" -s:v 1890x970 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p '+dir+'/fv2d_vz.mp4')
os.system('rm -r ' +dir+ '/img_*.png')
