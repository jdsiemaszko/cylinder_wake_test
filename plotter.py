import os
import sys
import yaml
import pandas
import numpy as np
import csv
import re
import timeit
from pathlib import Path
import blobs_solver2 as pHyFlow
import matplotlib.pyplot as plt

case_dir = os.getcwd()
data_dir = os.path.join(case_dir,'data_single_ref')
plots_dir = os.path.join(case_dir,'plots_single_ref')
case = "single_patch_ref"
gammaC = 0.
nTimeSteps = 500
writeInterval_plots = 25
coreSize = 'variable'
deltaTc = 0.02

uxNorm = np.array([])
uyNorm = np.array([])
omegaNorm = np.array([])
t_norm = np.array([])
#Line plots
times_file = os.path.join(data_dir,"times_{}.csv".format(case))
times_data = pandas.read_csv(times_file)

time = times_data['Time']
noBlobs = times_data['NoBlobs']
evolution_time = times_data['Evolution_time']
circulation = times_data['Circulation']

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time,noBlobs, label='No of Particles')
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Total number of particles')
plt.ylabel('Particles')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/number_of_particles_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time,circulation- gammaC, label='Circulation deficit')
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('absolute error in circulation')
plt.ylabel('circulation')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/circulation_error_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

fig = plt.subplots(figsize=(6,6))
index = np.arange(len(evolution_time))
width = 0.8
lagrangian = plt.bar(index[1:]*deltaTc, evolution_time[1:], width)
plt.ylabel('Time (s)')
plt.xlabel('Simulation time (s)')
plt.title('Evolution time')
plt.savefig("{}/times_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

for timeStep in range(nTimeSteps+1):
    if timeStep%writeInterval_plots == 0:
        ####Fields
        lagrangian_file = os.path.join(data_dir,'results_{}_{n:06d}.csv'.format(case,n=timeStep))
        lagrangian_data = np.genfromtxt(lagrangian_file)

        xplot = lagrangian_data[:,0]
        yplot = lagrangian_data[:,1]
        length = int(np.sqrt(len(xplot)))
        xPlotMesh = xplot.reshape(length,length)
        yPlotMesh = yplot.reshape(length,length)

        lagrangian_ux = lagrangian_data[:,2]
        lagrangian_uy = lagrangian_data[:,3]
        lagrangian_omega = lagrangian_data[:,4]

        # xTicks = np.linspace(-2,2,5)
        # yTicks = np.linspace(-2,2,5)

        fig, ax = plt.subplots(1,1,figsize=(12,6))
        ax.set_aspect("equal")
        # ax.set_xticks(xTicks)
        # ax.set_yticks(yTicks)
        plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cax = ax.contourf(xPlotMesh,yPlotMesh,lagrangian_omega.reshape(length,length),levels=100,cmap='RdBu',extend="both")
        cbar = fig.colorbar(cax,format="%.4f")
        cbar.set_label("Vorticity (1/s)")
        plt.tight_layout()
        plt.savefig("{}/vorticity_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(1,1,figsize=(12,6))
        ax.set_aspect("equal")
        # ax.set_xticks(xTicks)
        # ax.set_yticks(yTicks)
        plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cax = ax.contourf(xPlotMesh,yPlotMesh,lagrangian_ux.reshape(length,length),levels=100,cmap='RdBu',extend="both")
        cbar = fig.colorbar(cax,format="%.4f")
        cbar.set_label("Velocity (1/s)")
        plt.tight_layout()
        plt.savefig("{}/velocity_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
        plt.close(fig)


#### Blobs distribution

        blobs_file = os.path.join(data_dir,'blobs_{}_{n:06d}.csv'.format(case,n=timeStep))
        blobs_data = np.genfromtxt(blobs_file)

        blobs_x = blobs_data[:,0]
        blobs_y = blobs_data[:,1]
        blobs_g = blobs_data[:,2]

        if coreSize == 'variable':
            blobs_sigma = blobs_data[:,3]

            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.scatter(blobs_x,blobs_y,c=blobs_g, s= blobs_sigma*30)
            plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.scatter(blobs_x,blobs_y,c=blobs_g, s=0.2)
            plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)