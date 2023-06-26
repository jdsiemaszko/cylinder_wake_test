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
from blobs_solver2.blobs.base.induced import vorticity
import matplotlib.pyplot as plt

case_dir = os.getcwd()
data_dir = os.path.join(case_dir,'data_single_2')
ref_dir = os.path.join(case_dir,'data_single_ref_2')
plots_dir =  os.path.join(case_dir, 'comparison_tutty')

Path(plots_dir).mkdir(parents=True, exist_ok=True)

case = "single_patch"
case_ref = "single_patch_ref"
gammaC = 0.
nTimeSteps = 500
writeInterval_plots = 25
coreSize = 'variable'
deltaTc = 0.02

length = 200
xmin, xmax, ymin, ymax = 17., 30., -2., 2.

xplot,yplot = np.meshgrid(np.linspace(xmin,xmax,length),np.linspace(ymin,ymax,length))
xplotflat = xplot.flatten()
yplotflat = yplot.flatten()

# uxNorm = np.array([])
# uyNorm = np.array([])
# omegaNorm = np.array([])
# t_norm = np.array([])
#Line plots
times_file = os.path.join(data_dir,"times_{}.csv".format(case))
times_ref_file = os.path.join(ref_dir,"times_{}.csv".format(case_ref))
times_data = pandas.read_csv(times_file)
times_ref_data = pandas.read_csv(times_ref_file)

time1 = times_data['Time']
noBlobs1 = times_data['NoBlobs']
evolution_time1 = times_data['Evolution_time']
circulation1 = times_data['Circulation']

time2 = times_ref_data['Time']
noBlobs2 = times_ref_data['NoBlobs']
evolution_time2 = times_ref_data['Evolution_time']
circulation2 = times_ref_data['Circulation']

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time1,noBlobs1, label=case)
ax.plot(time2,noBlobs2, label=case_ref)

plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Total number of particles')
plt.ylabel('Particles')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/number_of_particles_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time1,circulation1- gammaC, label='Run w/ Compression')
ax.plot(time2,circulation2- gammaC, label='Run w/o Compression')

plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Circulation')
plt.ylabel('circulation')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/circulation_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

fig = plt.subplots(figsize=(6,6))
index = np.arange(len(evolution_time1))
width = deltaTc
lagrangian = plt.bar(index[1:nTimeSteps]*deltaTc, evolution_time1[1:nTimeSteps] - evolution_time2[1:nTimeSteps], width)
plt.ylabel('Time (s)')
plt.xlabel('Simulation time(s)')
plt.title('Evolution time difference')
plt.savefig("{}/times_dif_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

time = []
L2error = []
Linferror = []
for timeStep in range(nTimeSteps+1):
    if timeStep%writeInterval_plots == 0:
        ####Fields
        # lagrangian_file = os.path.join(data_dir,'results_{}_{n:06d}.csv'.format(case,n=timeStep))
        # lagrangian_data = np.genfromtxt(lagrangian_file)

        # lagrangian_ref_file = os.path.join(ref_dir,'results_{}_{n:06d}.csv'.format(case_ref,n=timeStep))
        # lagrangian_ref_data = np.genfromtxt(lagrangian_ref_file)

        # xplot = lagrangian_data[:,0]
        # yplot = lagrangian_data[:,1]
        # length = int(np.sqrt(len(xplot)))
        # xPlotMesh = xplot.reshape(length,length)
        # yPlotMesh = yplot.reshape(length,length)

        # lagrangian_ux = lagrangian_data[:,2]
        # lagrangian_uy = lagrangian_data[:,3]
        # lagrangian_omega = lagrangian_data[:,4]

        # xplot_ref = lagrangian_ref_data[:,0]
        # yplot_ref = lagrangian_ref_data[:,1]
        # length_ref = int(np.sqrt(len(xplot_ref)))
        # xPlotMesh_ref = xplot.reshape(length,length)
        # yPlotMesh_ref = yplot.reshape(length,length)

        # lagrangian_ref_ux = lagrangian_ref_data[:,2]
        # lagrangian_ref_uy = lagrangian_ref_data[:,3]
        # lagrangian_ref_omega = lagrangian_ref_data[:,4]

        # ux, uy = blobs.evaluateVelocity(xplotflat,yplotflat)

        blobs_file = os.path.join(data_dir,'blobs_{}_{n:06d}.csv'.format(case,n=timeStep))
        blobs_data = np.genfromtxt(blobs_file)
        blobs_x = blobs_data[:,0]
        blobs_y = blobs_data[:,1]
        blobs_g = blobs_data[:,2]
        if coreSize == 'variable':
            blobs_sigma = blobs_data[:,3]
        else:
            blobs_sigma = sigma_ref
        # print(blobs_x, blobs_y, blobs_g, blobs_sigma)

        blobs_ref_file = os.path.join(ref_dir,'blobs_{}_{n:06d}.csv'.format(case_ref,n=timeStep))
        blobs_ref_data = np.genfromtxt(blobs_ref_file)
        ref_x = blobs_ref_data[:,0]
        ref_y = blobs_ref_data[:,1]
        ref_g = blobs_ref_data[:,2]
        if coreSize == 'variable':
            ref_sigma = blobs_ref_data[:,3]
        else:
            ref_sigma = sigma_ref
        # print(ref_x, ref_y, ref_g, ref_sigma)

        # print(sum(blobs_g) - sum(ref_g))


        omega = vorticity(np.array(blobs_x), np.array(blobs_y), np.array(blobs_g), np.array(blobs_sigma), xEval = np.array(xplotflat), yEval = np.array(yplotflat))
        omega_ref = vorticity(np.array(ref_x), np.array(ref_y), np.array(ref_g), np.array(ref_sigma), xEval = np.array(xplotflat), yEval = np.array(yplotflat))
        # print(omega, omega_ref)
        # print(np.linalg.norm(omega-omega_ref))
        # print(max(omega-omega_ref), max(np.abs(omega_ref)), max(np.abs(omega)))
        time.append(timeStep)
        L2error.append(np.linalg.norm(omega-omega_ref))
        Linferror.append(max(np.abs(omega-omega_ref)))
        
        xPlotMesh = xplotflat.reshape(length, length)
        yPlotMesh = yplotflat.reshape(length, length)
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
        # BEWARE OF SKETCHY SOLUTION FOR LEVELS
        e_abs = (omega.reshape(length,length) - omega_ref.reshape(length,length))
        e_abs_max = np.max(np.abs(e_abs))
        step = e_abs_max / 50
        try:
            cax = ax.contourf(xPlotMesh,yPlotMesh, e_abs ,levels=np.arange(-e_abs_max, e_abs_max+step, step),cmap='RdBu',extend="both")
            cbar = fig.colorbar(cax,format="%.4f")
            cbar.set_label("Absolute Vorticity Error (1/s)")
            plt.tight_layout()
            plt.savefig("{}/absolute_vorticity_error_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except:
            pass

        fig, ax = plt.subplots(1,1,figsize=(12,6))
        ax.set_aspect("equal")
        # ax.set_xticks(xTicks)
        # ax.set_yticks(yTicks)
        plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # BEWARE OF SKETCHY SOLUTION FOR LEVELS
        e_rel = (omega.reshape(length,length) - omega_ref.reshape(length,length))/ max(omega_ref) * 100
        e_rel_max = np.max(np.abs(e_rel))
        step_rel = e_rel_max/50
        try:
            cax = ax.contourf(xPlotMesh,yPlotMesh, e_rel ,levels=np.arange(-e_rel_max, e_rel_max+step_rel, step_rel),cmap='RdBu',extend="both")
            cbar = fig.colorbar(cax,format="%.4f")
            cbar.set_label("Relative Vorticity Error (%)")
            plt.tight_layout()
            plt.savefig("{}/relative_vorticity_error_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except:
            pass
        # fig, ax = plt.subplots(1,1,figsize=(12,6))
        # ax.set_aspect("equal")
        # # ax.set_xticks(xTicks)
        # # ax.set_yticks(yTicks)
        # plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
        # plt.minorticks_on()
        # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # cax = ax.contourf(xPlotMesh,yPlotMesh,lagrangian_ux.reshape(length,length) - lagrangian_ref_ux.reshape(length,length),levels=100,cmap='RdBu',extend="both")
        # cbar = fig.colorbar(cax,format="%.4f")
        # cbar.set_label("Velocity Error (1/s)")
        # plt.tight_layout()
        # plt.savefig("{}/velocity_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
        # plt.close(fig)


#### Blobs distribution

        # blobs_file = os.path.join(data_dir,'blobs_{}_{n:06d}.csv'.format(case,n=timeStep))
        # blobs_data = np.genfromtxt(blobs_file)

        # blobs_x = blobs_data[:,0]
        # blobs_y = blobs_data[:,1]
        # blobs_g = blobs_data[:,2]

        # if coreSize == 'variable':
        #     blobs_sigma = blobs_data[:,3]

        #     fig, ax = plt.subplots(1,1,figsize=(6,6))
        #     ax.scatter(blobs_x,blobs_y,c=blobs_g, s= blobs_sigma*30)
        #     plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
        #     plt.close(fig)
        # else:
        #     fig, ax = plt.subplots(1,1,figsize=(6,6))
        #     ax.scatter(blobs_x,blobs_y,c=blobs_g, s=0.2)
        #     plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
        #     plt.close(fig)
plt.clf()
fig, ax1 = plt.subplots(figsize=(6,6))
ax1.plot(time, L2error, label='L2', color = 'tab:blue')
ax2 = ax1.twinx()
ax2.plot(time, Linferror, label='Linf', color='tab:red')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('L2 Error', color='tab:blue')
ax2.set_ylabel('Linf error', color='tab:red')
plt.title('Error Evolution Over Time')
plt.savefig("{}/error_evolution_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")