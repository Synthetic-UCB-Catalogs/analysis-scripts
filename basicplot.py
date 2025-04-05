import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


##
def P_from_a(m1, m2, a):
    Mtot = m1 + m2
    return 0.116*np.sqrt(a**3/(Mtot))


def point_plot(M1,M2,log_P,log_tau,mask_LISA,title):

    # Create the figure and grid for the subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 10), gridspec_kw={'height_ratios': [0.2, 1, 1], 'width_ratios': [0.2, 1, 1]})
    fig.suptitle(title,y=0.92)

    sns.kdeplot(x=log_P, ax=axs[0,1], fill=True, color='blue')
    sns.kdeplot(y=M1, ax=axs[1,0], fill=True, color='blue')
    sns.kdeplot(y=M2, ax=axs[2,0], fill=True, color='blue')
    axs[2, 0].set_ylabel(r'$M_2 [M_\odot]$')
    axs[1, 0].set_ylabel(r'$M_1 [M_\odot]$')

    # Top left: Plot of log_P vs M1
    axs[1, 1].plot(log_P, M1,'.')
    axs[1, 1].plot(log_P[mask_LISA], M1[mask_LISA],'y.')
    axs[1, 1].set_yticks([])

    # Bottom left: Plot of log_P vs M2
    axs[2, 1].plot(log_P, M2,'.')
    axs[2, 1].plot(log_P[mask_LISA], M2[mask_LISA],'y.')
    axs[2, 1].set_xlabel(r'$\log P$ [d]')
    axs[2, 1].set_yticks([])

    # Bottom right: Hexbin of M1 vs M2
    axs[2, 2].plot(M1, M2,'.')
    axs[2, 2].plot(M1[mask_LISA], M2[mask_LISA],'y.')
    axs[2, 2].set_xlabel(r'$M_1 [M_\odot]$')
    axs[2, 2].set_ylabel(r'$M_2 [M_\odot]$')
    axs[2, 2].yaxis.tick_right()
    axs[2, 2].yaxis.set_label_position("right")

    # Remove the space between the panels
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.delaxes(axs[1, 2])
    fig.delaxes(axs[0, 0])
    fig.delaxes(axs[0, 2])

    # Top right: KDE plot of log_tau
    # Create the figure and grid for the subplots
    ax_new = fig.add_axes([0.6, 0.55, 0.3, 0.25])
    sns.kdeplot(log_tau, ax=ax_new, fill=True, color='green')
    ax_new.set_xlabel(r'$\log \tau$ [Myr]')

    # Show the plot
    plt.savefig(title+'.png')
    #plt.show()

def hexbin_plot(M1,M2,log_P,log_tau,title,bins=None,colmap='Blues',Ngrid=30):

    fig, axs = plt.subplots(3, 3, figsize=(12, 10), gridspec_kw={'height_ratios': [0.2, 1, 1], 'width_ratios': [0.2, 1, 1]})
    fig.suptitle(title,y=0.92)

    sns.kdeplot(x=log_P, ax=axs[0,1], fill=True, color='blue')
    sns.kdeplot(y=M1, ax=axs[1,0], fill=True, color='blue')
    sns.kdeplot(y=M2, ax=axs[2,0], fill=True, color='blue')
    axs[2, 0].set_ylabel(r'$M_2 [M_\odot]$')
    axs[1, 0].set_ylabel(r'$M_1 [M_\odot]$')

    # Top left: Hexbin of log_P vs M1
    hb = axs[1, 1].hexbin(log_P, M1, gridsize=Ngrid, cmap=colmap,bins=bins,mincnt=1)
    axs[1, 1].set_yticks([])
    #fig.colorbar(hb, ax=axs[0, 0])

    # Bottom left: Hexbin of log_P vs M2
    hb2 = axs[2, 1].hexbin(log_P, M2, gridsize=Ngrid, cmap=colmap,bins=bins,mincnt=1)
    axs[2, 1].set_xlabel(r'$\log P$ [d]')
    axs[2, 1].set_yticks([])
    #fig.colorbar(hb2, ax=axs[1, 0])

    # Bottom right: Hexbin of M1 vs M2
    hb3 = axs[2, 2].hexbin(M1, M2, gridsize=Ngrid, cmap=colmap,bins=bins,mincnt=1)
    axs[2, 2].set_xlabel(r'$M_1 [M_\odot]$')
    axs[2, 2].set_ylabel(r'$M_2 [M_\odot]$')
    axs[2, 2].yaxis.tick_right()
    axs[2, 2].yaxis.set_label_position("right")

    # Remove the space between the panels
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.delaxes(axs[1, 2])
    fig.delaxes(axs[0, 0])
    fig.delaxes(axs[0, 2])

    # Top right: KDE plot of log_tau
    # Create the figure and grid for the subplots
    ax_new = fig.add_axes([0.6, 0.55, 0.3, 0.25])
    sns.kdeplot(log_tau, ax=ax_new, fill=True, color='green')
    ax_new.set_xlabel(r'$\log \tau$ [Myr]')

    # Show the plot
    plt.savefig(title+'.png')
    #plt.show()




