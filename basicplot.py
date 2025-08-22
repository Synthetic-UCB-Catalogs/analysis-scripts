import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


##
def P_from_a(m1, m2, a):
    Mtot = m1 + m2
    return 0.116*np.sqrt(a**3/(Mtot))


def point_plot(M1,M2,log_P,log_tau,mask_LISA,title,limits=None):

    Pclip = None
    M1clip = None
    M2clip = None
    tclip=None

    if (limits):
        Pclip = limits[0:2]
        M1clip = limits[2:4]
        M2clip = limits[4:6]
        tclip = limits[6:8]


   # Create the figure and grid for the subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 10), gridspec_kw={'height_ratios': [0.2, 1, 1], 'width_ratios': [0.2, 1, 1]})
    fig.suptitle(title,y=0.92)

    sns.kdeplot(x=log_P, ax=axs[0,1], fill=True, color='blue',cut=0,clip=Pclip)
    sns.kdeplot(y=M1, ax=axs[1,0], fill=True, color='blue',cut=0,clip=M1clip)
    sns.kdeplot(y=M2, ax=axs[2,0], fill=True, color='blue',cut=0,clip=M2clip)
    axs[2, 0].set_ylabel(r'$M_2 [M_\odot]$')
    axs[1, 0].set_ylabel(r'$M_1 [M_\odot]$')
    if (limits):
        axs[0, 1].set_xlim(Pclip[0],Pclip[1])
        axs[1, 0].set_ylim(M1clip[0],M1clip[1])
        axs[2, 0].set_ylim(M2clip[0],M2clip[1])

    # Top left: Plot of log_P vs M1
    axs[1, 1].plot(log_P, M1,'.')
    axs[1, 1].plot(log_P[mask_LISA], M1[mask_LISA],'y.')
    #axs[1, 1].set_yticks([])
    axs[1, 1].tick_params(direction="in")
    axs[1, 1].tick_params(left=True,top=True,labelbottom=False,labelleft=False)
    if (limits):
         axs[1, 1].set_xlim(Pclip[0],Pclip[1])
         axs[1, 1].set_ylim(M1clip[0],M1clip[1])

    # Bottom left: Plot of log_P vs M2
    axs[2, 1].plot(log_P, M2,'.')
    axs[2, 1].plot(log_P[mask_LISA], M2[mask_LISA],'y.')
    axs[2, 1].set_xlabel(r'$\log P$ [d]')
    axs[2, 1].yaxis.set_ticklabels([])
    axs[2, 1].tick_params(direction="in")

    if (limits):
         axs[2, 1].set_xlim(Pclip[0],Pclip[1])
         axs[2, 1].set_ylim(M2clip[0],M2clip[1])

    # Bottom right: Hexbin of M1 vs M2
    axs[2, 2].plot(M1, M2,'.')
    axs[2, 2].plot(M1[mask_LISA], M2[mask_LISA],'y.')
    axs[2, 2].set_xlabel(r'$M_1 [M_\odot]$')
    axs[2, 2].set_ylabel(r'$M_2 [M_\odot]$')
    #axs[2, 2].yaxis.tick_right()
    #axs[2, 2].yaxis.set_label_position("right")
    axs[2, 2].tick_params(left=True,right=True,labelright=True,labelleft=False)
    axs[2, 2].yaxis.set_label_position("right")


    if (limits):
         axs[2, 2].set_xlim(M1clip[0],M1clip[1])
         axs[2, 2].set_ylim(M2clip[0],M2clip[1])


    # Remove the space between the panels
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.delaxes(axs[1, 2])
    fig.delaxes(axs[0, 0])
    fig.delaxes(axs[0, 2])

    # Top right: KDE plot of log_tau
    # Create the figure and grid for the subplots
    ax_new = fig.add_axes([0.6, 0.55, 0.3, 0.25])
    sns.kdeplot(log_tau, ax=ax_new, fill=True, color='green',cut=0,clip=tclip)
    ax_new.set_xlabel(r'$\log \tau$ [Myr]')

    # Show the plot
    plt.savefig(title+'.png')
    #plt.show()

def hexbin_plot(M1,M2,log_P,log_tau,title,bins=None,colmap='Blues',Ngrid=30,limits=None):

    Pclip = None
    M1clip = None
    M2clip = None
    tclip=None
    extentPM1 = None
    extentPM2 = None
    extentM1M2 = None

    if (limits != None):
        Pclip = limits[0:2]
        M1clip = limits[2:4]
        M2clip = limits[4:6]
        tclip = limits[6:8]
        extentPM1 = limits[0:4]
        extentPM2 = limits[0:2]+limits[4:6]
        extentM1M2 = limits[2:6]

    fig, axs = plt.subplots(3, 3, figsize=(12, 10), gridspec_kw={'height_ratios': [0.2, 1, 1], 'width_ratios': [0.2, 1, 1]})
    fig.suptitle(title,y=0.92)

    sns.kdeplot(x=log_P, ax=axs[0,1], fill=True, color='blue',cut=0,clip=Pclip)
    sns.kdeplot(y=M1, ax=axs[1,0], fill=True, color='blue',cut=0,clip=M1clip)
    sns.kdeplot(y=M2, ax=axs[2,0], fill=True, color='blue',cut=0,clip=M2clip)
    axs[2, 0].set_ylabel(r'$M_2 [M_\odot]$')
    axs[1, 0].set_ylabel(r'$M_1 [M_\odot]$')

    # Top left: Hexbin of log_P vs M1
    hb = axs[1, 1].hexbin(log_P, M1, gridsize=Ngrid, cmap=colmap,bins=bins,mincnt=1,extent=extentPM1)
    #axs[1, 1].yaxis.set_ticklabels([])
    axs[1, 1].tick_params(direction="in")
    axs[1, 1].tick_params(left=True,top=True,labelbottom=False,labelleft=False)


    # Bottom left: Hexbin of log_P vs M2
    hb2 = axs[2, 1].hexbin(log_P, M2, gridsize=Ngrid, cmap=colmap,bins=bins,mincnt=1,extent=extentPM2)
    axs[2, 1].set_xlabel(r'$\log P$ [d]')
    axs[2, 1].yaxis.set_ticklabels([])
    axs[2, 1].tick_params(direction="in")


    # Bottom right: Hexbin of M1 vs M2
    hb3 = axs[2, 2].hexbin(M1, M2, gridsize=Ngrid, cmap=colmap,bins=bins,mincnt=1,extent=extentM1M2)
    axs[2, 2].set_xlabel(r'$M_1 [M_\odot]$')
    axs[2, 2].set_ylabel(r'$M_2 [M_\odot]$')
#    axs[2, 2].yaxis.tick_right()
    axs[2, 2].tick_params(left=True,right=True,labelright=True,labelleft=False)
    axs[2, 2].yaxis.set_label_position("right")

    # Remove the space between the panels
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.delaxes(axs[1, 2])
    fig.delaxes(axs[0, 0])
    fig.delaxes(axs[0, 2])

    # Top right: KDE plot of log_tau
    # Create the figure and grid for the subplots
    ax_new = fig.add_axes([0.6, 0.55, 0.3, 0.25])
    sns.kdeplot(log_tau, ax=ax_new, fill=True, color='green',cut=0,clip=tclip)
    ax_new.set_xlabel(r'$\log \tau$ [Myr]')

    # Show the plot
    plt.savefig(title+'.png')
    #plt.show()




