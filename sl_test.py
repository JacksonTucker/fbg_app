import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt

st.write('Interactive FBG Spectra')

# Input interactions
kappa = st.slider("𝜅 - Modulation intensity (/m)", 100,500,250,1)
l_mm = st.slider("l - Length of grating (mm)", 20,50,35,1)
cwl = st.slider("λ - Central wavelength (nm)", 1500,1600,1550,1)
n_pi = st.radio("n - Number of π phase shifts", [0,1,3,5,7], index=0, horizontal=True)
# Choice of X-axis (removed in favour of double labelling)
#xoptions = ["GHz","pm"]
#xchoice = st.radio("X axis",xoptions, horizontal = True)

# Plotting inputs
cwl_m = cwl*1E-9
span = 200E-12
pts = 1000
xs = np.linspace(-span,span,pts)+cwl_m

# Minimal versions of necessary functions
def transfer_matrix(wl, L, p, n_eff, dn_eff):
    wl_b = 2*n_eff*p
    D = 2*np.pi*n_eff*(1/wl - 1/wl_b)
    sig = 2*np.pi/wl*dn_eff
    S = D + sig
    K = np.pi/wl*dn_eff
    G = sqrt(K**2-S**2)
    transfer_m = np.array([[np.cosh(G*L)-1j*S/G*np.sinh(G*L), -1j*K/G*np.sinh(G*L)],[1j*K/G*np.sinh(G*L), np.cosh(G*L)+ 1j*S/G*np.sinh(G*L)]])
    return transfer_m

def period(wl_b, n_eff):
    p = wl_b/(2*n_eff)
    return p

def bragg(n_eff,p):
    wl_B = 2*n_eff*p 
    return wl_B

def getP(wl,n_eff,dn_eff):
    p = period(wl, n_eff)
    wlb_nm = bragg(n_eff+dn_eff,p)
    return p,wlb_nm

def trueDesign(wl,n_eff,dn_eff):
    design_wavelength=wl
    p,wlb_nm=getP(design_wavelength,n_eff,dn_eff)
    off = wlb_nm-design_wavelength
    design_wavelength=wl-off
    p,wlb_nm=getP(design_wavelength,n_eff,dn_eff)
    return p,wlb_nm

def fbg_spectrum_shifts_dB(wls, wlb_nm, l_mm, n_pi, kappa):

    wlb = wlb_nm*1E-9
    n_eff = 1.4682
    dn_eff = wlb*kappa/np.pi
    p,WLB_NM = trueDesign(wlb,n_eff,dn_eff)
    ph = np.pi
    l = l_mm*1E-3
    transmissiondB_f = []
    if n_pi == 0:
        shifts = np.array([1])*l
    else:
        edgeShift=lambda x: np.cumsum(np.array([1/2,*np.ones(x+1-2),1/2])*1/x)
        shifts = edgeShift(n_pi)*l
    for wl in wls:
        T=transfer_matrix(wl, shifts[0], p, n_eff, dn_eff)
        lengths=np.diff(shifts)
        if type(ph)==float:
            ph=np.ones_like(shifts)*ph
        elif len(ph)!=len(shifts):
            raise ValueError("If individual phase shifts are specified they must match then number of phase shifts")
        for i,l in enumerate(lengths):
            T_l = transfer_matrix(wl, l, p, n_eff, dn_eff)
            P = np.round(np.array([[np.exp(-1j*np.pi/2),0],[0, np.exp(1j*np.pi/2)]]))
            T = np.matmul(T,P)
            T = np.matmul(T,T_l)
        trans_f =  np.abs(1/T[0,0])**2
        transdB_f = 10*np.log10(trans_f/1)
        transmissiondB_f.append(transdB_f)
    return transmissiondB_f

# Functions for double labelling x-axis
def pms2ghz(pms,cwl_m):
    c = 299792458
    #cwl_m = 1550*1E-9 # Quick Fix
    fs = c/(pms*1E-12 + cwl_m)
    ghz = (fs-c/cwl_m)*1E-9
    return ghz

def ghz2pms(ghz,cwl_m):
    c = 299792458
    #cwl_m = 1550*1E-9 # Quick Fix
    fs = ghz*1E9+c/cwl_m
    lams = c/fs
    pms = (lams-cwl_m)*1E12
    return pms

def p2g(pms):
    return pms2ghz(pms,cwl_m)

def g2p(ghz):
    return ghz2pms(ghz,cwl_m)

# Just converting λ (in m) to f (in GHz)
def xs2ghz(xs,cwl_m):
    c = 299792458
    fs = c/xs
    ghz = (fs-c/cwl_m)*1E-9
    return ghz

fbgdB = fbg_spectrum_shifts_dB(xs,cwl,l_mm,n_pi,kappa)
pms = (xs-cwl_m)*1E12
ghz = xs2ghz(xs,cwl_m)

fig, ax = plt.subplots()

#if xchoice == "pm":
    #ax.plot(pms,fbgdB)
    #ax.set_xlabel('Wavelength Detuning [pm]')
#elif xchoice == 'GHz':
    #ax.plot(ghz,fbgdB)
    #ax.set_xlabel('Frequency Detuning [GHz]')

def p2g(pms):
    return pms2ghz(pms,cwl_m)

def g2p(ghz):
    return ghz2pms(ghz,cwl_m)

ax.plot(pms,fbgdB)
ax.set_xlabel('Wavelength Detuning [pm]')
ax.set_ylabel('Transmission [dB]')
ax.set_title(f'{n_pi}πFBG spectrum')
secax = ax.secondary_xaxis('top', functions=(p2g,g2p))
secax.set_xlabel('Frequency Detuning [GHz]')
plt.show()

#plotting part
#st.line_chart(fbg)
st.pyplot(fig)