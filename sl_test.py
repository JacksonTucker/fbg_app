import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
import fbgfunctions as func

st.write('Interactive FBG Spectra')

# Input interactions
kappa = st.slider("𝜅 - Modulation intensity (/m)", 100,500,250,1)
l_mm = st.slider("l - Length of grating (mm)", 20,50,35,1)
cwl = st.slider("λ - Central wavelength (nm)", 1500,1600,1550,1)
n_pi = st.radio("n - Number of π phase shifts", [0,1,3,5,7], index=0, horizontal=True)
# Choice of X-axis (removed in favour of double labelling)
#xoptions = ["GHz","pm"]
#xchoice = st.radio("X axis",xoptions, horizontal = True)
if n_pi>0:
    view = st.radio("View", ['Wide','Peak'], horizontal = True)
else:
    view = 'Wide'

# Plotting inputs
cwl_m = cwl*1E-9
span = 200E-12
pts = 1000
xs = np.linspace(-span,span,pts)+cwl_m

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

# Plotting part
fbgdB = func.fbg_spectrum_shifts_dB(xs,cwl,l_mm,n_pi,kappa)
pms = (xs-cwl_m)*1E12
ghz = func.xs2ghz(xs,cwl_m)
PBwig, max_suppression, min_detune_pm, min_detune_ghz, PBwidth_pm, PBwidth_ghz = func.fbg_characteristics(xs,fbgdB,cwl,n_pi)

fig, ax = plt.subplots()
ax.plot(pms,fbgdB)
ax.set_xlabel('Wavelength Detuning [pm]')
ax.set_ylabel('Transmission [dB]')
ax.set_title(f'{n_pi}πFBG spectrum')
if view == 'Peak' and n_pi>0:
    if PBwidth_pm == 'N/A':
        ax.set_xlim([-5,5])
    else:
        ax.set_xlim([-2*PBwidth_pm,2*PBwidth_pm])
        
secax = ax.secondary_xaxis('top', functions=(p2g,g2p))
secax.set_xlabel('Frequency Detuning [GHz]')
plt.show()

st.pyplot(fig)

# Characteristics readouts
st.write(f'Max suppression: {np.round(max_suppression,2)} dB')
st.write(f'Minima location: {np.round(min_detune_pm,4)} pm, {np.round(min_detune_ghz,4)} GHz detuning')
if n_pi > 0:
        if PBwig == 'N/A':
            st.write('**Central peak is too sharp to determine passband width and variation**')
        else:
            st.write(f'Passband width: {PBwidth_pm} pm, {PBwidth_ghz} GHz')
            st.write(f'Passband variation: {PBwig} dB')
