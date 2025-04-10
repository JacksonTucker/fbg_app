import numpy as np
from numpy.lib.scimath import sqrt

# Minimal versions of necessary functions

# Constructs uniform grating transfer matrix
def transfer_matrix(wl, L, p, n_eff, dn_eff):
    wl_b = 2*n_eff*p
    D = 2*np.pi*n_eff*(1/wl - 1/wl_b)
    sig = 2*np.pi/wl*dn_eff
    S = D + sig
    K = np.pi/wl*dn_eff
    G = sqrt(K**2-S**2)
    transfer_m = np.array([[np.cosh(G*L)-1j*S/G*np.sinh(G*L), -1j*K/G*np.sinh(G*L)],[1j*K/G*np.sinh(G*L), np.cosh(G*L)+ 1j*S/G*np.sinh(G*L)]])
    return transfer_m

# Gets grating period
def period(wl_b, n_eff):
    p = wl_b/(2*n_eff)
    return p

# Gets Bragg wavelength
def bragg(n_eff,p):
    wl_B = 2*n_eff*p 
    return wl_B

# Gets period and Bragg wl
def getP(wl,n_eff,dn_eff):
    p = period(wl, n_eff)
    wlb_nm = bragg(n_eff+dn_eff,p)
    return p,wlb_nm

# Gets true p and Bragg wl
def trueDesign(wl,n_eff,dn_eff):
    design_wavelength=wl
    p,wlb_nm=getP(design_wavelength,n_eff,dn_eff)
    off = wlb_nm-design_wavelength
    design_wavelength=wl-off
    p,wlb_nm=getP(design_wavelength,n_eff,dn_eff)
    return p,wlb_nm

# Calculates π phase shifted fbg transmission spectrum in dB
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

# Converta λ (in m) to f (in GHz)
def xs2ghz(xs,cwl_m):
    c = 299792458
    fs = c/xs
    ghz = (fs-c/cwl_m)*1E-9
    return ghz

# Single label function from
#if xchoice == "pm":
    #ax.plot(pms,fbgdB)
    #ax.set_xlabel('Wavelength Detuning [pm]')
#elif xchoice == 'GHz':
    #ax.plot(ghz,fbgdB)
    #ax.set_xlabel('Frequency Detuning [GHz]')

# Calculates fbg characteristics
def fbg_characteristics(wls,fbg,wlb_m,n_pi):
    wlb = wlb_m*1E-9
    # Locate midpoint and a single minima
    min_loc = np.where(fbg == min(fbg))[0][0]
    #print(min_loc)
    wlb_loc = int(np.round(np.median(np.where(np.round(wls*1E9,3) == np.round(wlb*1E9,3))[0])))
    #print(wlb_loc)
    c = 299792458 # m/s
    # Display max suppression point
    #print(f'Max suppression of {np.round(min(fbg),3)} dB @ {abs(np.round((wls[min_loc]-wlb)*1E12,3))} pm detuning')
    min_detune_pm = abs(np.round((wls[min_loc]-wlb)*1E12,3)) # pm
    min_detune_ghz = abs(np.round(((c/wls[min_loc])-(c/wlb))*1E-9,3)) # GHz
    max_suppression = abs(min(fbg))
    # Uniform fbg stats
    if n_pi == 0:
        PBedge = 'Not Applicable'
        PBwig = 'Not Applicable'
        PBwidth_pm = 50
        PBwidth_ghz = 5
    # Pi shift fbg stats
    else:
        # Locate the minima side pass band edge
        if wlb_loc > min_loc:
            #PBedge = np.argsort(np.diff(fbg[min_loc:wlb_loc]))[-1]+1
            try:
                PBedge = np.where(np.diff(fbg[min_loc:wlb_loc])<0)[0][0]+min_loc
                PBwig = np.round(min(fbg[PBedge:wlb_loc]),2)
                PBwidth_pm = 2*abs(np.round((wls[PBedge]-wlb)*1E12,3)) # pm
                PBwidth_ghz = 2*abs(np.round(((c/wls[PBedge])-(c/wlb))*1E-9,3)) # GHz
            except IndexError:
                PBwidth_pm = 'N/A'
                PBwidth_ghz = 'N/A'
                PBwig = 'N/A'
                #print('Central peak is too sharp to determine width and variation')
        else:
            #PBedge = np.argsort(np.diff(fbg[wlb_loc:min_loc]))[-1]+1
            try:
                PBedge = np.where(np.diff(fbg[wlb_loc:min_loc]<0))[0][0]+wlb_loc
                PBwig = np.round(min(fbg[wlb_loc:PBedge]),2)
                PBwidth_pm = np.round(2*abs(np.round((wls[PBedge]-wlb)*1E12,3)),4) # pm
                PBwidth_ghz = np.round(2*abs(np.round(((c/wls[PBedge])-(c/wlb))*1E-9,3)),4) # GHz
            except IndexError:
                #print('Central peak too ssharp to determine width and variation')
                PBwidth_pm = 'N/A'
                PBwidth_ghz = 'N/A'
                PBwig = 'N/A'
    #return wlb_loc, min_loc, PBedge, PBwig, max_suppression, min_detune_pm, min_detune_ghz,  PBwidth_pm, PBwidth_ghz
    return PBwig, max_suppression, min_detune_pm, min_detune_ghz, PBwidth_pm, PBwidth_ghz   