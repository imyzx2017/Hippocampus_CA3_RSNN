import math
import matplotlib.pyplot as plt
import numpy as np

class CA3_Pyramidal_neuron:
    def __init__(self):
        # self.Steady_Inject_I = 0.02
        # self.V0 = -65
        # self.g_Leak = 1.0/60000
        # self.g_Na = 2.0
        # self.g_CaT = 1e-3  # [-1.1e-3, 2e-3]
        # self.g_KDR = 0.08
        # self.g_A = 0.0001
        # # self.g_M = 0.00002
        # # self.g_AHP = 1.8e-6
        # self.g_KC = 0.0004
        # self.g_CaL = 0.0025
        # self.g_CaN = 0.0025
        # self.Ca_Ionic_extra = 2       # initial extracellular Ca2+ ionic Concentration [nM]
        # self.Ca_Inoic_intracell = 50  # initial intracellular Ca2+ ionic Concentration [nM]
        self.cai_init = 5e-5
        self.cao = 2
        self.E_K = -91
        self.E_Na = 50
        self.v0 = -65

        self.dt = 0.01
        self.t = [0]  # record updating time bar

        #
        # self.m0 = 0.05  # 05 # 0.2
        # self.h0 = 0.6  # 6 # 0.3
        # self.n0 = 0.32  # 32 # 0.8
        self.V = []
        # constants for calculating
        self.F = 96520
        self.R = 8.3134
        self.PI = math.pi
        self.celsius = 30
        self.Temp = self.celsius + 273.14
        # membrane properties
        self.Rm = 60000
        self.cm = 1e-3
        self.Ra = 2e-2

        # define ionic conductance(S/cm^2)
        self.g_na11 = 2.0
        self.g_catbar = 0.45
        self.g_cat31 = 0.45
        self.g_cat32 = 0.45
        self.g_cat33 = 0.45
        self.g_catbar_C4565 = 1.6*0.45  # mutation of CaT channel
        self.g_canbar = 0.0025
        self.g_calbar = 0.0025
        self.g_kdr = 0.08
        self.g_ka = 0.0001
        self.g_km = 0.00002
        self.g_kc = 0.00005
        self.g_kahp = 0.0000018
        self.g_pas = 1/self.Rm

        # define ionic concentration & currents
        self.cai = [5.0e-5]
        self.ina = []
        self.ina11 = []
        self.icat, self.icat_alpha1G, self.icat_alpha1H, self.icat_alpha1I = [], [], [], []
        self.iC4565 = []
        self.ican, self.ical, self.ica = [0], [0], [0]
        self.ikdr, self.ika, self.ikm, self.ikc, self.ikahp, self.ipas = [], [], [], [], [], []
        self.ion, self.current = [], []
        self.Ist = 0

        ##################################
        #
        # gating variables
        #
        ##################################
        # Na1.1 gates
        self.Na_MC3, self.Na_MC, self.Na_MO, self.Na_MI = [], [], [], []
        self.Na_MIs, self.Na_MC2, self.IC3, self.IC2 = [], [], [], []
        self.Na_MIs2, self.M_C3, self.M_C2, self.M_C, self.M_O = [], [], [], [], []
        self.M_OP, self.Na_OP, self.Na_O, self.OPs, self.M_IF = [], [], [], [], []
        self.ina11_wt = 0.0
        self.ina11_mt = 0.0

        # Ca N and L type channel gates
        self.mn, self.hn, self.ml = [], [], []
        # CaT gates
        self.nth, self.lth, self.nti, self.lti = [], [], [], []
        self.ntg_1G, self.ltg_1G, self.ntg_1H, self.ltg_1H, self.ntg_1I, self.ltg_1I = [], [], [], [], [], []

        # CaT mutant gates
        self.ntC4565, self.ltC4565 = [], []
        
        # K gates
        self.nkdr, self.lkdr, self.nka, self.lka, self.mkm, self.okc, self.wkahp = [], [], [], [], [], [], []

        # global variables
        self.seg = 1    # segment
        # geometry parameters
        self.area, self.diam, self.dx, self.g = 0, 0, 0, 0

    def ca_dyn(self):
        try:
            self.cai.append(self.cai[-1] + 0.1*self.dt*(-0.15*self.ica[-1] + (-self.cai[-1])/5))
        except:
            raise Exception('cai or ica with no initialization')

    def geo(self):
        # define dimension
        self.r3, self.L3 = 15, 25.5
        # define segment length, diameter ,area assumg dx=1, and resistance
        self.diam = 2*self.r3
        self.dx = self.L3/1
        self.area = self.PI * self.diam
        self.g = 4 * self.Ra / (self.PI * self.diam * self.diam)
    
    def cable(self):
        self.icat.append(self.icat_alpha1G[-1] + self.icat_alpha1H[-1] + self.icat_alpha1I[-1])
        self.ica.append(self.icat[-1] + self.ican[-1] + self.ical[-1])
        self.ion.append(self.icat[-1] + self.ican[-1] + self.ical[-1] + self.ikdr[-1] + self.ika[-1] + self.ikm[-1] + self.ikc[-1] + self.ikahp[-1] + self.ipas[-1])
        # self.ion.append(
        #     self.ina11[-1] + self.icat[-1] + self.ican[-1] + self.ical[-1] + self.ikdr[-1] + self.ika[-1] + self.ikm[
        #         -1] + self.ikc[-1] + self.ikahp[-1] + self.ipas[-1])
        self.current.append(self.ion[-1])

        # inject stimulation current at soma
        self.current[-1] = self.current[-1] + self.Ist
        self.V.append(self.V[-1] - self.dt*self.current[-1]/self.cm)   # Euler Equation

    ######################################
    #
    #   Updating channels
    #
    ######################################
    
    # Human T-type calcium channels based on: <<Chemin et al., J Physiol., 2002, 540, 3-14.
    # CaT-alpha 1G CaV3.1
    def cat_human_alpha1G(self):
        gbar = 0.008
        vhalfn = -49.3
        vhalfl = -74.2
        kn = 4.6
        kl = -5.5
        q10 = 2.3
        qt = math.pow(q10, (self.celsius - 28)/10)
        pcabar = 0.0001   # cm/s
        z = 2

        # initialization  t=[0, dt]
        if self.t[-1] < self.dt:
            vv = - 65
            self.n_inf_cat1g = 1/(1 + math.exp(-(vv - vhalfn)/kn))
            self.l_inf_cat1g = 1/(1 + math.exp(-(vv - vhalfl)/kl))

            if vv>-56:  # useless in this step
                self.tau_n_1G = (0.8 + 0.025 * math.exp(-vv / 14.5))/qt
            else:
                self.tau_n_1G = (1.71 * math.exp((vv + 120) / 38)) / qt
            self.tau_l_1G = (12.3 + 0.12 * math.exp(- vv / 10.8))/qt
            self.ntg_1G.append(self.n_inf_cat1g)
            self.ltg_1G.append(self.l_inf_cat1g)

        self.n_inf_cat1g = 1 / (1 + math.exp(-(self.V[-1] - vhalfn) / kn))
        self.l_inf_cat1g = 1 / (1 + math.exp(-(self.V[-1] - vhalfl) / kl))
        if self.V[-1] > -56:
            self.tau_n_1G = (0.8 + 0.025 * math.exp(-self.V[-1] / 14.5)) / qt
        else:
            self.tau_n_1G = (1.71 * math.exp((self.V[-1] + 120) / 38)) / qt
        if self.V[-1] > -60:
            self.tau_l_1G = (12.3 + 0.12 * math.exp(-self.V[-1] / 10.8)) / qt
        else:
            tau_l = 137/qt
        self.ntg_1G.append((self.ntg_1G[-1] + self.dt * self.n_inf_cat1g / self.tau_n_1G) / (1 + self.dt / self.tau_n_1G))
        self.ltg_1G.append((self.ltg_1G[-1] + self.dt * self.l_inf_cat1g / self.tau_l_1G) / (1 + self.dt / self.tau_l_1G))

        w = self.V[-1]*0.001*z*self.F/(self.R*(self.celsius + 273.16))
        ghk = -0.001*z*self.F*(self.cao - self.cai[-1]*math.exp(w))*w/(math.exp(w) - 1)
        self.icat_alpha1G.append(self.g_cat31*pcabar*self.ntg_1G[-1]*self.ntg_1G[-1]*self.ltg_1G[-1]*ghk)

    def cat_human_alpha1H(self):
        gbar = 0.008
        vhalfn = -48.4
        vhalfl = -75.6
        kn = 5.2
        kl = -6.2
        q10 = 2.3
        qt = math.pow(q10, (self.celsius - 28)/10.0)
        pcabar = 0.0001
        z = 2

        # initialization  t=[0, dt]
        if self.t[-1] < self.dt:
            vv = -65
            self.n_inf_cat1H = 1 / (1 + math.exp(-(vv - vhalfn) / kn))
            self.l_inf_cat1H = 1 / (1 + math.exp(-(vv - vhalfl) / kl))
            if vv>-56:  # useless in this step
                self.tau_n_1H = (1.34 + 0.035 * math.exp(-vv / 11.8))/qt
            else:
                self.tau_n_1H = (2.44 * math.exp((vv + 120) / 40)) / qt
            self.tau_l_1H = (18.3 + 0.005 * math.exp(- vv / 6.2))/qt
            self.ntg_1H.append(self.n_inf_cat1H)
            self.ltg_1H.append(self.l_inf_cat1H)

        self.n_inf_cat1H = 1 / (1 + math.exp(-(self.V[-1] - vhalfn) / kn))
        self.l_inf_cat1H = 1 / (1 + math.exp(-(self.V[-1] - vhalfl) / kl))
        if self.V[-1] > -56:
            self.tau_n_1H = (1.34 + 0.035 * math.exp(-self.V[-1] / 11.8)) / qt
        else:
            self.tau_n_1H = (2.44 * math.exp((self.V[-1] + 120) / 40)) / qt
        if self.V[-1] > -60:
            self.tau_l_1H = (18.3 + 0.005 * math.exp(-self.V[-1] / 6.2)) / qt
        else:
            self.tau_l_1H = 500 / qt
        self.ntg_1H.append((self.ntg_1H[-1] + self.dt * self.n_inf_cat1H / self.tau_n_1H) / (1 + self.dt / self.tau_n_1H))
        self.ltg_1H.append((self.ltg_1H[-1] + self.dt * self.l_inf_cat1H / self.tau_l_1H) / (1 + self.dt / self.tau_l_1H))

        w = self.V[-1] * 0.001 * z * self.F / (self.R * (self.celsius + 273.16))
        ghk = -0.001 * z * self.F * (self.cao - self.cai[-1] * math.exp(w)) * w / (math.exp(w) - 1)
        self.icat_alpha1H.append(self.g_cat32 * pcabar * self.ntg_1H[-1] * self.ntg_1H[-1] * self.ltg_1H[-1] * ghk)

    def cat_human_alpha1I(self):
        gbar = 0.008
        vhalfn = -41.5
        vhalfl = -69.8
        kn = 6.2
        kl = -6.1
        q10 = 2.3
        qt = math.pow(q10, (self.celsius - 28)/10.0)
        pcabar = 0.0001
        z = 2

        # initialization  t=[0, dt]
        if self.t[-1] < self.dt:
            vv = -65
            self.n_inf_cat1I = 1 / (1 + math.exp(-(vv - vhalfn) / kn))
            self.l_inf_cat1I = 1 / (1 + math.exp(-(vv - vhalfl) / kl))
            if vv>-60:  # useless in this step
                self.tau_n_1I = (7.2 + 0.02 * math.exp(-vv / 14.7))/qt
            else:
                self.tau_n_1I = (0.875 * math.exp((vv + 120) / 41)) / qt
            if vv > -70:  # useless in this step
                self.tau_l_1I = (79.5 + 2.0 * math.exp(-vv / 9.3)) / qt
            else:
                self.tau_l_1I = 260 / qt
            self.ntg_1I.append(self.n_inf_cat1I)
            self.ltg_1I.append(self.l_inf_cat1I)

        self.n_inf_cat1I = 1 / (1 + math.exp(-(self.V[-1] - vhalfn) / kn))
        self.l_inf_cat1I = 1 / (1 + math.exp(-(self.V[-1] - vhalfl) / kl))
        if self.V[-1] > -60:
            self.tau_n_1I = (7.2 + 0.02 * math.exp(-self.V[-1] / 14.7)) / qt
        else:
            self.tau_n_1I = (0.875 * math.exp((self.V[-1] + 120) / 41)) / qt
        if self.V[-1] > -60:
            self.tau_l_1I = (79.5 + 2.0 * math.exp(-self.V[-1] / 9.3)) / qt
        else:
            self.tau_l_1I = 260 / qt
        self.ntg_1I.append((self.ntg_1I[-1] + self.dt * self.n_inf_cat1I / self.tau_n_1I) / (1 + self.dt / self.tau_n_1I))
        self.ltg_1I.append((self.ltg_1I[-1] + self.dt * self.l_inf_cat1I / self.tau_l_1I) / (1 + self.dt / self.tau_l_1I))

        w = self.V[-1] * 0.001 * z * self.F / (self.R * (self.celsius + 273.16))
        ghk = -0.001 * z * self.F * (self.cao - self.cai[-1] * math.exp(w)) * w / (math.exp(w) - 1)
        self.icat_alpha1I.append(self.g_cat32 * pcabar * self.ntg_1I[-1] * self.ntg_1I[-1] * self.ltg_1I[-1] * ghk)

    def CaN_channel(self):
        if self.t[-1] < self.dt:
            if len(self.V)==0:
                self.V.append(-65)
            self.am_CaN = 0.1967 * (-self.V[-1] + 19.88) / (math.exp((-self.V[-1] + 19.88)/10.0) - 1)
            self.bm_CaN = 0.046 * math.exp(-self.V[-1] / 20.73)
            self.ah_CaN = 1.6e-4 * math.exp(-self.V[-1] / 48.4)
            self.bh_CaN = 1 / (math.exp((-self.V[-1] + 39) / 10.0) + 1)
            self.tau_m_CaN = 1 / (self.am_CaN + self.bm_CaN)
            self.tau_h_CaN = 1 / (self.ah_CaN + self.bh_CaN)
            self.m_inf_CaN = self.am_CaN * self.tau_m_CaN
            self.h_inf_CaN = self.ah_CaN * self.tau_h_CaN
            self.mn.append(self.m_inf_CaN)
            self.hn.append(self.h_inf_CaN)
            if len(self.ican) == 0:
                self.ican.append(0.0)
        self.am_CaN = 0.1967 * (-self.V[-1] + 19.88) / (math.exp((-self.V[-1] + 19.88) / 10.0) - 1)
        self.bm_CaN = 0.046 * math.exp(-self.V[-1] / 20.73)
        self.ah_CaN = 1.6e-4 * math.exp(-self.V[-1] / 48.4)
        self.bh_CaN = 1 / (math.exp((-self.V[-1] + 39) / 10.0) + 1)
        self.tau_m_CaN = 1 / (self.am_CaN + self.bm_CaN)
        self.tau_h_CaN = 1 / (self.ah_CaN + self.bh_CaN)
        self.m_inf_CaN = self.am_CaN * self.tau_m_CaN
        self.h_inf_CaN = self.ah_CaN * self.tau_h_CaN
        self.mn.append((self.mn[-1] + self.dt * self.m_inf_CaN / self.tau_m_CaN) / (1 + self.dt / self.tau_m_CaN))
        self.hn.append((self.hn[-1] + self.dt * self.h_inf_CaN / self.tau_h_CaN) / (1 + self.dt / self.tau_h_CaN))

        ki = 0.001
        KTF = 25 / 293.15 * (self.celsius + 273.15)
        f = KTF / 2
        nu = self.V[-1] / f

        if(np.fabs(nu)<1e-4):
            efun = 1 - nu/2
        else:
            efun = nu / (math.exp(nu) - 1)

        ghk = -f*(1 - self.cai[-1]/self.cao * math.exp(nu)) * efun
        gcan = self.g_canbar * self.mn[-1] * self.mn[-1] * self.hn[-1] * ki / (ki + self.cai[-1])
        self.ican.append(gcan * ghk)

    def CaL_channel(self):
        if self.t[-1] < self.dt:
            if len(self.V)==0:
                self.V.append(-65)
            self.am_CaL = 15.69 * (-self.V[-1] + 81.5) / (math.exp((-self.V[-1] + 81.5)/10.0) - 1)
            self.bm_CaL = 0.29 * math.exp(-self.V[-1] / 10.86)
            self.tau_m_CaL = 1 / (self.am_CaL + self.bm_CaL)
            self.m_inf_CaL = self.am_CaL * self.tau_m_CaL
            self.ml.append(self.m_inf_CaL)
            if len(self.ican) == 0:
                self.ican.append(0.0)
        self.am_CaL = 15.69 * (-self.V[-1] + 81.5) / (math.exp((-self.V[-1] + 81.5) / 10.0) - 1)
        self.bm_CaL = 0.29 * math.exp(-self.V[-1] / 10.86)
        self.tau_m_CaL = 1 / (self.am_CaL + self.bm_CaL)
        self.m_inf_CaL = self.am_CaL * self.tau_m_CaL

        self.ml.append((self.ml[-1] + self.dt * self.m_inf_CaL / self.tau_m_CaL) / (1 + self.dt / self.tau_m_CaL))

        ki = 0.001
        gcal = self.g_canbar * self.ml[-1] * self.ml[-1] * (ki / (ki + self.cai[-1]))
        KTF = 25 / 293.15 * (self.celsius + 273.15)
        f = KTF / 2
        nu = self.V[-1] / f

        if(np.fabs(nu)<1e-4):
            efun = 1 - nu/2
        else:
            efun = nu / (math.exp(nu) - 1)

        ghk = -f*(1 - self.cai[-1]/self.cao * math.exp(nu)) * efun
        self.ical.append(gcal * ghk)

    def Kdr_channel(self):
        if self.t[-1] < self.dt:
            if len(self.V)==0:
                self.V.append(-65)
            self.an_Kdr = 0.03 * math.exp(1e-3 * 5 * 0.4 * (self.V[-1] + 32) * self.F / (self.R + self.Temp))
            self.bn_Kdr = 0.03 * math.exp(1e-3 * 5 * 0.6 * (self.V[-1] + 32) * self.F / (self.R + self.Temp))
            self.al_Kdr = 0.001 * math.exp(1e-3 * 2 * (self.V[-1] + 61) * self.F / (self.R + self.Temp))
            self.bl_Kdr = 0.001
            self.tau_n_Kdr = 1 / (self.an_Kdr + self.bn_Kdr)
            self.tau_l_Kdr = 1 / (self.al_Kdr + self.bl_Kdr)
            self.n_inf_Kdr = self.an_Kdr * self.tau_n_Kdr
            self.l_inf_Kdr = self.al_Kdr * self.tau_l_Kdr
            self.nkdr.append(self.n_inf_Kdr)
            self.lkdr.append(self.l_inf_Kdr)
            self.ikdr.append(0.0)

        self.an_Kdr = 0.03 * math.exp(1e-3 * 5 * 0.4 * (self.V[-1] + 32) * self.F / (self.R + self.Temp))
        self.bn_Kdr = 0.03 * math.exp(1e-3 * 5 * 0.6 * (self.V[-1] + 32) * self.F / (self.R + self.Temp))
        self.al_Kdr = 0.001 * math.exp(1e-3 * 2 * (self.V[-1] + 61) * self.F / (self.R + self.Temp))
        self.bl_Kdr = 0.001
        self.tau_n_Kdr = 1 / (self.an_Kdr + self.bn_Kdr)
        self.tau_l_Kdr = 1 / (self.al_Kdr + self.bl_Kdr)
        self.n_inf_Kdr = self.an_Kdr * self.tau_n_Kdr
        self.l_inf_Kdr = self.al_Kdr * self.tau_l_Kdr

        self.nkdr.append((self.nkdr[-1] + self.dt * self.n_inf_Kdr / self.tau_n_Kdr) / (1 + self.dt/self.tau_n_Kdr))
        self.lkdr.append((self.lkdr[-1] + self.dt * self.l_inf_Kdr / self.tau_l_Kdr) / (1 + self.dt / self.tau_l_Kdr))
        self.ikdr.append(self.g_kdr * (self.nkdr[-1]**3) * self.lkdr[-1] * (self.V[-1] + 91))

    def Ka_Channel(self):
        if self.t[-1] < self.dt:
            if len(self.V)==0:
                self.V.append(-65)
            self.an_Ka = 0.02 * math.exp(1e-3 * 3 * 0.6 * (self.V[-1] + 33.6) * self.F / (self.R + self.Temp))
            self.bn_Ka = 0.02 * math.exp(1e-3 * 5 * 0.6 * (self.V[-1] + 33.6) * self.F / (self.R + self.Temp))
            self.al_Ka = 0.08 * math.exp(1e-3 * 4 * (self.V[-1] + 83) * self.F / (self.R + self.Temp))
            self.bl_Ka = 0.08
            self.tau_n_Ka = 1 / (self.an_Ka + self.bn_Ka)
            self.tau_l_Ka = 1 / (self.al_Ka + self.bl_Ka)
            self.n_inf_Ka = self.an_Ka * self.tau_n_Ka
            self.l_inf_Ka = self.al_Ka * self.tau_l_Ka
            self.nka.append(self.n_inf_Ka)
            self.lka.append(self.l_inf_Ka)
            self.ika.append(0.0)

        self.an_Ka = 0.02 * math.exp(1e-3 * 3 * 0.6 * (self.V[-1] + 33.6) * self.F / (self.R + self.Temp))
        self.bn_Ka = 0.02 * math.exp(1e-3 * 5 * 0.6 * (self.V[-1] + 33.6) * self.F / (self.R + self.Temp))
        self.al_Ka = 0.08 * math.exp(1e-3 * 4 * (self.V[-1] + 83) * self.F / (self.R + self.Temp))
        self.bl_Ka = 0.08
        self.tau_n_Ka = 1 / (self.an_Ka + self.bn_Ka)
        self.tau_l_Ka = 1 / (self.al_Ka + self.bl_Ka)
        self.n_inf_Ka = self.an_Ka * self.tau_n_Ka
        self.l_inf_Ka = self.al_Ka * self.tau_l_Ka

        self.nka.append((self.nka[-1] + self.dt * self.n_inf_Ka / self.tau_n_Ka) / (1 + self.dt / self.tau_n_Ka))
        self.lka.append((self.lka[-1] + self.dt * self.l_inf_Ka / self.tau_l_Ka) / (1 + self.dt / self.tau_l_Ka))

        self.ika.append(self.g_ka * self.nka[-1] * self.lka[-1] * (self.V[-1] + 91.0))

    def Km_Channel(self):
        q10 = math.pow(5.0, ((self.celsius - 23) / 10))
        if self.t[-1] < self.dt:
            if len(self.V)==0:
                self.V.append(-65)
            self.am_Km = q10 * 0.006 * math.exp(1e-3 * 10 * 0.06 * (self.V[-1] + 55) * self.F / (self.R + self.Temp))
            self.bm_Km = q10 * 0.006 * math.exp(-1e-3 * 9.4 * (self.V[-1] + 55) * self.F / (self.R + self.Temp))
            self.tau_m_Km = 1 / (self.am_Km + self.bm_Km)
            self.m_inf_Km = self.am_Km * self.tau_m_Km
            self.mkm.append(self.m_inf_Km)
            self.ikm.append(0.0)
        self.am_Km = q10 * 0.006 * math.exp(1e-3 * 10 * 0.06 * (self.V[-1] + 55) * self.F / (self.R + self.Temp))
        self.bm_Km = q10 * 0.006 * math.exp(-1e-3 * 9.4 * (self.V[-1] + 55) * self.F / (self.R + self.Temp))
        self.tau_m_Km = 1 / (self.am_Km + self.bm_Km)
        self.m_inf_Km = self.am_Km * self.tau_m_Km

        self.mkm.append((self.mkm[-1] + self.dt * self.m_inf_Km / self.tau_m_Km) / (1 + self.dt / self.tau_m_Km))
        self.ikm.append(self.g_km * self.mkm[-1] * (self.V[-1] + 91))

    def Kc_Channel(self):
        k1 = 0.48e-3
        k2 = 0.13e-6
        F1 = 96.4853

        if self.t[-1] < self.dt:
            if len(self.V)==0:
                self.V.append(-65)
            self.ao_Kc = self.cai[-1] * 0.28 / (self.cai[-1] + k1 * math.exp(-2 * 0.84 * self.V[-1] * F1 / (self.R * self.Temp)))
            self.bo_Kc = 0.48 / (1 + self.cai[-1] / k2 * math.exp(-2 * 1.0 * self.V[-1] * F1 / (self.R * self.Temp)))
            self.tau_o_Kc = 1 / (self.ao_Kc + self.bo_Kc)
            self.o_inf_Kc = self.ao_Kc * self.tau_o_Kc
            self.okc.append(self.o_inf_Kc)
            self.ikc.append(0.0)
        self.ao_Kc = self.cai[-1] * 0.28 / (
        self.cai[-1] + k1 * math.exp(-2 * 0.84 * self.V[-1] * F1 / (self.R * self.Temp)))
        self.bo_Kc = 0.48 / (1 + self.cai[-1] / k2 * math.exp(-2 * 1.0 * self.V[-1] * F1 / (self.R * self.Temp)))
        self.tau_o_Kc = 1 / (self.ao_Kc + self.bo_Kc)
        self.o_inf_Kc = self.ao_Kc * self.tau_o_Kc

        self.okc.append((self.okc[-1] + self.dt * self.o_inf_Kc / self.tau_o_Kc) / (1 + self.dt / self.tau_o_Kc))
        self.ikc.append(self.g_kc * self.okc[-1] * (self.V[-1] + 91))

    def Kahp_new_Channel(self):
        if self.t[-1] < self.dt:
            # self.cai
            self.aw_Kahp = 1e31
            self.bw_Kahp = 0.2
            self.tau0_Kahp = 5.0
            self.tau_w_Kahp = 1 / (self.aw_Kahp * math.pow(self.cai[-1], 2) + self.bw_Kahp) + self.tau0_Kahp
            self.w_inf_Kahp = self.aw_Kahp * math.pow(self.cai[-1], 2) / (self.aw_Kahp * (math.pow(self.cai[-1], 2)) + self.bw_Kahp)
            self.wkahp.append(self.w_inf_Kahp)
            self.ikahp.append(0.0)
        self.aw_Kahp = 1e31
        self.bw_Kahp = 0.2
        self.tau_w_Kahp = 1 / (self.aw_Kahp * math.pow(self.cai[-1], 2) + self.bw_Kahp) + self.tau0_Kahp
        self.w_inf_Kahp = self.aw_Kahp * math.pow(self.cai[-1], 2) / (
        self.aw_Kahp * (math.pow(self.cai[-1], 2)) + self.bw_Kahp)

        self.wkahp.append((self.wkahp[-1] + self.dt * self.w_inf_Kahp / self.tau_w_Kahp) / (1 + self.dt / self.tau_w_Kahp))
        self.ikahp.append(self.g_kahp * math.pow(self.wkahp[-1], 8) * (self.V[-1] + 91))

    def Leak_Channel(self):
        self.ipas.append(self.g_pas * (self.V[-1] - self.v0))


if __name__=="__main__":
    tstop = 200
    dt = 0.01

    CA3_HH = CA3_Pyramidal_neuron()
    CA3_HH.V.append(-65)
    CA3_HH.geo()
    for i in range(20000):
        I = -0.03 # -0.2 * 100 / (CA3_HH.PI * CA3_HH.diam * CA3_HH.dx)
        # print(I)
        CA3_HH.Ist = I
        CA3_HH.cat_human_alpha1G()
        CA3_HH.cat_human_alpha1H()
        CA3_HH.cat_human_alpha1I()
        CA3_HH.CaN_channel()
        CA3_HH.CaL_channel()
        CA3_HH.Kdr_channel()
        CA3_HH.Ka_Channel()
        CA3_HH.Km_Channel()
        CA3_HH.Kc_Channel()
        CA3_HH.Kahp_new_Channel()
        CA3_HH.Leak_Channel()
        CA3_HH.ca_dyn()
        CA3_HH.cable()
        CA3_HH.t.append(i * dt)
        # print(CA3_HH.ican)

    plt.plot(CA3_HH.t, CA3_HH.V)
    plt.hold
    plt.plot(CA3_HH.t[1:], CA3_HH.ntg_1G[2:])
    plt.show()