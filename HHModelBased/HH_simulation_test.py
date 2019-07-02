import math
import matplotlib.pyplot as plt
import numpy as np


class HH:
    def __init__(self):
        self.V0 = -65
        self.g_l = 0.3
        self.g_na = 120
        self.g_k = 36
        self.E_l = -54.387
        self.E_na = 50
        self.E_k = -77
        self.dt = 0.01

        self.m0 = 0.2236    #0.05  # 05 # 0.2
        self.h0 = 0         #0.6  # 6 # 0.3
        self.n0 = 0.0582      # 32 # 0.8

        self.V = []

    def PlotHH(self):
        self.a_m = 0.1 * (self.V0 + 40) / (1 - math.exp(-(self.V0 + 40) / 10))
        self.b_m = 4 * math.exp(-(self.V0 + 65) / 18)
        self.a_h = 0.07 * math.exp(-(self.V0 + 65) / 20)
        self.b_h = 1 / (math.exp(-(self.V0 + 35) / 10) + 1)
        self.a_n = 0.01 * (self.V0 + 55) / (1 - math.exp(-(self.V0 + 55) / 10))
        self.b_n = 0.125 * math.exp(-(self.V0 + 65) / 80)

        self.m0 = self.m0 + self.dt * (self.a_m * (1 - self.m0) - self.b_m * self.m0)
        self.h0 = self.h0 + self.dt * (self.a_h * (1 - self.h0) - self.b_h * self.h0)
        self.n0 = self.n0 + self.dt * (self.a_n * (1 - self.n0) - self.b_n * self.n0)

        # self.m0 = self.m0 + self.dt * (self.a_m - (self.a_m + self.b_m) * self.m0)
        # self.h0 = self.h0 + self.dt * (self.a_h - (self.a_h + self.b_h) * self.h0)
        # self.n0 = self.n0 + self.dt * (self.a_n - (self.a_n + self.b_n) * self.n0)

        self.V_temp = self.g_l * (self.E_l - self.V0) + self.g_na * np.power(self.m0, 3) * self.h0 * (
                    self.E_na - self.V0) + self.g_k * np.power(self.n0, 4) * (self.E_k - self.V0) + 10
        self.V0 = self.V0 + self.dt * self.V_temp
        self.V.append(self.V0)

    # def PlotBursting(self):
    #     self.m_inf = 1 / (1 + exp(-(self.V0 + 30) / 9.5))
    #     self.h_inf = 1 / (1 + exp((self.V0 + 45) / 7))
    #     self.n_inf = 1 / (1 + exp(-(self.V0 + 35) / 10))
    #     self.p_inf = 1 / (1 + exp(-(self.V0 + 45) / 3))
    #     self.a_inf = 1 / (1 + exp(-(self.V0 + 50) / 20))
    #     self.b_inf = 1 / (1 + exp((self.V0 + 80) / 6))
    #     self.z_inf = 1 / (1 + exp(-(self.V0 + 39) / 5))
    #     self.tao_h = 0.1 + 0.75 / (1 + exp(self.V0 + 40.5) / 6)
    #     self.tao_n = 0.1 + 0.5 / (1 + exp((self.V0 + 27) / 15))
    #     self.tao_b = 15
    #     self.tao_z = 75
    #
    #     ###############updating h,n,b,z,V
    #     self.h = self.h + self.dt * (self.h_inf - self.h) / self.tao_h
    #     self.n = self.n + self.dt * (self.n_inf - self.n) / self.tao_n
    #     self.b = self.b + self.dt * (self.b_inf - self.b) / self.tao_b
    #     self.z = self.z + self.dt * (self.z_inf - self.z) / self.tao_z
    #     self.new_V_delta = -self.g_Na * self.m_inf ** 3 * self.h * (self.V0 - self.V_Na) - self.g_K * self.n ** 4 * (
    #     self.V0 -
    #     self.V_K) - self.g_Nap * self.p_inf * (self.V0 - self.V_Na) - self.g_A * self.a_inf ** 3 * self.b * (self.V0 -
    #                                                                                                          self.V_K) - self.g_M * self.z * (
    #     self.V0 - self.V_K) - self.g_L * (self.V0 - self.V_L) + 0




Demo = HH()
# print(Demo.V0)
for i in range(100000):
    Demo.PlotHH()
t = np.linspace(0, 100000*Demo.dt, 100000)
plt.plot(t, Demo.V)
plt.xlabel('t/ms')
plt.ylabel('V/mV')
plt.title('I=10nA, Zedborad FPGA simulation result')
plt.show()