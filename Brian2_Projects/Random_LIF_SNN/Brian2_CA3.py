from brian2 import *
# Parameters
area = 20000*umetre**2
Cm = 1*ufarad*cm**-2 * area
gl = 5e-5*siemens*cm**-2 * area
El = -65*mV
EK = -90*mV
ENa = 50*mV
g_na = 100*msiemens*cm**-2 * area
g_kd = 30*msiemens*cm**-2 * area
VT = -63*mV
# The model
eqs_HH = '''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
I : amp
dge/dt = -ge/(5*ms)                : volt
dgi/dt = -gi/(10*ms)               : volt
'''
P = NeuronGroup(1000, eqs_HH,
                    threshold='v > -40*mV',
                    refractory='v > -40*mV',
                    method='exponential_euler')
# eqs = '''
# dv/dt  = gl * (El-v) + gNa * m**3 * h * (ENa-v) + gK * n**4 * (EK-v) : volt
# dge/dt = -ge/(5*ms)                : volt
# dgi/dt = -gi/(10*ms)               : volt
# '''
# P = NeuronGroup(4000, eqs, threshold='v>-50*mV', reset='v=-60*mV')
P.v = -65*mV

Pe = P[:200]
Pi = P[200:]
Pe.I = 0.1*nA
Pi.I = 0.2*nA
# start_scope()
Ce = Synapses(Pe, P, on_pre='ge+=0.02*mV')
Ce.connect(p=0.02)
Ci = Synapses(Pi, P, on_pre='gi-=0.09*mV')
Ci.connect(p=0.02)
M = SpikeMonitor(P)
M2 = StateMonitor(P, ('v', 'I'), record=[0, 10, 100])
run(2*second)
plot(M2.t_, M2.v[0]/mV, 'black', label='v')
plt.hold
plot(M.t_, M.i, '.')
show()