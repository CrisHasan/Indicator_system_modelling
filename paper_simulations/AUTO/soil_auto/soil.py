
source "$HOME/auto-07p/cmds/auto.env.sh"
cd /Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/AUTO/soil_auto
auto

import numpy as np

s0 = run(e='soil',c='soil', NBC = 4, ICP =  ['T','delta', 'S1'], UZSTOP = {'T': 50.0})
first = run(s0('UZ1'), NPR = 1200, DS=-0.001, DSMAX = 0.1, ICP = ['delta', 'S0', 'S1'], UZSTOP =  {'delta': 0.0})
tau_upper_range = np.arange(0.3, 1.0, 0.01)
tau_upper = [None] * len(tau_upper_range)
for i, tau in enumerate(tau_upper_range):
    if tau == 0.3:
        shift = first
    else:
        shift = run(first('UZ1'), DS=0.001, ICP = ['tau', 'S0', 'S1'], UZSTOP = {'tau': tau})
    
    tau_l = run(shift('UZ1'), ICP = ['F', 'S0', 'S1'], ITMX= 40, UZSTOP = {'S1': 0.999})
    tau_r = run(shift('UZ1'), DS='-', ICP = ['F', 'S0', 'S1'], UZSTOP =  {'S1': 0.999})
    if np.isclose(tau, 0.99): # For some reason, reducing DS avoids the MX point.
        tau_l = run(shift('UZ1'), ICP = ['F', 'S0', 'S1'], DS=0.005, ITMX= 40, UZSTOP = {'S1': 0.999})
    
    tau_upper[i] = rl(merge(tau_l + tau_r))
    str_int = 'tau_curve_' + str(int(tau*100))
    save(tau_upper[i], str_int)

########
tau = 0.999
shift = run(first('UZ1'), DS=0.001, ICP = ['tau', 'S0', 'S1'], UZSTOP = {'tau': tau})
tau_l = run(shift('UZ1'), ICP = ['F', 'S0', 'S1'], DS=0.005, ITMX= 40, UZSTOP = {'S1': 0.999})
tau_r = run(shift('UZ1'), DS='-', ICP = ['F', 'S0', 'S1'], UZSTOP =  {'S1': 0.999})
tau_upper_999 = rl(merge(tau_l + tau_r))
save(tau_upper_999, 'tau_curve_999')

pl(tau_upper[0] + tau_upper[10] + tau_upper[20] + tau_upper[30] + tau_upper[40] + tau_upper[50] + tau_upper[60] + tau_upper[-1])
########   
tau_lower_range = np.arange(0.02, 0.3, 0.01)
tau_lower_range = tau_lower_range[::-1]
tau_lower = [None] * len(tau_lower_range)
shift_lower = rl(run(first('UZ1'), DS=-0.001, ICP = ['tau', 'S0', 'S1'], UZSTOP = {'tau': 0.02}, UZR = {'tau': [i for i in tau_lower_range[:-1]]}))
for i, tau in enumerate(tau_lower_range):
    uz_label = 'UZ' + str(i+1)
    tau_l = run(shift_lower(uz_label), ICP = ['F', 'S0', 'S1'], UZSTOP =  {'S1': 0.999})
    tau_r = run(shift_lower(uz_label), DS='-', ICP = ['F', 'S0', 'S1'], UZSTOP =  {'S1': 0.999})
    tau_lower[i] = rl(merge(tau_l + tau_r))
    str_int = 'tau_curve_' + str(int(tau*100))
    save(tau_lower[i], str_int)

pl(tau_lower[-1])
cl()
############################################################
# Get individual curves for individual S0 values
# Unfortunately, always runs into an 'MX point' error
S_0 = 0.3
shift_3 = run(first('UZ1'), DS=-0.001, ICP = ['F', 'S0', 'S1'], UZSTOP = {'S0': S_0})
s0_3 = run(shift_3('UZ1'), ICP = ['F', 'tau', 'S1'], DS=-0.005, ITMX= 40)

S_0 = 0.4
shift_4 = run(first('UZ1'), DS=-0.001, ICP = ['F', 'S0', 'S1'], UZSTOP = {'S0': S_0})
s0_4 = run(shift_4('UZ1'), ICP = ['F', 'tau', 'S1'], DS=-0.005, ITMX= 40)

S_0 = 0.5
shift_5 = run(first('UZ1'), DS=-0.001, ICP = ['F', 'S0', 'S1'], UZSTOP = {'S0': S_0})
s0_5 = run(shift_5('UZ1'), ICP = ['F', 'tau', 'S1'], DS=-0.005, ITMX= 40)

pl(s0_3 + s0_4 + s0_5)
############################################################
nearby_0 = run(s0('UZ1'), NTST = 3000, NPR = 1200, DS=-0.001, DSMAX = 0.1, ICP = ['delta', 'S0', 'S1'], UZSTOP =  {'delta': 0.0})
nearby_l = run(nearby_0('UZ1'), NPR = 1, NMX= 10, DS = -0.01, DSMAX = 0.1, ICP = ['S0', 'S1', 'delta'])
nearby_r = run(nearby_0('UZ1'), NPR = 1, NMX= 10, DS = 0.01, DSMAX = 0.1, ICP = ['S0', 'S1', 'delta'])
nearby_all = rl(merge(nearby_l + nearby_r))
pl(nearby_all)
save(nearby_all, 'nearby_all')
############################################################
extended_crit_r = run(first('UZ1'), NTST=1600, NPR =100, NMX = 6000, DS = 0.001, DSMAX = 0.1, ICP = ['T','S0', 'S1'], UZSTOP = {'T': 160.0})
extended_crit_l = run(first('UZ1'), NTST=1600, NPR =100, NMX = 6000, DS = -0.001, DSMAX = 0.1, ICP = ['T','S0', 'S1'], UZSTOP = {'T': 20.0})
extended_crit_all = rl(merge(extended_crit_l + extended_crit_r))
pl(extended_crit_all)
save(extended_crit_all, 'extended_crit_all')

############################################################
# Using integral condition to find the critical point
integer_values = np.arange(1, 50)
i0 = run(e='soil',c='soil', NBC = 4, NINT = 1, ICP =  ['T','delta', 'drift', 'S1'], UZSTOP = {'T': 10.0})
crit_0 = run(i0('UZ1'), NPR = 10,NMX=100,DSMAX = 0.1, ICP = ['drift', 'S0', 'S1','delta'], UZR =  {'drift': 0.0})
crit_1 = run(crit_0('UZ1'), NTST=8000, NPR = 4000, NMX = 4000, ICP = ['T', 'S0', 'S1','delta'], UZR = {'T': integer_values}, UZSTOP = {'T': 50.0})


# Define array of integer values from 1 to 49
integer_values = np.arange(1, 50)
del_0 = run(i0('UZ1'), DS=0.001, NPR = 10,NMX=100,DSMAX = 0.1, ICP = ['drift', 'S0', 'S1','delta'], UZR =  {'delta': 0.0})
del_1 = run(del_0('UZ1'), NTST=8000, NPR = 4000, NMX = 4000, ICP = ['T', 'S0', 'S1','drift'], UZR = {'T': integer_values}, UZSTOP =  {'T': 50.0})
