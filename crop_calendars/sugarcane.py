#%% Irvine, Cane Sugar Handbook, 10th ed., 1977, P. 16.

import numpy as np

water_frac = (0.73 + 0.76)/2
solids_frac = (0.10 + 0.16)/2
solids_frac_of_dry = solids_frac / (1 - water_frac)
# Sugar types: sucrose, glucose, fructose
sugars_frac_of_solids = np.array([(0.70 + 0.88), (0.02 + 0.04), (0.02 + 0.04)])/2

# What fraction of the total is sugar?
total_sugar_frac_of_solids = np.sum(sugars_frac_of_solids)
total_sugar_frac_of_total = total_sugar_frac_of_solids * solids_frac
total_sugar_frac_of_dry = total_sugar_frac_of_solids * solids_frac_of_dry
print(f'{total_sugar_frac_of_total*100}% of millable cane is sugar (mostly sucrose, some glucose and fructose).')
print(f'That\'s {np.round(total_sugar_frac_of_dry*100, 2)}% of the dry portion of millable cane.')

def get_c_frac(c, h, o):
   c = np.array(c)
   h = np.array(h)
   o = np.array(o)
   c_wt = 12.011
   h_wt = 1.00784
   o_wt = 15.9994
   return c_wt*c / (c_wt*c + h_wt*h + o_wt*o)

sugars_cfrac = get_c_frac([12, 6, 6], [22, 12, 12], [11, 6, 6])

mean_sugar_cfrac = np.sum(sugars_cfrac * sugars_frac_of_solids / np.sum(sugars_frac_of_solids))
print(f'Sugar in cane is {int(np.round(100*mean_sugar_cfrac))}% C.')

