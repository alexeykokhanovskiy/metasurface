import matlab.engine
import os
import matplotlib.pyplot as plt
import numpy as np

# path to the folder with solver functions
path = './V10_2025'


eng = matlab.engine.start_matlab()
fpath = eng.genpath(path)
eng.addpath(fpath, nargout=0)
# output1, ouput2, output3 = eng.run(f"{path}cmaes_exp_fit",1.7, 4.0, 0.090, nargout=3)
output2, output3 = eng.r_t_calc_multi_param(1.7, 4.0, 50, 0.12546719897087202, 1.1286971151594472, 0.9425861937065012, 0.22694812020681404, nargout=2)
plt.plot(np.array(output2).flatten())
plt.show()
