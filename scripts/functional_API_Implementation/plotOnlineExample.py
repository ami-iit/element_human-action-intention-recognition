import numpy as np
import matplotlib.pyplot as plt
import datetime

now = datetime.datetime.now()
year, month, day, hour, minute, second = now.year, now.month, now.day, now.hour, now.minute, now.second
time_now = '_{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minute, second)
models_path = 'models' + time_now
print(models_path)

# plt.ion()
# plt.show()
# p = []
# for i in range(10):
#     p+=[i]
#     plt.plot(p)
#     plt.pause(0.5)
#
