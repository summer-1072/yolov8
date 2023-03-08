# def build_scheduler(self, optimizer, epochs, one_cycle=False, lrf=0.01, start_epoch=0):
#     if one_cycle:
#         lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
#     else:
#         lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear
#
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#     scheduler.last_epoch = start_epoch - 1
#
#     return scheduler
#
#
# # Warmup
# ni = i + nb * epoch
# if ni <= nw:
#     xi = [0, nw]  # x interp
#     self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
#     for j, x in enumerate(self.optimizer.param_groups):
#         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
#         x['lr'] = np.interp(
#             ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
#         if 'momentum' in x:
#             x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

import math
import numpy as np
import matplotlib.pyplot as plt

epochs = 100

x = [i for i in range(epochs)]
y = [((1 - math.cos(i * math.pi / epochs)) / 2) * (0.01 - 1) + 1 for i in range(epochs)]
# y = [(1 - i / epochs) * (1.0 - 0.01) + 0.01 for i in range(epochs)]

plt.figure()
plt.plot(x, y)
plt.show()
