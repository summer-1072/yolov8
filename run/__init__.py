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

epochs = 50
warmup_epochs = 5

x = [i for i in range(epochs)]

lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (0.01 - 1) + 1
y1 = [np.interp(i, [0, warmup_epochs], [0, 0.01 * lf(i)]) for i in range(epochs)]
y2 = [np.interp(i, [0, warmup_epochs], [0.02, 0.01 * lf(i)]) for i in range(epochs)]
# y = [((1 - math.cos(i * math.pi / epochs)) / 2) * (0.01 - 1) + 1 for i in range(epochs)]
# y = [(1 - i / epochs) * (1.0 - 0.01) + 0.01 for i in range(epochs)]

plt.figure(figsize=(14, 6))
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.075, right=0.98, hspace=0.25, wspace=0.25)

plt.subplot(1, 2, 1)
plt.title('warm_up + cos_anneal')
plt.xlabel('epoch')
plt.ylabel('lr')
plt.plot(x, y1)

plt.subplot(1, 2, 2)
plt.title('warm_down + cos_anneal')
plt.xlabel('epoch')
plt.ylabel('lr')
plt.plot(x, y2)
plt.show()
