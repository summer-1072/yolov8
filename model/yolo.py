from modules import *


class YOLO(nn.Module):
    def __init__(self, param, reg_max, chs, strides, num_cls, training):
        super().__init__()

        # BackBone
        self.p1 = eval(param[0])
        self.p2 = eval(param[1])
        self.c2f_1 = eval(param[2])
        self.p3 = eval(param[3])
        self.c2f_2 = eval(param[4])
        self.p4 = eval(param[5])
        self.c2f_3 = eval(param[6])
        self.p5 = eval(param[7])
        self.c2f_4 = eval(param[8])
        self.sppf = eval(param[9])

        # Neck
        self.c2f_5 = eval(param[10])
        self.c2f_6 = eval(param[11])
        self.p6 = eval(param[12])
        self.c2f_7 = eval(param[13])
        self.p7 = eval(param[14])
        self.c2f_8 = eval(param[15])
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # Head
        ch1, ch2 = max(16, reg_max * 4, chs[0] // 4), max(num_cls, chs[0])

        self.conv1 = nn.ModuleList(
            nn.Sequential(Conv(ch, ch1, 3, 1), Conv(ch1, ch1, 3, 1), nn.Conv2d(ch1, reg_max * 4, 1)) for ch in chs)
        self.conv2 = nn.ModuleList(
            nn.Sequential(Conv(ch, ch2, 3, 1), Conv(ch2, ch2, 3, 1), nn.Conv2d(ch2, num_cls, 1)) for ch in chs)
        self.anchor = Anchor(num_cls, reg_max, strides)

        self.training = training

    def forward(self, x):
        # BackBone
        y1 = self.p1(x)
        y2 = self.p2(y1)
        y3 = self.c2f_1(y2)
        y4 = self.p3(y3)
        y5 = self.c2f_2(y4)
        y6 = self.p4(y5)
        y7 = self.c2f_3(y6)
        y8 = self.p5(y7)
        y9 = self.c2f_4(y8)
        y10 = self.sppf(y9)

        # Neck
        # FPN UP
        y11 = torch.cat([self.upsample1(y10), y7], 1)
        y12 = self.c2f_5(y11)
        y13 = torch.cat([self.upsample2(y12), y5], 1)

        # FPN DOWN
        y14 = self.c2f_6(y13)
        y15 = torch.cat([self.p6(y14), y12], 1)
        y16 = self.c2f_7(y15)
        y17 = torch.cat([self.p7(y16), y10], 1)
        y18 = self.c2f_8(y17)

        # Head
        y = [y14, y16, y18]
        y = [torch.cat((self.conv1[i](y[i]), self.conv2[i](y[i])), 1) for i in range(len(y))]

        if self.training:
            return y
        else:
            return self.anchor(y)
