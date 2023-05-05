from modules import *


class YOLO(nn.Module):
    def __init__(self, network, reg_max, chs, strides, cls):
        super().__init__()

        # BackBone
        self.p1 = eval(network[0])
        self.p2 = eval(network[1])
        self.c2f_1 = eval(network[2])
        self.p3 = eval(network[3])
        self.c2f_2 = eval(network[4])
        self.p4 = eval(network[5])
        self.c2f_3 = eval(network[6])
        self.p5 = eval(network[7])
        self.c2f_4 = eval(network[8])
        self.sppf = eval(network[9])

        # Neck
        self.c2f_5 = eval(network[10])
        self.c2f_6 = eval(network[11])
        self.p6 = eval(network[12])
        self.c2f_7 = eval(network[13])
        self.p7 = eval(network[14])
        self.c2f_8 = eval(network[15])
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # Head
        ch1, ch2 = max(16, reg_max * 4, chs[0] // 4), max(len(cls), chs[0])

        self.conv1 = nn.ModuleList(
            nn.Sequential(Conv(ch, ch1, 3, 1), Conv(ch1, ch1, 3, 1), nn.Conv2d(ch1, reg_max * 4, 1)) for ch in chs)
        self.conv2 = nn.ModuleList(
            nn.Sequential(Conv(ch, ch2, 3, 1), Conv(ch2, ch2, 3, 1), nn.Conv2d(ch2, len(cls), 1)) for ch in chs)
        self.anchor = Anchor(cls, reg_max, strides)

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

        return self.anchor(y)


class YOLOBI(nn.Module):
    def __init__(self, network, reg_max, chs, strides, cls):
        super().__init__()

        # BackBone
        self.p1 = eval(network[0])
        self.p2 = eval(network[1])
        self.c3f_1 = eval(network[2])
        self.p3 = eval(network[3])
        self.c3f_2 = eval(network[4])
        self.p4 = eval(network[5])
        self.c3f_3 = eval(network[6])
        self.p5 = eval(network[7])
        self.c3f_4 = eval(network[8])
        self.p6 = eval(network[9])
        self.c3f_5 = eval(network[10])
        self.sppf = eval(network[11])

        # Neck
        self.p7 = eval(network[12])
        self.c3f_6 = eval(network[13])
        self.p8 = eval(network[14])
        self.c3f_7 = eval(network[15])
        self.p9 = eval(network[16])
        self.c3f_8 = eval(network[17])
        self.p10 = eval(network[18])
        self.c3f_9 = eval(network[19])
        self.p11 = eval(network[20])
        self.c3f_10 = eval(network[21])
        self.p12 = eval(network[22])
        self.c3f_11 = eval(network[23])

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        # Head
        ch1, ch2 = max(16, reg_max * 4, chs[0] // 4), max(len(cls), chs[0])

        self.conv1 = nn.ModuleList(
            nn.Sequential(Conv(ch, ch1, 3, 1), Conv(ch1, ch1, 3, 1), nn.Conv2d(ch1, reg_max * 4, 1)) for ch in chs)
        self.conv2 = nn.ModuleList(
            nn.Sequential(Conv(ch, ch2, 3, 1), Conv(ch2, ch2, 3, 1), nn.Conv2d(ch2, len(cls), 1)) for ch in chs)
        self.anchor = Anchor(cls, reg_max, strides)

    def forward(self, x):
        # BackBone
        y1 = self.p1(x)
        y2 = self.p2(y1)
        y3 = self.c3f_1(y2)
        y4 = self.p3(y3)
        y5 = self.c3f_2(y4)
        y6 = self.p4(y5)
        y7 = self.c3f_3(y6)
        y8 = self.p5(y7)
        y9 = self.c3f_4(y8)
        y10 = self.p6(y9)
        y11 = self.c3f_5(y10)
        y12 = self.sppf(y11)

        # Neck
        # FPN UP
        y13 = self.p7(y12)
        y14 = self.c3f_6(torch.cat([self.upsample1(y13), y9], 1))

        y15 = self.p8(y14)
        y16 = self.c3f_7(torch.cat([self.upsample2(y15), y7], 1))

        y17 = self.p9(y16)
        y18 = self.c3f_8(torch.cat([self.upsample3(y17), y5], 1))

        # FPN DOWN
        y19 = self.c3f_9(torch.cat([self.p10(y18), y17, y7], 1))

        y20 = self.c3f_10(torch.cat([self.p11(y19), y15, y9], 1))

        y21 = self.c3f_11(torch.cat([self.p12(y20), y13], 1))

        # Head
        y = [y18, y19, y20, y21]
        y = [torch.cat((self.conv1[i](y[i]), self.conv2[i](y[i])), 1) for i in range(len(y))]

        return self.anchor(y)
