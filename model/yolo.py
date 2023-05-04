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
        self.sppf = eval(network[9])

        # Neck
        self.p6 = eval(network[10])
        self.c3f_5 = eval(network[11])
        self.p7 = eval(network[12])
        self.c3f_6 = eval(network[13])
        self.p8 = eval(network[14])
        self.c3f_7 = eval(network[15])
        self.p9 = eval(network[16])
        self.c3f_8 = eval(network[17])

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
        y3 = self.c3f_1(y2)
        y4 = self.p3(y3)
        y5 = self.c3f_2(y4)
        y6 = self.p4(y5)
        y7 = self.c3f_3(y6)
        y8 = self.p5(y7)
        y9 = self.c3f_4(y8)
        y10 = self.sppf(y9)

        # Neck
        # FPN UP
        y11 = self.p6(y10)
        y12 = torch.cat([self.upsample1(y11), y7], 1)
        y13 = self.c3f_5(y12)
        y14 = self.p7(y13)
        y15 = torch.cat([self.upsample2(y14), y5], 1)
        y16 = self.c3f_6(y15)

        # FPN DOWN
        y17 = torch.cat([self.p8(y16), y14], 1)
        y18 = self.c3f_7(y17)
        y19 = torch.cat([self.p9(y18), y11], 1)
        y20 = self.c3f_8(y19)

        # Head
        y = [y16, y18, y20]
        y = [torch.cat((self.conv1[i](y[i]), self.conv2[i](y[i])), 1) for i in range(len(y))]

        return self.anchor(y)
