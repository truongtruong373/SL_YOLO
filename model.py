import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    """Tự tính padding để giữ kích thước feature map khi stride=1."""
    if  d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """
    Conv2d + BatchNorm2d + Activation
    Tên module con là conv, bn, act.
    """
    default_act = nn.SiLU(inplace=False)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        if act is True:
            self.act = nn.SiLU(inplace=True)
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    """Depthwise Conv."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=c1, d=d, act=act)

class Bottleneck(nn.Module):
    """
    Bottleneck gồm:
    cv1: Conv
    cv2: Conv
    Có shortcut nếu c1 == c2
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1, c_, k=3, s=1)
        self.cv2 = Conv(c_, c2, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3k(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c1, c_, k=1, s=1)
        self.cv3 = Conv(2*c_, c2, k=1, s=1)

        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut=shortcut, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3k2(nn.Module):
    """
    C3k2 block.
    Forward gần với C2f: cv1 -> split 2 phần -> qua các block trong m -> concat -> cv2
    """
    def __init__(self, c1, c2, n=1, c3k=False, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1)

        self.m = nn.ModuleList(
            [
                C3k(self.c, self.c, n=2, shortcut=shortcut) if c3k else Bottleneck(self.c, self.c, shortcut=shortcut, e=0.5) for _ in range(n)
            ]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast.
    cv1 -> maxpool 3 lần -> concat -> cv2
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_ * 4, c2, k=1, s=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))

class Attention(nn.Module):
    """
    Attention block trong C2PSA.
    """
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = self.head_dim // 2
        self.scale = self.key_dim ** -0.5

        qkv_channels = dim + 2 * num_heads * self.key_dim
        self.qkv = Conv(dim, qkv_channels, k=1, s=1, act=False)
        self.proj = Conv(dim, dim, k=1, s=1, act=False)
        self.pe = Conv(dim, dim, k=3, s=1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x)
        qkv = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (v@attn.transpose(-2, -1)).reshape(B, C, H, W)
        out = out + self.pe(v.reshape(B, C, H, W))
        return self.proj(out)

class PSABlock(nn.Module):
    """
    PSA bao gồm attn và ffn
    """
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c, num_heads=(max(c // 64, 1)))
        self.ffn = nn.Sequential(
            Conv(c, c * 2, k=1, s=1),
            Conv(c * 2, c, k=1, s=1, act=False)
        )

    def forward(self,x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class C2PSA(nn.Module):
    """
    cv1: 256 -> 256
    split thành 128 + 128
    nhánh sau qua PSABlock
    Concat lại rồi qua cv2.
    """
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = Conv(2 * self.c, c2, k=1, s=1)
        self.m = nn.Sequential(*[PSABlock(self.c) for _ in range(n)])

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a,b), dim=1))

class Concat(nn.Module):
    """Concat theo channel"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, xs):
        return torch.cat(xs, dim=self.d)

class DFL(nn.Module):
    """
    Distribution Focal Loss Layer.
    Dùng trong Detect khi decode bbox.
    Ở đây chủ yếu để khớp kiến trúc và state_dict.
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, stride=1, bias=False)
        x = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.conv.weight.data[:] = x
        self.conv.requires_grad_(False)
        self.c1 = c1

    def forward(self, x):
        # x shape: [B, 4 * reg_max, N]
        b, c, n = x.shape
        return self.conv(x.view(b, 4, self.c1, n).transpose(2, 1).softmax(1)).view(b, 4, n)

class Detect(nn.Module):
    """
    Detect head dùng cho training.

    Mỗi feature level trả về một tuple ``(box_logits, class_logits)``
    để khớp với đầu vào của ``YOLODetectionLoss``:

    [
        ([B, 4 * reg_max, 80, 80], [B, nc, 80, 80]),
        ([B, 4 * reg_max, 40, 40], [B, nc, 40, 40]),
        ([B, 4 * reg_max, 20, 20], [B, nc, 20, 20]),
    ]
    """
    def __init__(self, nc=80, ch=(64, 128, 256), reg_max=16):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.no = nc + reg_max * 4

        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(ch[0], 64, k=3, s=1),
                    Conv(64, 64, k=3, s=1),
                    nn.Conv2d(64, 4 * reg_max, kernel_size=1, stride=1)
                ),
                nn.Sequential(
                    Conv(ch[1], 64, k=3, s=1),
                    Conv(64, 64, k=3, s=1),
                    nn.Conv2d(64, 4 * reg_max, kernel_size=1, stride=1)
                ),
                nn.Sequential(
                    Conv(ch[2], 64, k=3, s=1),
                    Conv(64, 64, k=3, s=1),
                    nn.Conv2d(64, 4 * reg_max, kernel_size=1, stride=1)
                ),
            ]
        )

        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Sequential(
                        DWConv(ch[0], ch[0], k=3, s=1),
                        Conv(ch[0], 80, k=1, s=1),
                    ),
                    nn.Sequential(
                        DWConv(80, 80, k=3, s=1),
                        Conv(80, 80, k=1, s=1),
                    ),
                    nn.Conv2d(80, nc, kernel_size=1, stride=1)
                ),
                nn.Sequential(
                    nn.Sequential(
                        DWConv(ch[1], ch[1], k=3, s=1),
                        Conv(ch[1], 80, k=1, s=1),
                    ),
                    nn.Sequential(
                        DWConv(80, 80, k=3, s=1),
                        Conv(80, 80, k=1, s=1),
                    ),
                    nn.Conv2d(80, nc, kernel_size=1, stride=1)
                ),
                nn.Sequential(
                    nn.Sequential(
                        DWConv(ch[2], ch[2], k=3, s=1),
                        Conv(ch[2], 80, k=1, s=1),
                    ),
                    nn.Sequential(
                        DWConv(80, 80, k=3, s=1),
                        Conv(80, 80, k=1, s=1),
                    ),
                    nn.Conv2d(80, nc, kernel_size=1, stride=1)
                ),
            ]
        )
        self.dfl = DFL(reg_max)

    def forward(self, x):
        # x gồm 3 feature maps: P3, P4, P5
        outputs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            outputs.append((box, cls))
        return outputs

class MyYOLODetectionModel(nn.Module):
    def __init__(self, nc=80):
        super().__init__()

        self.model = nn.Sequential(
            # 0, 1
            Conv(3, 16, k=3, s=2),
            Conv(16, 32, k=3, s=2),

            # 2
            C3k2(32, 64, n=1, c3k=False, e=0.25),

            # 3, 4
            Conv(64, 64, k=3, s=2),
            C3k2(64, 128, n=1, c3k=False, e=0.25),

            # 5, 6
            Conv(128, 128, k=3, s=2),
            C3k2(128, 128, n=1, c3k=True),

            # 7, 8, 9, 10
            Conv(128, 256, k=3, s=2),
            C3k2(256, 256, n=1, c3k=True),
            SPPF(256, 256, k=5),
            C2PSA(256, 256, n=1),

            # 11, 12, 13
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            C3k2(384, 128, n=1, c3k=False),

            # 14, 15, 16
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            C3k2(256, 64, n=1, c3k=False),

            # 17, 18, 19
            Conv(64, 64, k=3, s=2),
            Concat(),
            C3k2(192, 128, n=1, c3k=False),

            # 20, 21, 22
            Conv(128, 128, k=3, s=2),
            Concat(),
            C3k2(384, 256, n=1, c3k=True),

            # 23
            Detect(nc=nc, ch=(64, 128, 256), reg_max=16)
        )

    def forward(self, x):
        y = []

        # Backbone
        x0 = self.model[0](x)  # [B, 16, 320, 320]
        x1 = self.model[1](x0)  # [B, 32, 160, 160]
        x2 = self.model[2](x1)  # [B, 64, 160, 160]
        x3 = self.model[3](x2)  # [B, 64, 80, 80]
        x4 = self.model[4](x3)  # [B, 128, 80, 80]   P3 backbone
        x5 = self.model[5](x4)  # [B, 128, 40, 40]
        x6 = self.model[6](x5)  # [B, 128, 40, 40]   P4 backbone
        x7 = self.model[7](x6)  # [B, 256, 20, 20]
        x8 = self.model[8](x7)  # [B, 256, 20, 20]
        x9 = self.model[9](x8)  # [B, 256, 20, 20]
        x10 = self.model[10](x9)  # [B, 256, 20, 20]  P5 backbone

        # Neck top-down
        x11 = self.model[11](x10)  # [B, 256, 40, 40]
        x12 = self.model[12]([x11, x6])  # [B, 384, 40, 40]
        x13 = self.model[13](x12)  # [B, 128, 40, 40]

        x14 = self.model[14](x13)  # [B, 128, 80, 80]
        x15 = self.model[15]([x14, x4])  # [B, 256, 80, 80]
        x16 = self.model[16](x15)  # [B, 64, 80, 80]   P3 detect

        # Neck bottom-up
        x17 = self.model[17](x16)  # [B, 64, 40, 40]
        x18 = self.model[18]([x17, x13])  # [B, 192, 40, 40]
        x19 = self.model[19](x18)  # [B, 128, 40, 40]  P4 detect

        x20 = self.model[20](x19)  # [B, 128, 20, 20]
        x21 = self.model[21]([x20, x10])  # [B, 384, 20, 20]
        x22 = self.model[22](x21)  # [B, 256, 20, 20]  P5 detect

        out = self.model[23]([x16, x19, x22])
        return out
