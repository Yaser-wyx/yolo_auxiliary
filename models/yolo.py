# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

import torch

from utils.common_utils import info
import torch.nn.init as init

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None,
                 is_auxiliary=False, superposition=None, P=0.65,
                 net_layers=-1):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        self.with_auxiliary = False
        self.superposition = None
        self.feature_lambda = 0

        self.superposition_start = 0
        self.random_layers = 0  # 随机应用几层
        self.p = P  # mixstyle起作用的概率
        self.is_auxiliary = False
        self.auxiliary_forward_time = 0
        self.progressive = False
        self.progressive_strategy_map = None
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, layers_save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.layers_save = layers_save if is_auxiliary else None
        self.max_layer = max(layers_save)
        self.net_layers = len(self.model) if net_layers == -1 else net_layers  # 能够应用辅助网络与mixstyle的前x层
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')
        self.is_auxiliary = is_auxiliary
        self.superposition = superposition

    def set_exp_parameter(self, random_layers, feature_lambda, superposition_start, progressive, epochs,
                          with_auxiliary=True):
        # 设置辅助网络以及训练策略
        self.with_auxiliary = with_auxiliary
        self.progressive = progressive
        self.random_layers = random_layers
        self.feature_lambda = feature_lambda
        self.superposition_start = superposition_start

        if progressive:
            assert epochs > superposition_start, "Error, auxiliary start later than total epochs"
            self._generate_progressive_strategy(superposition_start, epochs)

    def _generate_progressive_strategy(self, superposition_start, epochs):
        random_layer_list = [i for i in range(1, self.random_layers + 1, 1)]
        feature_lambda_list = [round(i, 2) for i in np.arange(0.05, self.feature_lambda + 0.001, 0.05)]
        assert epochs > superposition_start + max(len(random_layer_list), len(feature_lambda_list))

        self.progressive_strategy_map = {
            "random_layers": {
                superposition_start + i: v for i, v in enumerate(random_layer_list)
            },
            "feature_lambda": {
                superposition_start + i: v for i, v in enumerate(feature_lambda_list)
            },
        }
        LOGGER.info(f"progressive_strategy: {self.progressive_strategy_map}")

    def forward(self, x, augment=False, profile=False, visualize=False, epoch=0, auxiliary_output=None):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        if hasattr(self, "is_auxiliary") and self.is_auxiliary:
            # 如果是辅助网络
            return self._forward_once_auxiliary(x, profile, visualize)
        else:
            return self._forward_once(x, profile=profile, visualize=visualize,
                                      epoch=epoch, auxiliary_output=auxiliary_output)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    @staticmethod
    def apply_direct(another_output, x, _lambda):
        # 使用特征直接叠加的方式
        x = _lambda * another_output + (1 - _lambda) * x
        return x

    @staticmethod
    def apply_mixstyle(another_output, x, _lambda):
        # 使用mixstyle的特征叠加方式
        mu = torch.mean(x, dim=(2, 3), keepdim=True).detach().to(x.device)
        var = torch.var(x, dim=(2, 3), keepdim=True).detach()
        sig = torch.sqrt(var + 1e-6).detach().to(x.device)
        x_normed = (x - mu) / sig  # normalize the x
        # calculate the mean and std of the auxiliary_output
        another_mu = torch.mean(another_output, dim=(2, 3), keepdim=True).detach().to(x.device)
        another_var = torch.var(another_output, dim=(2, 3), keepdim=True).detach()
        another_sig = torch.sqrt(another_var + 1e-6).detach().to(x.device)
        mu_mix = (1 - _lambda) * another_mu + _lambda * mu  # calculate the mean of the mix
        sig_mix = (1 - _lambda) * another_sig + _lambda * sig  # calculate the std of the mix
        x = (x_normed * sig_mix) + mu_mix  # mix the x and auxiliary_output
        return x

    def _forward_once(self, x, profile=False, visualize=False, epoch=0, auxiliary_output=None):
        # 先将输入给辅助网络跑一遍，得到结果
        if self.progressive:
            # 更新参数
            random_layers_map = self.progressive_strategy_map["random_layers"]
            feature_lambda_map = self.progressive_strategy_map["feature_lambda"]
            self.random_layers = random_layers_map.get(epoch, self.random_layers)
            self.feature_lambda = feature_lambda_map.get(epoch, self.feature_lambda)
            # 选择要进行随机叠加的层索引
        random_select = np.random.choice(range(self.net_layers), self.random_layers, replace=False)

        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            satisfy_flag = self.superposition is not None \
                           and random.random() > self.p \
                           and m.i in random_select \
                           and epoch >= self.superposition_start \
                           and self.training
            before_auxiliary = x.clone() if satisfy_flag else None

            if satisfy_flag:
                # 如果满足进使用特征图叠加的条件
                auxiliary_flag = self.with_auxiliary and \
                                 auxiliary_output is not None

                if auxiliary_flag:
                    # 如果使用辅助网络
                    assert m.i in auxiliary_output.keys()
                    another_output = auxiliary_output[m.i]
                else:
                    # 否则就直接使用自身进行叠加
                    another_output = x.clone()

                # shuffle the batch
                shuffle_idx = torch.randperm(another_output.size(0))
                another_output = another_output[shuffle_idx]
                feature_lambda = self.feature_lambda / self.random_layers
                _lambda = torch.distributions.Beta(feature_lambda, feature_lambda).sample((x.shape[0], 1, 1, 1)).to(
                    x.device)

                if self.superposition == "MixStyle":
                    x = self.apply_mixstyle(another_output, x, _lambda)
                elif self.superposition == "Direct":
                    x = self.apply_direct(another_output, x, _lambda)

            y.append(x if m.i in self.save else None)  # save output
            if self.training:
                if visualize and satisfy_flag:
                    # 可视化显示应用叠加的特征图
                    differ = x - before_auxiliary
                    feature_visualization(differ.cpu().detach(), m.type, m.i, save_dir=Path("run/feature_show"),
                                          name=f"epoch_{epoch}_differ")
                    feature_visualization(before_auxiliary.cpu().detach(), m.type, m.i,
                                          save_dir=Path("run/feature_show"),
                                          name=f"epoch_{epoch}_before")
                    feature_visualization(x.cpu().detach(), m.type, m.i, save_dir=Path("run/feature_show"),
                                          name=f"epoch_{epoch}_after")
            else:
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

        return x

    def _forward_once_auxiliary(self, x, profile=False, visualize=False):
        # 运行辅助网络

        y, dt, layers_output = [], [], {}  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # 当前是辅助网络，保存需要的输出
            if m.i in self.layers_save:
                layers_output[m.i] = x

        return layers_output

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    # 在生成网络的同时，记录哪些是backbone
    added_layers = []
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        add_flag = True
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Detect]:
            add_flag = False

        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        if add_flag:
            added_layers.append(i)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), added_layers


RANDOM_INIT_FUNC_MAP = {
    0: init.normal_,
    1: init.xavier_normal_,  # >=2
    2: init.kaiming_normal_,  # >=2
    3: init.orthogonal_,  # >=2

    # 2: init.xavier_uniform_,  # >=2
    # 3: init.xavier_normal_,  # >=2
    # 4: init.kaiming_uniform_,  # >=2
    # 5: init.orthogonal_,  # >=2
    # 6: init.kaiming_normal_,  # >=2

}


def random_apply(data):
    dim = data.dim()
    if dim < 2:
        random_value = 0
    else:
        random_value = random.randint(0, 3)

    init_func = RANDOM_INIT_FUNC_MAP[random_value]
    init_func(data)


def random_weights_init(m):
    if type(m) in {nn.Conv2d, nn.Linear}:
        random_apply(m.weight.data)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):

        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
