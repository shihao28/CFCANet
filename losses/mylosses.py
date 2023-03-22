import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import models


class SoftMarginTripletLoss(nn.Module):
    """
    Calculate Soft Margin Triplet Loss
    """
    def __init__(self, isBinaryLoss=True):
        super(SoftMarginTripletLoss, self).__init__()
        self._l2_dist = nn.PairwiseDistance()
        self.isBinaryLoss = isBinaryLoss

    # def _l2_dist(self, a):
    #     eps = 1e-5
    #     sq_sum_a = torch.sum(a * a, 1)
    #     dist = sq_sum_a.reshape((-1, 1)) + sq_sum_a.reshape((1, -1)) - 2 * torch.matmul(a, a.T)
    #     # dist[dist < 0] = 0 # for computational stability
    #     return dist + eps

    def forward(self, features, labels):
        """
        :param features: torch.Tensor
            A matrix of shape N x M that contains M-dimensional feature vector of  N objects (normally 128-D)
        :param labels: torch.Tensor
            The one-dimensional array of length N containing the associated id for each object
        -----------
        :return: torch.Tensor
            A scalar triplet loss function value.
        """
        loss = nn.Softplus()        
        almost_inf = 1e+10
        # sqr_dist = self._l2_dist(features)
        # dist_features = torch.sqrt(sqr_dist * (sqr_dist > 0))
        dist_features = torch.cdist(features, features)


        # dist_features = torch.sqrt(self._l2_dist(features))
        labels_mask = (labels.reshape((-1, 1)) == labels.reshape((1, -1))).double()

        if self.isBinaryLoss:
            dist_p, _ = torch.max(labels_mask * dist_features, axis=1)
            dist_n, _ = torch.min(labels_mask * almost_inf + dist_features, axis=1)
            loss = torch.mean(nn.Softplus()(dist_p - dist_n))
        else:
            # https://arxiv.org/pdf/1803.10859.pdf
            weight_p = torch.where(labels_mask.double()==1, dist_features.double(), 0.)
            weight_p = nn.Softmax(dim=-1)(weight_p)
            weight_n = torch.where(labels_mask.double()==0, dist_features.double(), 0.)
            weight_n = nn.Softmax(dim=-1)(-1 * weight_n)
            loss_p = (weight_p * dist_features).sum(dim=-1)
            loss_n = (weight_n * dist_features).sum(dim=-1)
            loss = loss_p - loss_n
            loss = torch.mean(nn.Softplus()(loss))
        return loss


class con_loss(nn.Module):
    # https://arxiv.org/pdf/2103.07976.pdf
    # https://github.com/TACJu/TransFG/blob/master/models/modeling.py
    def __init__(self, use_upper_diag, alpha=0.4):
        super(con_loss, self).__init__()
        self.use_upper_diag = use_upper_diag
        self.alpha = alpha

    def forward(self, features, labels):
        B, _ = features.shape
        features = F.normalize(features)
        cos_matrix = features.mm(features.t())
        if self.use_upper_diag:
            cos_matrix.triu(diagonal=1) # to use only upper triangular matrix
        pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - self.alpha
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        if self.use_upper_diag:
            element_count = torch.range(start=0, end=B-1).sum()
            loss /= element_count
        else:
            loss /= (B * B)
        return loss


class MagnetLoss(nn.Module):
    """
    Calculate Simple Unimodal Magnet loss
    """
    def __init__(self, margin = 1.0):
        super(MagnetLoss, self).__init__()
        self.margin = margin

    def _l2_dist(self, a, b):
        eps = 1e-5
        sq_sum_a = torch.sum(a * a, 1)
        sq_sum_b = torch.sum(b * b, 1)
        dist = sq_sum_a.reshape((-1, 1)) + sq_sum_b.reshape((1, -1)) - 2 * torch.matmul(a, b.T)
        return  dist + eps

    def forward(self, features, labels, unique_labels = None):
        """
        :param features: torch.Tensor
            A matrix of shape N x M that contains M-dimensional feature vector of  N objects (normally 128-D)
        :param labels: torch.Tensor
            The one-dimensional array of length N containing the associated id for each object
        :param unique_labels: torch.Tensor
            Optional tensor of unique values in `labels`. If None given, computed from data.
        :return: torch.Tensor
            A scalar triplet loss function value.
        """
        num_per_class = None
        if unique_labels is None:
            unique_labels, num_per_class = torch.unique_consecutive(labels, return_counts = True)
            num_per_class = num_per_class.double()

        y_mat = (labels.reshape((-1, 1)) == unique_labels.reshape((1, -1))).double()
        # If class_means is None, compute from batch data.
        if num_per_class is None:
            # num_per_class = torch.sum(y_mat.sum, axis = 0)
            num_per_class = torch.sum(y_mat, axis = 0)

        class_means = torch.sum(torch.unsqueeze(y_mat.T, -1) * torch.unsqueeze(features, 0), axis = 1) / torch.unsqueeze(num_per_class, -1)
        
        squared_distance = self._l2_dist(features, class_means.float())

        num_samples = float(labels.shape[0]) 
        variance = torch.sum(y_mat * squared_distance) / (num_samples - 1.0)

        eps = 1e-4
        const = 1. / (-2 * (variance + eps))
        linear = const * squared_distance - y_mat * self.margin

        maxi, _ = torch.max(linear, 1, keepdim = True)
        loss_mat = torch.exp(linear - maxi)

        a = torch.sum(y_mat * loss_mat, axis = 1)
        b = torch.sum((1 - y_mat) * loss_mat, axis = 1)
        loss = -torch.log(eps + a / (eps + b))
        loss *= (loss > 0)

        return torch.mean(loss), class_means, variance


class FocalLoss(nn.Module):
    # Vehicle Re-Identication Model Based on Optimized DenseNet121 with Joint Loss
    def __init__(self, gamma = 0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, outputs, label):
        onehot_label = torch.zeros_like(outputs)
        onehot_label.scatter_(1, label.data.unsqueeze(1), 1)
        outputs = nn.Softmax(dim = 1)(outputs)
        prob = (outputs * onehot_label).sum(dim = 1)
        loss = (1 - prob) ** self.gamma * -1 * torch.log(prob)
        return torch.mean(loss)


class LabelSmoothingLoss(nn.Module):
    # https://arxiv.org/pdf/2011.10951v2.pdf --> This paper quote Label Smoothing and explains its benefits
    # https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmoothingLossWithFocal(nn.Module):
    def __init__(self, gamma = 1):
        super(LabelSmoothingLossWithFocal, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        pred = nn.Softmax(dim=-1)(pred)
        weight = torch.zeros_like(pred)
        weight.fill_(0.1 ** self.gamma)
        # weight = torch.clone(pred ** self.gamma)
        # for i in range(len(target)):
        #     weight[i, target[i]] = (1 - pred[i, target[i]]) ** self.gamma
        weight.scatter_(1, target.data.unsqueeze(1), 0.9 ** self.gamma)
        pred = torch.log(pred)
        loss = torch.mean(torch.sum(-pred * weight, dim=-1))
        return loss


class LabelSmoothingLossWithThres(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLossWithThres, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = nn.Softmax(dim=self.dim)(pred)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist[pred < 0.1] = 0
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * torch.log(pred), dim=self.dim))


class Gradient_CE(nn.Module):
    """
    https://arxiv.org/pdf/1912.06842v1.pdf
    Gradient Boosting Cross Entropy Loss
    """
    def __init__(self, k = 15):
        super(Gradient_CE, self).__init__()
        self.k = k
    
    def forward(self, outputs, label):
        onehot_label = torch.zeros_like(outputs)
        onehot_label.scatter_(1, label.data.unsqueeze(1), 1)
        outputs_k_negative_class_only = torch.clone(outputs)
        outputs_k_negative_class_only.scatter_(1, label.data.unsqueeze(1), 0)
        _, idx = torch.sort(outputs_k_negative_class_only, dim=1)
        idx = idx[:, :-self.k]
        outputs_k_negative_class_only.scatter_(1, idx, 0)
        for i in range(len(label)): outputs_k_negative_class_only[i, label[i]] = outputs[i, label[i]]
        outputs_k_negative_class_only = outputs_k_negative_class_only.log_softmax(dim=-1)
        outputs_k_negative_class_only = torch.where(outputs_k_negative_class_only==0, 1e-10, outputs_k_negative_class_only.double())
        loss = torch.mean(torch.sum(-onehot_label * outputs_k_negative_class_only, dim=-1))
        return loss


class SoftTargetCrossEntropy(nn.Module):
    # https://github.com/rwightman/pytorch-image-models/blob/25e7c8c5e548f2063ffe8d83659dc4eea1d249cd/timm/loss/cross_entropy.py#L29
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class supervisor(nn.Module):
    """
    # https://github.com/PRIS-CV/Mutual-Channel-Loss/blob/master/CUB-200-2011.py
    """
    def __init__(self):
        super(supervisor, self).__init__()

    def Mask(self, nb_batch, channels):
        
        # For class count = 200
        # foo = [1] * 2 + [0] *  1
        # bar = []
        # for i in range(200):
        #     np.random.shuffle(foo)
        #     bar += foo
        # bar = [bar for i in range(nb_batch)]
        # bar = np.array(bar).astype("float32")
        # bar = bar.reshape(nb_batch,200*channels,1,1)
        # bar = torch.from_numpy(bar)
        # bar = bar.cuda()
        # bar = Variable(bar)
        
        # compcars
        cls_split = [107, 431 - 107]
        foo0 = [1] * 2 + [0] * 2
        foo1 = [1] * 2 + [0] *  3
        # end

        # car
        # cls_split = [108, 196 - 108]
        # foo0 = [1] * 5 + [0] * 5
        # foo1 = [1] * 5 + [0] *  6
        # end

        bar = []
        for i in range(cls_split[0]):
            np.random.shuffle(foo0)
            bar += foo0
        
        for i in range(cls_split[1]):
            np.random.shuffle(foo1)
            bar += foo1
        bar = [bar for i in range(nb_batch)]
        bar = np.array(bar).astype("float32")
        bar = bar.reshape(nb_batch,-1,1,1)
        bar = torch.from_numpy(bar)
        bar = bar.cuda()
        bar = Variable(bar)
        return bar, cls_split, foo0, foo1

    def forward(self, x, targets, height, cnum):
        # loss_ce = nn.CrossEntropyLoss()(x, targets)

        mask, cls_split, foo0, foo1 = self.Mask(x.size(0), cnum)
        split_cls = [cls_split[0] * len(foo0)]
        split_cls += [cls_split[1] * len(foo1)]

        branch_2 = x
        branch_2 = branch_2.reshape(branch_2.size(0),branch_2.size(1), branch_2.size(2) * branch_2.size(3))
        branch_2 = F.softmax(branch_2, 2)
        branch_2 = branch_2.reshape(branch_2.size(0), branch_2.size(1), x.size(2), x.size(2))
        # branch = F.max_pool2d(branch.transpose(3,1), kernel_size=(1,cnum), stride=(1,cnum)).transpose(3, 1).contiguous()
        branch_2_1, branch_2_2 = branch_2.split(split_cls, dim=1)
        branch_2_1 = F.max_pool2d(branch_2_1.transpose(3, 1), kernel_size=(1,len(foo0)), stride=(1,len(foo0))).transpose(3, 1).contiguous()
        branch_2_2 = F.max_pool2d(branch_2_2.transpose(3, 1), kernel_size=(1,len(foo1)), stride=(1,len(foo1))).transpose(3, 1).contiguous()
        branch_2 = torch.cat([branch_2_1, branch_2_2], 1)
        branch_2 = branch_2.reshape(branch_2.size(0),branch_2.size(1), branch_2.size(2) * branch_2.size(3))
        # loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch_2,2))/cnum # set margin = 3.0
        loss_2 = torch.sum(branch_2,2).mean(1).mean()

        branch_1 = x * mask 

        split_cls = [cls_split[0] * len(foo0)]
        split_cls += [cls_split[1] * len(foo1)]
        branch_1_1, branch_1_2 = branch_1.split(split_cls, dim=1)
        # branch_1 = F.max_pool2d(branch_1.transpose(3, 1), kernel_size=(1,cnum), stride=(1,cnum)).transpose(3, 1).contiguous()
        branch_1_1 = F.max_pool2d(branch_1_1.transpose(3, 1), kernel_size=(1,len(foo0)), stride=(1,len(foo0))).transpose(3, 1).contiguous()
        branch_1_2 = F.max_pool2d(branch_1_2.transpose(3, 1), kernel_size=(1,len(foo1)), stride=(1,len(foo1))).transpose(3, 1).contiguous()
        branch_1 = torch.cat([branch_1_1, branch_1_2], 1)
        branch_1 = nn.AvgPool2d(kernel_size=(height,height))(branch_1)
        branch_1 = branch_1.view(branch_1.size(0), -1)

        loss_1 = nn.CrossEntropyLoss()(branch_1, targets)
    
        return loss_1 - 10*loss_2


class PerceptualLoss(nn.Module):
    def __init__(self, model, device):
        super(PerceptualLoss, self).__init__()
        if model == 'vgg':
            self.model = models.vgg16_bn(True)
            encoder = list(self.model.features.children())
            self.stage1_encoder = nn.Sequential(*encoder[:7])
            self.stage2_encoder = nn.Sequential(*encoder[7:14])
            self.stage3_encoder = nn.Sequential(*encoder[14:24])
            self.stage4_encoder = nn.Sequential(*encoder[24:34])
            self.stage5_encoder = nn.Sequential(*encoder[34:-1])

        elif model == 'resnet':
            self.model = models.resnet50(True)
            self.stage1_encoder = nn.Sequential(*list(model.children())[:4])
            self.stage2_encoder = self.model.layer1
            self.stage3_encoder = self.model.layer2
            self.stage4_encoder = self.model.layer3
            self.stage5_encoder = self.model.layer4

        self.model.to(device)

    def forward(self, x_recon, x_raw):
        loss = 0
        stages = [
            self.stage1_encoder, self.stage2_encoder, self.stage3_encoder,
            self.stage4_encoder, self.stage5_encoder
            ]
        for stage in stages:
            x_recon = stage(x_recon)
            x_raw = stage(x_raw)
            loss += ((x_recon - x_raw)**2).mean()
        return loss


class CE_HS(nn.Module):
    """
    Cross entropy on Hard Sample
    """
    def __init__(self, label_smooth=0.1, conf_thres=0.5):
        super(CE_HS, self).__init__()
        self.label_smooth = label_smooth
        self.conf_thres = conf_thres
    
    def forward(self, pred, label):
        # Old
        # pred = nn.Softmax(1)(pred)
        # pred_clone = pred.clone()
        # true_dist = torch.zeros_like(pred)
        # idx_dim0, idx_dim1 = torch.where(pred > self.conf_thres)
        # if len(idx_dim0) > 0:
        #     for x, y in zip(idx_dim0, idx_dim1):
        #         true_dist[x,y] = self.label_smooth
        #         pred_clone[x, y] = 1 - pred_clone[x, y]
        # true_dist.scatter_(1, label.data.unsqueeze(1), 1 - self.label_smooth)
        # pred = pred_clone
        # if (true_dist.sum(1) > 1 - self.label_smooth).any():
        #     print('Hard Sample Treatment activated ....................')

        # new
        pred_tmp = nn.Softmax(1)(pred)
        pred_clone = pred.clone()
        true_dist = torch.zeros_like(pred)
        idx_dim0, idx_dim1 = torch.where(pred_tmp > self.conf_thres)
        if len(idx_dim0) > 0:
            for x, y in zip(idx_dim0, idx_dim1):
                true_dist[x, y] = self.label_smooth
                pred_clone[x, y] = 1 - pred_clone[x, y]
        true_dist.scatter_(1, label.data.unsqueeze(1), 1 - self.label_smooth)
        pred = pred_clone

        return torch.mean(torch.sum(-true_dist * torch.log(pred), dim=1))


class CCL(nn.Module):
    """ Coupled Cluster Loss """
    def __init__(self, alpha=0.2):
        super(CCL, self).__init__()
        self.alpha = alpha
    
    def forward(self, preds, labels):
        labels_mask = (labels.reshape((-1, 1)) == labels.reshape((1, -1))).double()
        pos_idx = [torch.where(labels_mask_tmp == 1)[0] for labels_mask_tmp in labels_mask]
        pos_centroid = [preds[pos_idx_tmp].mean(0) for pos_idx_tmp in pos_idx]
        pos_centroid = torch.stack(pos_centroid, 0)
        pos_sample_dist = ((preds - pos_centroid)**2).sum(1)**0.5
        neg_sample_dist = torch.cdist(preds, pos_centroid)
        neg_sample_dist, _ = torch.min(labels_mask * 1e10 + neg_sample_dist, 1)
        loss = (nn.Softplus()(pos_sample_dist - neg_sample_dist + self.alpha)).mean()
        return loss


class MinimaxLoss(nn.Module):
    def __init__(self, num_classes, smoothing):
        super(MinimaxLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits, targets):
        probs = nn.LogSoftmax(-1)(logits)
        onehot_targets = torch.nn.functional.one_hot(targets, self.num_classes)
        target_cls_prob = (probs * onehot_targets).sum(1)
        minimax_part1 = (1 - self.smoothing) * target_cls_prob
        
        onehot_nontargets = 1 - onehot_targets
        nontarget_cls_prob, _ = (probs * onehot_nontargets + onehot_targets * 1e10).min(-1)
        minimax_part2 = self.smoothing * nontarget_cls_prob

        minimax = (minimax_part1 + minimax_part2).mean() * -1

        return minimax


class CenterLoss(nn.Module):
    # https://github.com/raoyongming/CAL/blob/master/fgvc/utils.py
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


class CenterLossv1(nn.Module):
    # https://github.com/KaiyangZhou/pytorch-center-loss/issues/20
    def __init__(self, num_class=10, num_feature=2):
        super(CenterLossv1, self).__init__()
        self.num_class = num_class
        self.num_feature = num_feature
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))
        nn.init.kaiming_normal_(self.centers, mode='fan_out', nonlinearity='relu')

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class CenterLossv2(nn.Module):
    # https://github.com/AkonLau/DTRG/blob/a708287fe2eb4ef44f3c139c8a6b33910307003e/coding_functions/center_loss.py#L4
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLossv2, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(),beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class OnlineLabelSmoothing(nn.Module):
    # https://github.com/AkonLau/DTRG/blob/a708287fe2eb4ef44f3c139c8a6b33910307003e/coding_functions/online_label_smooth.py
    def __init__(self, num_classes=10, use_gpu=False):
        super().__init__()
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes))
        self.grad = torch.zeros((num_classes, num_classes))
        self.count = torch.zeros((num_classes, 1))
        self.ce_criterion = nn.CrossEntropyLoss().cuda()
        if use_gpu:
            self.matrix = self.matrix.cuda()
            self.grad = self.grad.cuda()
            self.count = self.count.cuda()

    def forward(self, x, target):
        if self.training:
            # accumulate correct predictions
            p = torch.softmax(x.detach(), dim=1)
            _, pred = torch.max(p, 1)
            correct_index = pred.eq(target)
            correct_p = p[correct_index]
            correct_label = target[correct_index]

            self.grad[correct_label] += correct_p
            self.grad.index_add_(0, correct_label, correct_p)
            self.count.index_add_(0, correct_label, torch.ones_like(correct_label.view(-1, 1), dtype=torch.float32))


        target = target.view(-1,)
        logprobs = torch.log_softmax(x, dim=-1)

        softlabel = self.matrix[target]
        ols_loss = (- softlabel * logprobs).sum(dim=-1)

        loss = 0.5 * self.ce_criterion(x, target) + 0.5 * ols_loss.mean()

        return loss

    def update(self):
        index = torch.where(self.count > 0)[0]
        self.grad[index] = self.grad[index] / self.count[index]
        # reset matrix and update
        nn.init.constant_(self.matrix, 0.)
        norm = self.grad.sum(dim=1).view(-1, 1)
        index = torch.where(norm > 0)[0]
        self.matrix[index] = self.grad[index] / norm[index]
        # reset
        nn.init.constant_(self.grad, 0.)
        nn.init.constant_(self.count, 0.)


class OnlineLabelSmoothingv1(nn.Module):
    # https://github.com/ankandrew/online-label-smoothing-pt/blob/main/ols/online_label_smooth.py
    """
    Implements Online Label Smoothing from paper
    https://arxiv.org/pdf/2011.12562.pdf
    """

    def __init__(self, alpha: float, n_classes: int, smoothing: float = 0.1):
        """
        :param alpha: Term for balancing soft_loss and hard_loss
        :param n_classes: Number of classes of the classification problem
        :param smoothing: Smoothing factor to be used during first epoch in soft_loss
        """
        super(OnlineLabelSmoothingv1, self).__init__()
        assert 0 <= alpha <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        self.n_classes = n_classes
        # Initialize soft labels with normal LS for first epoch
        self.register_buffer('supervise', torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Update matrix is used to supervise next epoch
        self.register_buffer('update', torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer('idx_count', torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: torch.Tensor, y: torch.Tensor):
        # Calculate the final loss
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: torch.Tensor, y: torch.Tensor):
        """
        Calculates the soft loss and calls step
        to update `update`.
        :param y_h: Predicted logits.
        :param y: Ground truth labels.
        :return: Calculates the soft loss based on current supervise matrix.
        """
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: torch.Tensor, y: torch.Tensor) -> None:
        """
        Updates `update` with the probabilities
        of the correct predictions and updates `idx_count` counter for
        later normalization.
        Steps:
            1. Calculate correct classified examples.
            2. Filter `y_h` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).
        Note on (5): This is done outside this function since we only need to
                     normalize at the end of the epoch.
        """
        # 1. Calculate predicted classes
        y_h_idx = y_h.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `memory`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """
        This function should be called at the end of the epoch.
        It basically sets the `supervise` matrix to be the `update`
        and re-initializes to zero this last matrix and `idx_count`.
        """
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()


if __name__=='__main__':
    labels = torch.randint(3, size=(7,))
    features = torch.randn((7, 128))
    loss_func = MinimaxLoss(128, 0.1)
    loss = loss_func(features, labels)
    print('triplet', loss)
