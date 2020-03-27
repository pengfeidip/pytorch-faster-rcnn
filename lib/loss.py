import torch.nn as nn
import torch, logging
from . import region
import torch.nn.functional as F
from . import utils

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    logging.debug('IN Focal Loss'.center(50, '#'))
    logging.info('pred: {}, target: {}'\
                 .format(pred.shape, target.shape))
    logging.info('target 0:{}, >0:{}'.format((target==0).sum(), (target>0).sum()))
    num_samp, num_cls = pred.shape
    bg_places = (target==0)
    target = target - 1
    target[target==-1]=0
    tgt_tsr = pred.new_full(pred.size(), 0)
    tgt_tsr[torch.arange(num_samp), target] = 1
    tgt_tsr[bg_places,:]=0
    target = tgt_tsr

    # debug target
    any_true = target.to(dtype=bool).any(1)
    logging.info('target tensor pos:{}, neg:{}, values:{}'\
                 .format((any_true==True).sum(), (any_true==False).sum(), target.unique()))
    
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = pred_sigmoid * target + (1-pred_sigmoid)*(1-target)
    focal_weight = (alpha*target + (1-alpha)*(1-target)) * (1-pt).pow(gamma)
    # pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss.sum()

def cut_n(n, l=9):
    n = str(n)
    if n == '1':
        n = '1'.center(5,'*')
    if 'e' in n:
        return n
    if len(n) >= l:
        return n[:l]
    else:
        return n+' '*(l-len(n))

def debug_positive_targets(target, p, t):
    logging.debug('checking sigmoid probability of positive targets')
    pos_places = (target>0)
    pos_sig = p[pos_places]
    pos_target = t[pos_places]
    for x, y in zip(pos_sig, pos_target):
        _, max_sig = x.max(0)
        max_vec = torch.zeros(20)
        max_vec[max_sig]= 1
        max_vec = max_vec.tolist()
        x = x.tolist()
        y = y.tolist()
        x = '   '.join([cut_n(xx) for xx in x])
        y = '   '.join([cut_n(int(xx)) for xx in y])
        max_vec = '   '.join([cut_n(int(xx)) for xx in max_vec])
        logging.debug('\n'+'\n'.join([x,y,max_vec]))
    

def sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0, fix_alpha=False):
    '''
    Args:
        pred: [n, num_cls]
        target: [n], where 0 means background
    '''
    logging.debug('sigmoid_focal_loss'.center(50, '='))
    logging.debug('alpha={}, gamma={}'.format(alpha, gamma))
    n_samp, n_cls = pred.size()
    logging.debug('pred: {}, target: {}, unique vals of target: {}'\
                  .format(pred.shape, target.shape, target.unique()))
    logging.debug('check pred logits, max={}, min={}, mean={}'.format(pred.max(), pred.min(), pred.mean()))
    tar_one_hot  = utils.one_hot_embedding(target, n_cls+1) # [n, num_cls+1]
    tar_one_hot  = tar_one_hot[:, 1:]  # [n, num_cls]
    logging.debug('target 1:{}, 0:{}(/20)'.format((tar_one_hot==1).sum(), (tar_one_hot==0).sum()/20))
    pred_sigmoid  = pred.sigmoid()
    tar_one_hot = tar_one_hot.to(dtype=pred_sigmoid.dtype)
    pt = pred_sigmoid * tar_one_hot + (1-pred_sigmoid) * (1-tar_one_hot)
    logging.debug('checking pt, pt.numel()={}, pt==0:{}'.format(pt.numel(), (pt==0).sum()))
    if fix_alpha:
        focal_weight = alpha
    else:
        focal_weight  = alpha*tar_one_hot + (1-alpha)*(1-tar_one_hot)
    focal_weight  = focal_weight * (1-pt).pow(gamma)
    # debug_positive_target(target, p, t)
    if focal_weight.max() == 0.75:
        print('hahahahah')
    logging.debug('focal weight max={}, min={}, mean={}'.format(
        focal_weight.max(), focal_weight.min(), focal_weight.mean()))
    focal_loss = F.binary_cross_entropy_with_logits(pred, tar_one_hot, reduction='none') * focal_weight
    return focal_loss.sum()


def zero_loss(device):
    return torch.tensor(0.0, device=device, requires_grad=True)

# with no reduction
def smooth_l1_loss(x, y, sigma):
    assert x.shape == y.shape
    abs_diff = torch.abs(x - y)
    sigma_sqr = sigma**2
    mask = abs_diff < 1 / (sigma_sqr)
    val = mask.float() * (sigma_sqr / 2.0) * (abs_diff**2) + \
          (1 - mask.float()) * (abs_diff - 0.5/sigma_sqr)
    return val.sum()

def smooth_l1_loss_v2(x, y, beta):
    assert beta > 0
    assert x.shape == y.shape
    abs_diff = torch.abs(x-y)
    mask = abs_diff < beta
    val = mask.float() * (abs_diff**2) / (2*beta) + (1-mask.float()) * (abs_diff - 0.5 * beta)
    return val.sum()

def normalize_weight(filt_weight, bias):
    assert len(filt_weight) > 0
    device = filt_weight[0].device
    wloss = []
    for w in filt_weight:
        w = w.view(w.shape[0], -1)
        num_w = w.shape[0]
        wnorm = torch.norm(w, dim=1)
        abs_diff = torch.abs(wnorm-1).sum()
        wloss.append(abs_diff/num_w)
    bloss = []
    for b in bias:
        bnorm = torch.norm(b)
        num_b = b.shape[0]
        bloss.append(torch.abs(bnorm-1)/num_b)
    return sum(wloss), sum(bloss)


class RPNLoss(object):
    def __init__(self, lamb=1.0, sigma=3.0):
        self.lamb = lamb
        self.sigma = sigma
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def __call__(self, cls_out, reg_out, label, param):
        if label.numel() == 0:
            logging.warning('RPN receives no samples to train.')
            return zero_loss(cls_out.device)
        cls_loss = self.ce(cls_out.t(), label.long())
        n_samples = len(label)
        pos_arg = (label==1)
        if pos_arg.sum() == 0:
            logging.warning('RPN receives no positive samples.')
            reg_loss = zero_loss(cls_out.device)
        else:
            # reg_loss = self.smooth_l1(reg_out[:, pos_arg], param[:, pos_arg]) / n_samples
            reg_loss = smooth_l1_loss(reg_out[:, pos_arg], param[:, pos_arg],
                                      self.sigma) / n_samples
        return cls_loss + self.lamb * reg_loss

class RCNNLoss(object):
    def __init__(self, lamb=1.0, sigma=1.0):
        self.lamb = lamb
        self.sigma = sigma
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def __call__(self, cls_out, reg_out, label, param):
        if label.numel() == 0 or cls_out is None or reg_out is None:
            logging.warning('RCNN receives no training rois.')
            return zero_loss(label.device)
        label = label.long()
        n_class = cls_out.shape[1]
        n_samples = len(label)
        cls_loss = self.ce(cls_out, label)
        reg_out = reg_out.view(-1, 4, n_class)
        reg_out = reg_out[torch.arange(n_samples), :, label]
        pos_arg = (label>0) 
        if pos_arg.sum() == 0:
            logging.warning('RCNN recieves no positive samples.')
            reg_loss = zero_loss(label.device)
        else:
            pos_reg = reg_out[pos_arg, :]
            # reg_loss = self.smooth_l1(pos_reg, param[:, pos_arg].t()) / n_samples
            reg_loss = smooth_l1_loss(pos_reg, param[:, pos_arg].t(),
                                      self.sigma) / n_samples
        return cls_loss + self.lamb * reg_loss

class FocalLoss(object):
    pass
