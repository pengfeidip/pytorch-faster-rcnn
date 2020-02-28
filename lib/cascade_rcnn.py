from torch import nn
from copy import deepcopy
from . import utils
import logging, time, random, traceback
import os.path as osp
import torch


class CascadeRCNN(nn.Module):
    def __init__(self,
                 backbone,
                 rpn_head,
                 rcnn_head,
                 train_cfg,
                 test_cfg):
        super(CascadeRCNN, self).__init__()
        from .registry import build_module
        self.backbone = build_module(backbone)
        self.rpn_head = build_module(rpn_head)
        self.num_stages = len(rcnn_head)
        heads = nn.ModuleList()
        for i in range(self.num_stages):
            heads.append(build_module(rcnn_head[i]))
        self.rcnn_head = heads
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # gt_bbox: (4, n) where n is number of gt bboxes
    # gt_label: (n,)
    # scale: ogiginal_image_size * scale = img_data image size
    # 
    def forward_train(self, img_data, gt_bbox, gt_label, scale):
        logging.info('Start to forward in train mode')
        img_size = img_data.shape[-2:]
        feat = self.backbone(img_data)
        train_cfg = deepcopy(self.train_cfg)
        rpn_cls_loss, rpn_reg_loss, props \
            = self.rpn_head.forward_train(feat, gt_bbox, img_size, train_cfg, scale)
        rcnn_cls_losses = []
        rcnn_reg_losses = []
        for i in range(self.num_stages - 1):
            cur_rcnn_head = self.rcnn_head[i]
            cur_rcnn_train_cfg = train_cfg.rcnn[i]
            rcnn_cls_loss, rcnn_reg_loss \
                = cur_rcnn_head.forward_train(feat, props, gt_bbox, gt_label, cur_rcnn_train_cfg)
            rcnn_cls_losses.append(rcnn_cls_loss)
            rcnn_reg_losses.append(rcnn_reg_loss)
            refined_props, label, score = cur_rcnn_head.refine_props(feat, props, img_size)
            props = refined_props
        last_rcnn_head = self.rcnn_head[-1]
        last_rcnn_train_cfg = train_cfg.rcnn[-1]
        rcnn_cls_loss, rcnn_reg_loss = last_rcnn_head.forward_train(feat, props, gt_bbox, gt_label,
                                                                    last_rcnn_train_cfg)
        rcnn_cls_losses.append(rcnn_cls_loss)
        rcnn_reg_losses.append(rcnn_reg_loss)
        return rpn_cls_loss, rpn_reg_loss, rcnn_cls_losses, rcnn_reg_losses


class CascadeRCNNTrain(object):
    def __init__(self,
                 cascade_cfg,
                 dataloader,
                 work_dir,
                 total_epochs,
                 optimizer,
                 log_file,
                 lr_decay,
                 save_interval,
                 device,
                 train_cfg,
                 test_cfg):
        self.cascade_cfg=cascade_cfg
        self.dataloader=dataloader
        self.work_dir=osp.realpath(work_dir)
        self.total_epochs=total_epochs
        self.optimizer_cfg = optimizer
        self.log_file=log_file
        self.lr_decay=lr_decay
        self.save_interval=save_interval
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
        
        # set logging
        self.log_file=log_file
        if log_file is not None:
            self.log_file=osp.join(self.work_dir, log_file)
        log_cfg = {
            'format':'%(asctime)s: %(message)s\t[%(levelname)s]',
            'datefmt':'%y%m%d_%H%M%S_%a',
            'level': logging.DEBUG
        }
        if self.log_file is not None:
            log_cfg['filename']=self.log_file
        logging.basicConfig(**log_cfg)
        self.current_epoch=1
        self.device=torch.device(device)

    def init_detector(self):
        from .registry import build_module
        self.cascade_rcnn = build_module(self.cascade_cfg, train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        logging.info('Initialized Cascade RCNN in CascadeRCNNTrain')
        logging.info(self.cascade_rcnn)

    def create_optimizer(self):
        optimizer_type = self.optimizer_cfg.type
        assert optimizer_type in ['SGD']
        weight_decay = self.optimizer_cfg.weight_decay
        momentum = self.optimizer_cfg.momentum
        lr = self.optimizer_cfg.lr
        params = []
        for key, value in dict(self.cascade_rcnn.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        self.optimizer=torch.optim.SGD(params, momentum=momentum)
        return self.optimizer

    def decay_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    def train_one_iter(self, iter_i, epoch, train_data):
        img_data, bboxes, labels, scale, _ = train_data
        img_data = img_data.to(self.device)
        scale = scale.item()
        labels = labels.squeeze(0) + 1
        labels = labels.to(self.device)
        bboxes = bboxes.squeeze(0).t().to(self.device)
        bboxes = torch.stack([bboxes[1], bboxes[0], bboxes[3], bboxes[2]])
        logging.info('At epoch {}, iteration {}'.center(50, '*').format(epoch, iter_i))
        self.optimizer.zero_grad()
        logging.debug('Image Shape: {}'.format(img_data.shape))
        logging.debug('GT bboxes: {}'.format(bboxes.t()))
        logging.debug('GT labels: {}'.format(labels))
        logging.debug('Scale: {}'.format(scale))

        rpn_cls_loss, rpn_reg_loss, rcnn_cls_losses, rcnn_reg_losses \
            = self.cascade_rcnn.forward_train(img_data, bboxes, labels, scale)

        rcnn_cls_loss_nums = [rcnn_cls_losses[i].item() * self.train_cfg.stage_loss_weight[i] \
                              for i in range(self.cascade_rcnn.num_stages)]
        rcnn_reg_loss_nums = [rcnn_reg_losses[i].item() * self.train_cfg.stage_loss_weight[i] \
                              for i in range(self.cascade_rcnn.num_stages)]

        logging.debug('rpn_cls_loss:  {}'.format(rpn_cls_loss.item()))
        logging.debug('rpn_reg_loss:  {}'.format(rpn_reg_loss.item()))
        logging.debug('rcnn_cls_loss: {}'.format(sum(rcnn_cls_loss_nums)))
        logging.debug('rcnn_reg_loss: {}'.format(sum(rcnn_reg_loss_nums)))
        for i in range(self.cascade_rcnn.num_stages):
            logging.debug('rcnn_cls_loss {}: {}'.format(i, rcnn_cls_loss_nums[i]))
            logging.debug('rcnn_reg_loss {}: {}'.format(i, rcnn_reg_loss_nums[i]))
        
        comb_loss = rpn_cls_loss + rpn_reg_loss
        for i in range(self.cascade_rcnn.num_stages):
            comb_loss = comb_loss \
                        + self.train_cfg.stage_loss_weight[i] * rcnn_cls_losses[i] \
                        + self.train_cfg.stage_loss_weight[i] * rcnn_reg_losses[i]
        logging.debug('Combined loss: {}'.format(comb_loss.item()))
        comb_loss.backward()
        self.optimizer.step()

    def train(self):
        logging.info('Start a new round of training, start with epoch {}'.format(self.current_epoch))
        self.cascade_rcnn.to(device=self.device)
        self.cascade_rcnn.train()

        dataset_size = len(self.dataloader)
        tot_iters = dataset_size * (self.total_epochs - self.current_epoch + 1)
        eta_iters, eta_ct, iter_ct = 200, 0, 0
        start = time.time()

        self.create_optimizer()
        for epoch in range(self.current_epoch, self.total_epochs+1):
            if epoch in self.lr_decay:
                decay = self.lr_decay[epoch]
                logging.info('Learning rate decay={} at epoch={}'.format(decay, epoch))
                self.decay_lr(decay)
            logging.info('Start to train epoch={} with lr={}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            for iter_i, train_data in enumerate(self.dataloader):
                try:
                    self.train_one_iter(iter_i, epoch, train_data)
                    eta_ct += 1
                    iter_ct += 1
                    if eta_ct == eta_iters:
                        secs = time.time() - start
                        logging.info('Eta time: {} mins.'\
                                     .format((tot_iters - iter_ct) / eta_iters * secs / 60))
                        logging.info('FPS={}'.format(eta_iters/secs))
                        eta_ct = 0
                        start = time.time()
                except Exception as e:
                    logging.error('Traceback:')
                    logging.error(traceback.format_exc())
                    error_model = osp.join(self.work_dir, 'error_epoch{}_iter{}.pth'.format(
                        epoch, iter_i))
                    torch.save(self.cascade_rcnn.state_dict(), error_model)
                    logging.error(
                        'Encounter an error at epoch {}, iter {}, saved current model to {}.'\
                        .format(epoch, iter_i, error_model))
                    print('Training is interrupted by an error, :(.')
                    exit()

            epoch_model=osp.join(self.work_dir, 'epoch_{}.pth'.format(epoch))
            if epoch % self.save_interval == 0:
                logging.info('Finished training epoch {}, save trained model to {}'.format(
                    epoch, epoch_model))
                torch.save(self.cascade_rcnn.state_dict(), epoch_model)
        print('Finished Training:)!!!')

        
