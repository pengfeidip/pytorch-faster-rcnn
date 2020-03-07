from torch import nn
from copy import deepcopy
from . import utils
import logging, time, random, traceback
import os.path as osp
import torch, torchvision
from .bbox import bbox_target
from torch.nn.modules.batchnorm import _BatchNorm


class CascadeRCNN(nn.Module):
    def __init__(self,
                 num_stages=3,
                 backbone=None,
                 rpn_head=None,
                 roi_extractor=None,
                 shared_head=None,
                 rcnn_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CascadeRCNN, self).__init__()
        from .registry import build_module
        assert num_stages > 0
        self.num_stages=num_stages
        self.backbone = build_module(backbone)
        self.rpn_head = build_module(rpn_head)
        if isinstance(roi_extractor, list):
            assert len(roi_extractor) >= self.num_stages
            self.roi_extractors = nn.ModuleList([build_module(roi_extractor[i]) for i in range(self.num_stages)])

        else:
            self.roi_extractors = nn.ModuleList([build_module(roi_extractor) for _ in range(self.num_stages)])
        if shared_head is not None:
            self.shared_head = build_module(shared_head)
            self.with_shared_head=True
        else:
            self.with_shared_head=False

        if isinstance(rcnn_head, list):
            assert len(rcnn_head) >= self.num_stages
            rcnn_head=[build_module(rcnn_head[i]) for i in range(self.num_stages)]
        else:
            assert self.num_stages==1, 'rcnn_head must be consistent with num_stages'
            rcnn_head = [build_module(rcnn_head)]

        self.rcnn_head=nn.ModuleList(rcnn_head)
            
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        logging.info('Constructed CascadeRCNN')
        logging.info('Number of stages: {}'.format(self.num_stages))
        logging.info(str(self))


    def init_weights(self):
        self.rpn_head.init_weights()
        for i, bbox_head in enumerate(self.rcnn_head):
            bbox_head.init_weights()
            logging.info('(weights of bbox_head={})'.format(i))
        if self.with_shared_head:
            self.shared_head.init_weights()
        logging.info('Initialized weights for CascadeRCNN')


    # gt_bbox: (4, n) where n is number of gt bboxes
    # gt_label: (n,)
    # scale: ogiginal_image_size * scale = img_data image size
    def forward_train(self, img_data, gt_bbox, gt_label, scale):
        logging.info('Start to forward in train mode')
        img_size = img_data.shape[-2:]
        feat = self.backbone(img_data)
        train_cfg = deepcopy(self.train_cfg)
        rpn_cls_loss, rpn_reg_loss, props \
            = self.rpn_head.forward_train(feat, gt_bbox, img_size, train_cfg, scale)
        logging.debug('proposals from RPN: {}'.format(props.shape))
        rcnn_cls_losses = []
        rcnn_reg_losses = []
        for i in range(self.num_stages):
            logging.debug('IN RCNN_HEAD {}'.center(50, '-').format(i))
            logging.debug('props before target selecting: {}'.format(props.shape))

            cur_rcnn_head = self.rcnn_head[i]
            cur_roi_extractor = self.roi_extractors[i]
            cur_rcnn_train_cfg = train_cfg.rcnn[i]
            
            # sample positive and negative bboxes to train rcnn_head
            tar_props, tar_bbox, tar_label, tar_param, tar_is_gt \
                = bbox_target(props, gt_bbox, gt_label, cur_rcnn_train_cfg.assigner,
                              cur_rcnn_train_cfg.sampler,
                              cur_rcnn_head.target_means, cur_rcnn_head.target_stds)
            
            logging.debug('target proposals after selecting: {}'.format(tar_props.shape))
            # extract features from backbone based on the target bboxes
            roi_out = cur_roi_extractor(feat, tar_props)

            logging.debug('roi_out after roi_extractor: {}'.format(roi_out.shape))

            # shared_head can be regarded as part of rcnn_head, but it is shared among all rcnn_heads
            if self.with_shared_head:
                roi_out = self.shared_head(roi_out)
                logging.info('RoI output after shared head: {}'.format(roi_out.shape))

            cls_out, reg_out = cur_rcnn_head(roi_out)
            logging.debug('cls_out of rcnn_head: {}'.format(cls_out.shape))
            logging.debug('reg_out of rcnn_head: {}'.format(reg_out.shape))
            rcnn_cls_loss, rcnn_reg_loss \
                = cur_rcnn_head.loss(cls_out, reg_out, tar_label, tar_param)
            
            rcnn_cls_losses.append(rcnn_cls_loss)
            rcnn_reg_losses.append(rcnn_reg_loss)
            if i < self.num_stages-1:
                with torch.no_grad():
                    refined_props = cur_rcnn_head.refine_props(tar_props, tar_label, reg_out, tar_is_gt)
                    props = refined_props
                
        return rpn_cls_loss, rpn_reg_loss, rcnn_cls_losses, rcnn_reg_losses

    def forward_test(self, img_data, scale):
        logging.info('Star to forward in eval mode')
        img_size=img_data.shape[-2:]
        logging.info('Image size: {}'.format(img_size))
        feat=self.backbone(img_data)
        logging.debug('Feature size: {}'.format(feat.shape))
        test_cfg=deepcopy(self.test_cfg)
        props, score = self.rpn_head.forward_test(feat, img_size, test_cfg, scale)
        logging.info('Proposals from RPN: {}'.format(props.shape))
        
        cls_scores = []
        for i in range(self.num_stages):
            logging.info('In stage: {}'.format(i))
            if i < self.num_stages-1:
                img_size_ = None # for non-last stages, do not restrict bbox to image size
            else:
                img_size_ = img_size

            cur_roi_extractor = self.roi_extractors[i]
            roi_out = cur_roi_extractor(feat, props)
            logging.debug('roi_out after current roi_extractor: {}'.format(roi_out.shape))
            if self.with_shared_head:
                roi_out = self.shared_head(roi_out)
                logging.debug('roi_out after shared_head: {}'.format(roi_out.shape))
                
            cur_rcnn_head = self.rcnn_head[i]
            props, label, cls_out = cur_rcnn_head.forward_test(roi_out, props, img_size_)
            cls_scores.append(cls_out)

        cls_score = sum(cls_scores) / self.num_stages

        soft = torch.softmax(cls_score, dim=1)
        score, label = torch.max(soft, dim=1)

        num_classes = self.rcnn_head[0].num_classes
        nms_iou = self.test_cfg.rcnn.nms_iou
        min_score = self.test_cfg.rcnn.min_score
        bbox_res, score_res, class_res = [], [], []
        for i in range(num_classes - 1):
            cur_label = (label==i+1)
            if cur_label.numel()==0:
                continue
            if cur_label.sum() == 0:
                continue
            cur_score = score[cur_label]
            cur_bbox = props[:, cur_label]
            non_small = cur_score > min_score
            cur_score = cur_score[non_small]
            cur_bbox = cur_bbox[:, non_small]
            keep = torchvision.ops.nms(cur_bbox.t(), cur_score, nms_iou)
            bbox_res.append(cur_bbox[:, keep])
            score_res.append(cur_score[keep])
            class_res += [i+1] * len(keep)
        if len(bbox_res) == 0:
            return bbox_res, score_res, class_res

        return torch.cat(bbox_res, dim=1), torch.cat(score_res), class_res
        


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

        
class CascadeRCNNTest(object):
    def __init__(self,
                 cascade_cfg,
                 train_cfg,
                 test_cfg,
                 device=torch.device('cpu')):
        self.device=device
        self.cascade_cfg=deepcopy(cascade_cfg)
        self.train_cfg=train_cfg
        self.test_cfg=deepcopy(test_cfg)
        self.cascade_rcnn=None


    def init_detector(self):
        from .registry import build_module
        self.cascade_rcnn=build_module(self.cascade_cfg, train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        self.cascade_rcnn.to(self.device)

    def load_ckpt(self, ckpt):
        assert self.cascade_rcnn is not None
        self.current_ckpt=ckpt
        self.cascade_rcnn.load_state_dict(
            torch.load(ckpt, map_location=self.device))
        logging.info('loaded ckpt: {}'.format(ckpt))

    def inference(self, dataloader):
        self.cascade_rcnn.eval()
        inf_res, ith = [], 0
        logging.info('Start to inference {} images...'.format(len(dataloader)))
        logging.info('CascadeRCNN config:')
        logging.info(str(self.cascade_cfg))
        logging.info('Test config:')
        logging.info(str(self.test_cfg))
        with torch.no_grad():
            for img_data, img_size, bbox, label, difficult, iid in dataloader:
                tsr_size = img_data.shape[2:]
                iid = int(iid[0])
                logging.info('Inference {}-th image with image id: {}'.format(ith, iid))
                ith += 1
                scale=tsr_size[0]/img_size[0].item()
                scale=torch.tensor(scale, device=self.device)
                img_w, img_h=img_size
                img_res={'width':img_w, 'height': img_h, 'image_id':iid}
                bbox, score, category=self.inference_one(img_data, scale)
                if len(bbox)==0:
                    logging.warning('0 predictions for image {}'.format(iid))
                    continue
                img_res['bbox']=utils.xyxy2xywh(bbox).t()/scale
                img_res['score']=score
                img_res['category']=category
                logging.info('{} bbox predictions for {}-th image with image id {}'.format(bbox.shape[1], ith, iid))
                inf_res.append(img_res)
        return inf_res

    def inference_one(self, img_data, scale):
        img_data = img_data.to(device=self.device)
        h, w = img_data.shape[-2:]
        return self.cascade_rcnn.forward_test(img_data, scale)


    
        

    
