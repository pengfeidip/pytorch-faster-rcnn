from torch import nn
from copy import deepcopy, copy
from . import utils
import logging, time, random, traceback
import os.path as osp
import torch, torchvision
from .bbox import bbox_target
from torch.nn.modules.batchnorm import _BatchNorm


class RetinaNet(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(RetinaNet, self).__init__()
        from .registry import build_module
        self.backbone = build_module(backbone)
        if neck is not None:
            self.neck = build_module(neck)
            self.with_neck = True
        else:
            self.with_neck = False

        self.bbox_head = build_module(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.init_weights()
        logging.info('Constructed RetinaNet')

    def init_weights(self):
        if self.with_neck:
            self.neck.init_weights()
        self.bbox_head.init_weights()
        logging.info('Initialized weights for RetinaNet')

    def extract_feat(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img_data, img_size, gt_bbox, gt_label, scale):
        logging.info('Start to forward in train mode')
        feats = self.extract_feat(img_data)
        train_cfg = self.train_cfg
        pad_size = img_data.shape[-2:]
        cls_loss, reg_loss = self.bbox_head.forward_train(feats, gt_bbox, gt_label,
                                                          img_size, pad_size, train_cfg, scale)
        return cls_loss, reg_loss

    def forward_test(self, img_data, img_size, scale):
        logging.info('Star to forward in eval mode')
        logging.info('Image size: {}'.format(img_size))
        feats=self.extract_feat(img_data)
        logging.debug('Feature size: \n{}'.format(
            '\n'.join([str(feat.shape) for feat in feats])))
        test_cfg=self.test_cfg
        pad_size = img_data.shape[-2:]
        bbox, score, label = self.bbox_head.forward_test(
            feats, img_size, pad_size, self.test_cfg, scale
        )
        return bbox, score, label


class RetinaNetTrain(object):
    def __init__(self,
                 retinanet_cfg,
                 dataloader,
                 work_dir,
                 total_epochs,
                 optimizer,
                 log_file,
                 save_interval,
                 device,
                 lr_cfg,
                 train_cfg,
                 test_cfg):
        self.retinanet_cfg=retinanet_cfg
        self.dataloader=dataloader
        self.work_dir=osp.realpath(work_dir)
        self.total_epochs=total_epochs
        self.optimizer_cfg = copy(optimizer)
        self.log_file=log_file
        self.save_interval=save_interval
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
        self.lr_cfg=lr_cfg
        
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
        self.current_iter=1
        self.device=torch.device(device)

    def init_detector(self):
        from .registry import build_module
        self.retinanet = build_module(self.retinanet_cfg, train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        logging.info('Initialized RetinaNet in RetinaNetTrain')
        logging.info(self.retinanet)

    def create_optimizer(self):
        optimizer_type = self.optimizer_cfg.pop('type')
        assert optimizer_type in ['SGD']
        self.optimizer = torch.optim.SGD(self.retinanet.parameters(), **self.optimizer_cfg)
        return self.optimizer

    def decay_lr(self, decay=0.1):
        new_lr = []
        for param_group in self.optimizer.param_groups:
            cur_lr = param_group['lr'] * decay
            new_lr.append(cur_lr)
            param_group['lr'] = cur_lr
        return new_lr

    def set_lr(self, lr):
        if isinstance(lr, (float, int)):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif isinstance(lr, list):
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr[i]
        else:
            raise ValueError('lr must either be a list of numbers or a single number')

    def warmup(self):
        lr_cfg = self.lr_cfg
        if self.current_iter > lr_cfg.warmup_iters:
            pass
        elif self.current_iter == lr_cfg.warmup_iters:
            self.set_lr(self.initial_lr)
        else:
            k = (1-self.current_iter/lr_cfg.warmup_iters) * (1-lr_cfg.warmup_ratio)
            self.set_lr(self.initial_lr * k)
        return

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def train_one_iter(self, iter_i, epoch, train_data):
        self.warmup()
        # get train data we need
        img_meta = train_data['img_meta'].data[0][0]
        img_data = train_data['img'].data[0]
        bboxes = train_data['gt_bboxes'].data[0][0]
        bboxes = bboxes.t()
        labels = train_data['gt_labels'].data[0][0]
        scale, img_size, pad_size = img_meta['scale_factor'], img_meta['img_shape'], img_meta['pad_shape']
        # get train data we need
        img_size = img_size[:2]
        img_data = img_data.to(self.device)

        labels = labels.to(self.device)
        bboxes = bboxes.to(self.device)

        logging.info('At epoch {}, iteration {}'.center(50, '*').format(epoch, iter_i))
        logging.info('Current lr: {}'.format(self.get_lr()))
        self.optimizer.zero_grad()
        logging.debug('Image size: {}'.format(img_size))
        logging.debug('Image data size: {}'.format(img_data.shape))
        logging.debug('Pad size: {}'.format(pad_size))
        logging.debug('GT bboxes: {}'.format(bboxes.t()))
        logging.debug('GT labels: {}'.format(labels))
        logging.debug('Scale: {}'.format(scale))

        cls_loss, reg_loss = self.retinanet.forward_train(img_data, img_size, bboxes, labels, scale)
        comb_loss = cls_loss + reg_loss
        logging.info('cls_loss: {}'.format(cls_loss.item()))
        logging.info('reg_loss: {}'.format(reg_loss.item()))
        logging.info('tot_loss: {}'.format(comb_loss.item()))

        comb_loss.backward()
        self.optimizer.step()
        logging.debug('AFTER step, check grad self.retinanet.bbox_head.retina_cls.weight.grad.mean:{}'\
                      .format(self.retinanet.bbox_head.retina_cls.weight.grad.mean()))
        self.current_iter += 1

    def train(self):
        logging.info('Start a new round of training, start with epoch {}'.format(self.current_epoch))
        self.retinanet.to(device=self.device)
        self.retinanet.train()

        dataset_size = len(self.dataloader)
        tot_iters = dataset_size * (self.total_epochs - self.current_epoch + 1)
        eta_iters, eta_ct, iter_ct = 200, 0, 0
        start = time.time()
        self.create_optimizer()
        self.initial_lr = self.get_lr()[0]
        logging.info('initial_lr: {}'.format(self.initial_lr))
        for epoch in range(self.current_epoch, self.total_epochs+1):
            if epoch in self.lr_cfg.lr_decay:
                decay = self.lr_cfg.lr_decay[epoch]
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
                    torch.save(self.retinanet.state_dict(), error_model)
                    logging.error(
                        'Encounter an error at epoch {}, iter {}, saved current model to {}.'\
                        .format(epoch, iter_i, error_model))
                    print('Training is interrupted by an error, :(.')
                    exit()

            epoch_model=osp.join(self.work_dir, 'epoch_{}.pth'.format(epoch))
            if epoch % self.save_interval == 0:
                logging.info('Finished training epoch {}, save trained model to {}'.format(
                    epoch, epoch_model))
                torch.save(self.retinanet.state_dict(), epoch_model)
        print('Finished Training:)!!!')

        
class RetinaNetTest(object):
    def __init__(self,
                 retinanet_cfg,
                 train_cfg,
                 test_cfg,
                 device=torch.device('cpu')):
        self.device=device
        self.retinanet_cfg=deepcopy(retinanet_cfg)
        self.train_cfg=train_cfg
        self.test_cfg=deepcopy(test_cfg)

    def init_detector(self):
        from .registry import build_module
        self.retinanet=build_module(self.retinanet_cfg, train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        self.retinanet.to(self.device)

    def load_ckpt(self, ckpt):
        assert self.retinanet is not None
        self.current_ckpt=ckpt
        self.retinanet.load_state_dict(
            torch.load(ckpt, map_location=self.device))
        logging.info('loaded ckpt: {}'.format(ckpt))

    def inference(self, dataloader):
        self.retinanet.eval()
        inf_res, ith = [], 0
        logging.info('Start to inference {} images...'.format(len(dataloader)))
        logging.info('RetinaNet config:')
        logging.info(str(self.retinanet_cfg))
        logging.info('Test config:')
        logging.info(str(self.test_cfg))
        with torch.no_grad():
            for test_data in dataloader:
                img_meta = test_data['img_meta'].data[0][0]
                img_data = test_data['img'].data[0]
                scale, img_size, pad_size = [img_meta[k] for k in ['scale_factor', 'img_shape', 'pad_shape']]
                img_size = img_size[:2]
                ori_size = img_meta['ori_shape'][:2]
                # need to improve this
                filename = img_meta['filename']
                filename = osp.basename(filename)
                iid = int(filename[:-4])

                logging.info('Inference {}-th image with image id: {}'.format(ith, iid))
                ith += 1
                scale=torch.tensor(scale, device=self.device)
                img_w, img_h=ori_size
                img_res={'width':img_w, 'height': img_h, 'image_id':iid}
                bbox, score, category=self.inference_one(img_data, img_size, scale)
                if len(bbox)==0:
                    logging.warning('0 predictions for image {}'.format(iid))
                    continue
                img_res['bbox']=utils.xyxy2xywh(bbox).t()/scale
                img_res['score']=score
                img_res['category']=category
                logging.info('{} bbox predictions for {}-th image with image id {}'.format(bbox.shape[1], ith, iid))
                inf_res.append(img_res)
        return inf_res

    def inference_one(self, img_data, img_size, scale):
        img_data = img_data.to(device=self.device)
        return self.retinanet.forward_test(img_data, img_size, scale)


    
        

    
