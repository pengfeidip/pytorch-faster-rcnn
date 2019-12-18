from . import region, modules, loss
import logging, random, traceback, time
import torch, torchvision
import torch.nn as nn
import os.path as osp
from PIL import Image


FASTER_ANCHOR_SCALES = [128, 256, 512]
FASTER_ANCHOR_ASPECT_RATIOS = [1.0, 0.5, 2.0]
FASTER_ROI_POOL_SIZE = (7, 7)


def ckpt_name(n):
    return 'epoch_{}.pth'.format(n)


# turn RCNN output into prediction result
# apply nms here indepently to each category
# returned bboxes are in xywh coordinates
def interpret_rcnn_output(rcnn_cls_out, rcnn_reg_out, props, nms_iou):
    logging.info('Interpret RCNN output(apply NMS with iou_thr={})'.format(nms_iou))
    logging.debug('Rcnn_cls_out: {}'.format(rcnn_cls_out.shape))
    logging.debug('Rcnn_reg_out: {}'.format(rcnn_reg_out.shape))
    logging.debug('Number of proposals: {}'.format(len(props)))

    batch_size, num_cates = rcnn_cls_out.shape
    rcnn_cls_soft = torch.softmax(rcnn_cls_out, 1)
    rcnn_cls_maxarg = torch.argmax(rcnn_cls_soft, 1)

    # put [cate, soft_score, param, adj_bbox] in the list
    raw_res = []

    for i, soft in enumerate(rcnn_cls_soft):
        cate = rcnn_cls_maxarg[i]
        soft_score = rcnn_cls_soft[i][cate]
        if cate == 0:
            continue
        raw_res.append([cate, soft_score.item(), rcnn_reg_out[i][cate*4:cate*4+4],
                        props[i]['adj_bbox']])
    logging.info('Positive bboxes: {}'.format(len(raw_res)))
    bboxes, scores, cates = [], [], []
    for cate in range(1, num_cates):
        cate_res = [tmp for tmp in raw_res if tmp[0]==cate]
        cur_bboxes_xywh = [region.param2xywh(tmp[2], tmp[3]) for tmp in cate_res]
        cur_bboxes_xyxy = [region.BBox(xywh=tmp).get_xyxy() for tmp in cur_bboxes_xywh]
        cur_scores = [tmp[1] for tmp in cate_res]
        nms_keep = torchvision.ops.nms(torch.tensor(cur_bboxes_xyxy),
                                       torch.tensor(cur_scores), nms_iou)
        for keep in nms_keep:
            bboxes.append(cur_bboxes_xywh[keep])
            scores.append(cate_res[keep][1])
            cates.append(cate)
    logging.info('Positive bboxes after applying NMS to each category independently: {}'\
                 .format(len(bboxes)))
    return bboxes, scores, cates


class FasterRCNNTest(object):
    r"""
    Utility to test a faster rcnn
    """
    def __init__(self,
                 faster_configs,
                 device=torch.device('cpu')):
        
        self.device = device
        self.faster_configs = faster_configs
        self.faster_rcnn = FasterRCNNModule(**faster_configs)
        

    def load_ckpt(self, ckpt):
        self.current_ckpt = ckpt
        self.faster_rcnn.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.faster_rcnn.to(self.device)
        
    # inference on a set of images and return coco format json, but only return bbox results
    # {'image_id', 'bbox', 'score', 'category_id'}
    def inference(self, dataloader, min_score=-1):
        coco_json = {
            'images': [],
            'annotations': []
        }
        img_id, cate_id = 0, 0
        self.faster_rcnn.eval_mode()
        logging.info('Start to inference images({}).'.format(len(dataloader)))
        with torch.no_grad():
            for img_data, amp, img_name, img_w, img_h in dataloader:
                logging.info('Inference image id: {}'.format(img_id))
                amp = amp.item()
                img_name = img_name[0]
                img_w, img_h = img_w.item(), img_h.item()

                img_json = {
                    'id': img_id,
                    'width': img_w,
                    'height': img_h,
                    'file_name': img_name,
                }
                coco_json['images'].append(img_json)            
                bboxes_xywh, scores, categories = self.inference_one(img_data)
                bbox_ct = 0
                for i, bbox in enumerate(bboxes_xywh):
                    score = scores[i]
                    if score < min_score:
                        continue
                    bbox = [coor/amp for coor in bbox]
                    coco_json['annotations'].append({
                        'id': cate_id,
                        'image_id':img_id,
                        'bbox':bbox,
                        'score':score,
                        'category_id':categories[i]
                    })
                    bbox_ct += 1
                    cate_id += 1
                logging.info('{} bbox predictions for image: {}'.format(bbox_ct, img_name))
                img_id += 1
        return coco_json


    def inference_one(self, img_data):
        img_data = img_data.to(device=self.device)
        rpn_cls_out, rpn_reg_out, rcnn_cls_out, rcnn_reg_out,\
            anchor_targets, props, props_targets = self.faster_rcnn(img_data, None)
        bboxes, scores, categories = interpret_rcnn_output(rcnn_cls_out,
                                                           rcnn_reg_out,
                                                           props,
                                                           self.faster_rcnn.test_props_nms_iou)
        return bboxes, scores, categories



####################################################################
###################### Vectorization ###############################
####################################################################

DEF_CUDA = torch.device('cuda:0')

class FasterRCNNModule(nn.Module):
    r"""
    It consists of a Backbone, a RPN and a RCNN.
    It contains region related utilities to generate anchors, region proposals, rois etc.
    It is a reguler nn.Module.
    """
    def __init__(self,
                 num_classes=20,
                 anchor_scales=[8, 16, 32],
                 anchor_aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_pos_iou=0.7,
                 anchor_neg_iou=0.3,
                 anchor_max_pos=128,
                 anchor_max_targets=256,
                 train_props_pre_nms=12000,
                 train_props_post_nms=2000,
                 train_props_nms_iou=0.7,
                 train_props_min_size=16,
                 test_props_pre_nms=6000,
                 test_props_post_nms=300,
                 test_props_nms_iou=0.5,
                 test_props_min_size=16,
                 props_pos_iou=0.5,
                 props_neg_iou_hi=0.5,
                 props_neg_iou_lo=0.1,
                 props_max_pos=32,
                 props_max_targets=128,
                 roi_pool_size=FASTER_ROI_POOL_SIZE,
                 transfer_rcnn_fc=True,
                 device=DEF_CUDA):
        super(FasterRCNNModule, self).__init__()
        self.num_classes=num_classes
        self.anchor_scales=anchor_scales,
        self.anchor_aspect_ratios=anchor_aspect_ratios
        self.anchor_pos_iou=anchor_pos_iou
        self.anchor_neg_iou=anchor_neg_iou
        self.anchor_max_pos=anchor_max_pos
        self.anchor_max_targets=anchor_max_targets
        self.train_props_pre_nms=train_props_pre_nms
        self.train_props_post_nms=train_props_post_nms
        self.train_props_nms_iou=train_props_nms_iou
        self.train_props_min_size = train_props_min_size
        self.test_props_pre_nms=test_props_pre_nms
        self.test_props_post_nms=test_props_post_nms
        self.test_props_nms_iou=test_props_nms_iou
        self.test_props_min_size=test_props_min_size
        self.props_pos_iou=props_pos_iou
        self.props_neg_iou_hi=props_neg_iou_hi
        self.props_neg_iou_lo=props_neg_iou_lo
        self.props_max_pos=props_max_pos
        self.props_max_targets=props_max_targets
        self.roi_pool_size=roi_pool_size
        self.device = device

        # init anchor creator
        self.anchor_creator = region.AnchorCreator(scales=anchor_scales,
                                                   aspect_ratios=anchor_aspect_ratios,
                                                   device = self.device)
        # init anchor target creator
        self.anchor_target_creator = region.AnchorTargetCreator(
            pos_iou=self.anchor_pos_iou,
            neg_iou=self.anchor_neg_iou,
            max_pos=self.anchor_max_pos,
            max_targets=self.anchor_max_targets)
        # init proposal creator for training
        self.train_props_creator = region.ProposalCreator(
            max_pre_nms=self.train_props_pre_nms,
            max_post_nms=self.train_props_post_nms,
            nms_iou=self.train_props_nms_iou,
            min_size=self.train_props_min_size)
        # init proposal creator for testing
        self.test_props_creator = region.ProposalCreator(
            max_pre_nms=self.test_props_pre_nms,
            max_post_nms=self.test_props_post_nms,
            nms_iou=self.test_props_nms_iou,
            min_size=self.test_props_min_size)
        # init proposal target creator
        self.props_target_creator = region.ProposalTargetCreator(
            max_pos=self.props_max_pos,
            max_targets=self.props_max_targets,
            pos_iou=self.props_pos_iou,
            neg_iou_hi=self.props_neg_iou_hi,
            neg_iou_lo=self.props_neg_iou_lo)
        # init cropping and pooling layer
        self.roi_crop = region.ROICropping()
        self.roi_pool = region.ROIPooling(out_size=roi_pool_size)
        # next init networks
        self.backbone = modules.VGGBackbone()
        self.rpn = modules.RPN(num_classes=num_classes,
                               num_anchors=len(anchor_scales)*len(anchor_aspect_ratios))
        vgg16 = self.backbone.vgg16[0]
        fc1_state_dict=vgg16.classifier[0].state_dict() if transfer_rcnn_fc else None
        fc2_state_dict=vgg16.classifier[3].state_dict() if transfer_rcnn_fc else None
        self.rcnn = modules.RCNN(num_classes,
                                 fc1_state_dict,
                                 fc2_state_dict)
        self.training = True
        self.to(device)

    # It assumes that x only contains one image,
    # i.e. it only supports train/test one image at a time.
    # X.shape may look like (1, 3, 600, 1000), the size is already resized to
    # longer=1000 and shorter=600
    # the purpose of ground truth 'gt' is to generate rcnn output for training mode
    #
    # Pipeline for training:
    #
    #                         gt-----
    #                               |
    # feature->rpn_out->props(train)-->props_targets-->roi_crops->roi_pool->rcnn_out
    #       |                                       ^
    #       |                                       |
    #       -----------------------------------------
    #
    # Pipeline for testing:
    # feature->rpn_out->props(test)-->roi_crops->roi_pool->rcnn_out
    #       |                      ^
    #       |                      |
    #       ------------------------

    def forward(self, x, gt_bbox, gt_label, scale):
        """
        Args:
            x (tensor): input images, with size=(1, 3, H, W)
            gt_bbox (tensor): gt bboxes with size=(4, n) where n is number of gt and 4 is
                              coordinates (x_min, y_min, x_max, y_max)
            gt_label (tensor): class label of each gt bbox, size=(n, )
        """
        logging.info('Start to pass forward...')
        img_size, feature = self.forward_backbone(x)
        logging.info('Finished feature extraction: {}'.format(feature.shape))
        rpn_cls_out, rpn_reg_out, rpn_tar_cls_out, \
            rpn_tar_reg_out, rpn_tar_label, rpn_tar_param, anchors \
            = self.forward_rpn(feature, img_size, gt_bbox)
        logging.info('Finished rpn forward pass \ncls_out.shape={} \nreg_out.shape={} \nanchors: {}'
                     .format(rpn_cls_out.shape, rpn_reg_out.shape, anchors.shape))
        rcnn_cls_out, rcnn_reg_out, rcnn_tar_label, rcnn_tar_param  \
            = self.forward_rcnn(feature, img_size, rpn_cls_out, rpn_reg_out, \
                                anchors, gt_bbox, gt_label, scale)
        logging.info('Finished rcnn forward pass \ncls_out.shape={} \nreg_out.shape={}'\
                     .format(rcnn_cls_out.shape, rcnn_reg_out.shape))
        return rpn_tar_cls_out, rpn_tar_reg_out, rpn_tar_label, rpn_tar_param, \
            rcnn_cls_out, rcnn_reg_out, rcnn_tar_label, rcnn_tar_param
            
    def forward_backbone(self, x):
        img_size = x.shape[-2:]
        feature = self.backbone(x)
        feat_size = feature.shape[-2:]
        return img_size, feature

    def forward_rpn(self, feat, img_size, gt_bbox):
        feat_size = feat.shape[-2:]
        rpn_cls_out, rpn_reg_out = self.rpn(feat)
        logging.info('rpn_cls_out shape: {}'.format(
            rpn_cls_out.shape if rpn_cls_out is not None else None))
        logging.info('rpn_reg_out shape: {}'.format(
            rpn_reg_out.shape if rpn_reg_out is not None else None))
        anchors = self.anchor_creator(img_size, feat_size)
        anchors = anchors.view(4, -1)
        tar_cls_out, tar_reg_out, tar_label, tar_params \
            = None, None, None, None

        if self.training:
            inside_idx = region.find_inside_index(anchors, img_size)
            in_anchors = anchors[:, inside_idx]
            # label is label of inside anchors, 1=pos, 0=neg and -1=ignore
            label, param, bbox_labels \
                = self.anchor_target_creator(img_size, feat_size, in_anchors, gt_bbox)
            # non_neg_label is chosen index of the inside anchors
            non_neg_label = (label!=-1)
            inside_arg = torch.nonzero(inside_idx)
            # chosen is chosen index of all the anchors
            chosen = inside_arg[label!=-1].squeeze()
            cls_out = rpn_cls_out.view(2, -1)
            reg_out = rpn_reg_out.view(4, -1)
            assert cls_out.shape[-1] == reg_out.shape[-1]
            tar_cls_out = cls_out[:, chosen]
            tar_reg_out = reg_out[:, chosen]
            tar_label = label[non_neg_label]
            tar_param = param[:, non_neg_label]
            logging.info('rpn tar_cls_out.shape: {}'.format(tar_cls_out.shape))
            logging.info('rpn tar_reg_out.shape: {}'.format(tar_reg_out.shape))
            logging.info('rpn tar_label.shape: {}'.format(tar_label.shape))
            logging.info('rpn tar_label: pos={}, neg={}, ignore={}'.format(
                (tar_label==1).sum(),
                (tar_label==0).sum(),
                (tar_label==-1).sum()))
            return \
            rpn_cls_out, rpn_reg_out, \
            tar_cls_out, tar_reg_out, tar_label, tar_param, \
            anchors

    def forward_rcnn(self,
                     feature, img_size, rpn_cls_out, rpn_reg_out, anchors,
                     gt_bbox, gt_label, scale):
        feat_size = feature.shape[-2:]
        if self.training:
            props, score \
                = self.train_props_creator(rpn_cls_out, rpn_reg_out, anchors, img_size, scale)
        else:
            props, score \
                = self.test_props_creator(rpn_cls_out, rpn_reg_out, anchors, img_size, scale)
        logging.info('proposals: {}'.format(props.shape))
        
        tar_label, tar_param = None, None
        if self.training:
            tar_bbox, tar_label, tar_param \
                = self.props_target_creator(props, gt_bbox, gt_label)
            logging.info('rcnn tar_bbox: {}'.format(tar_bbox.shape))
            logging.info('rcnn tar_label.shape: {}'.format(tar_label.shape))
            logging.info('rcnn tar_label: pos={}, neg={}'.format(
                (tar_label>0).sum(),
                (tar_label==0).sum()))
            if tar_label.numel()==0:
                logging.warning('rcnn receives 0 proposals to train, '
                                'this is probably due to low IoU of proposals with GT')

        if self.training:
            roi_crops = self.roi_crop(feature, tar_bbox, img_size)
            roi_pool_out = self.roi_pool(roi_crops)
            rcnn_cls_out, rcnn_reg_out = self.rcnn(roi_pool_out)
        else:
            roi_crops = self.roi_crop.crop(feature, props, img_size)
            roi_pool_out = self.roi_pool(roi_crops)
            rcnn_cls_out, rcnn_reg_out = self.rcnn(roi_pool_out)
        # anchor_targets, props_targets will be None for test mode
        return rcnn_cls_out, rcnn_reg_out, tar_label, tar_param

    def train_mode(self):
        super(FasterRCNNModule, self).train()
        self.training = True
    def eval_mode(self):
        super(FasterRCNNModule, self).eval()
        self.training = False



class FasterRCNNTrain(object):
    r"""
    Provide a utility to train a faster rcnn.
    """
    def __init__(self,
                 faster_configs,
                 dataloader,
                 work_dir,
                 max_epochs,
                 optim=torch.optim.SGD,
                 optim_kwargs=dict(lr=0.001,momentum=0.9,weight_decay=0.0005),
                 lr_scheduler=None,
                 rpn_loss_lambda=1.0,
                 rcnn_loss_lambda=1.0,
                 loss_lambda=1.0,
                 log_file=None,
                 log_level=logging.INFO,
                 device=torch.device('cpu'),
                 save_interval=2,
                 rpn_only=False):
        # get real path
        work_dir = osp.realpath(work_dir)
        
        # set model level configs
        self.faster_configs = faster_configs
        self.dataloader = dataloader
        self.max_epochs = max_epochs
        self.work_dir = work_dir
        assert osp.isdir(work_dir), 'work_dir not exists: {}'.format(work_dir)
        self.optim = optim
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler=lr_scheduler

        # set logging
        self.log_file = log_file
        self.log_level = log_level
        if log_file is None:
            logging.getLogger().disabled = True
        else:
            self.log_file = osp.join(work_dir, log_file)
            logging.basicConfig(filename=self.log_file,
                                format='%(asctime)s: %(message)s\t[%(levelname)s]',
                                datefmt='%y%m%d_%H%M%S_%a',
                                level=log_level)
        # do not init the net yet
        self.faster_rcnn = None
        self.current_epoch = 1
        self.device=device
        self.rpn_loss_lambda = rpn_loss_lambda
        self.rcnn_loss_lambda = rcnn_loss_lambda
        self.loss_lambda = loss_lambda
        self.save_interval = save_interval
        self.rpn_only = rpn_only

    def to(self, device):
        self.device=device

    def init_module(self):
        self.faster_rcnn = FasterRCNNModule(**(self.faster_configs))
        logging.info('Initialized faster rcnn in FasterRCNNTrain')
        logging.info(self.faster_rcnn)

    def resume_from(self, epoch):
        ckpt = self.get_ckpt(epoch)
        self.faster_rcnn.load_state_dict(torch.load(ckpt))
        self.current_epoch = epoch + 1
        logging.info('Resume from epoch: {}, ckpt: {}'.format(epoch, ckpt))
        logging.info(self.faster_rcnn)

    def get_ckpt(self, epoch):
        return osp.join(self.work_dir, ckpt_name(epoch))

    def train_one_iter(self, iter_i, epoch, train_data, optimizer, rpn_loss, rcnn_loss):
        logging.info('At epoch {}, iteration {}.'.center(50, '*').format(epoch, iter_i))
        optimizer.zero_grad()
        img_data, bboxes, lables, img_info = train_data
        bboxes = bboxes.squeeze(0).t()
        labels = lables.squeeze(0)
        img_data = img_data.to(self.device)
        bboxes = bboxes.to(self.device)
        labels = labels.to(self.device)
        scale = img_info['scale'].item()
        
        logging.debug('Image shape: {}'.format(img_data.shape))
        logging.debug('GT bboxes: {}'.format(bboxes))
        logging.debug('GT labels: {}'.format(labels))
        logging.debug('Image info: {}'.format(img_info))

        rpn_tar_cls, rpn_tar_reg, rpn_tar_label, rpn_tar_param, \
            rcnn_cls, rcnn_reg, rcnn_tar_label, rcnn_tar_param \
            = self.faster_rcnn(img_data, bboxes, labels, scale)
        
        rpnloss = rpn_loss(rpn_tar_cls, rpn_tar_reg, rpn_tar_label, rpn_tar_param)
        rcnnloss = rcnn_loss(rcnn_cls, rcnn_reg, rcnn_tar_label, rcnn_tar_param)
        combloss = rpnloss + self.loss_lambda * rcnnloss
        logging.info('RPN loss: {}'.format(rpnloss.item()))
        logging.info('RCNN loss: {}'.format(rcnnloss.item()))
        logging.info('Combined loss: {}'.format(combloss.item()))
        combloss.backward()
        optimizer.step()

    def train(self):
        logging.info('Start a new round of training, start with epoch {}'\
                     .format(self.current_epoch))
        rpn_loss = loss.RPNLoss(self.rpn_loss_lambda)
        rcnn_loss = loss.RCNNLoss(self.rcnn_loss_lambda)
        self.faster_rcnn.to(device=self.device)
        self.faster_rcnn.train_mode()

        dataset_size = len(self.dataloader)
        tot_iters = dataset_size * (self.max_epochs - self.current_epoch)
        eta_iters, eta_ct, iter_ct = 50, 0, 0
        start = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs+1):
            if self.lr_scheduler is not None:
                self.optim_kwargs['lr'] = self.lr_scheduler(epoch)
            optimizer = self.optim(self.faster_rcnn.parameters(),
                                  **(self.optim_kwargs))
            logging.info('Start to train epoch: {} using lr: {}.'\
                         .format(epoch, self.optim_kwargs['lr']))
            for iter_i, train_data in enumerate(self.dataloader):
                # train one image
                try:
                    self.train_one_iter(iter_i, epoch, train_data, optimizer, rpn_loss, rcnn_loss)
                    eta_ct += 1
                    iter_ct += 1
                    if eta_ct == eta_iters:
                        secs = time.time() - start
                        logging.info('Eta time: {} mins.'\
                                     .format((tot_iters - iter_ct) / eta_iters * secs / 60))
                        eta_ct = 0
                        start = time.time()
                except Exception as e:
                    logging.error('Traceback:')
                    logging.error(traceback.format_exc())
                    error_model = osp.join(self.work_dir, 'error_epoch{}_iter{}.pth'.format(
                        epoch, iter_i))
                    torch.save(self.faster_rcnn.state_dict(), error_model)
                    logging.error(
                        'Encounter an error at epoch {}, iter {}, saved current model to {}.'\
                        .format(epoch, iter_i, error_model))
                    print('Training is interrupted by an error, :(.')
                    exit()
                    
            epoch_model = osp.join(self.work_dir, ckpt_name(epoch))
            if epoch % self.save_interval == 0:
                logging.info('Finished traning epoch {}, save trained model to {}'.format(
                    epoch, epoch_model))
                torch.save(self.faster_rcnn.state_dict(), epoch_model)
        print('Finished Training:)!!!')

