from . import region, modules, loss
import logging, random, traceback, time
import torch, torchvision
import torch.nn as nn
import os.path as osp
from PIL import Image


FASTER_ANCHOR_SCALES = [128, 256, 512]
FASTER_ANCHOR_ASPECT_RATIOS = [1.0, 0.5, 2.0]
FASTER_ROI_POOL_SIZE = (7, 7)

class FasterRCNNModule(nn.Module):
    r"""
    It consists of a Backbone, a RPN and a RCNN.
    It contains region related utilities to generate anchors, region proposals, rois etc.
    It is a reguler nn.Module.
    """
    def __init__(self,
                 num_classes=20,
                 anchor_scales=FASTER_ANCHOR_SCALES,
                 anchor_aspect_ratios=FASTER_ANCHOR_ASPECT_RATIOS,
                 anchor_pos_iou=0.7,
                 anchor_neg_iou=0.3,
                 anchor_max_pos=128,
                 anchor_max_targets=256,
                 train_props_pre_nms=12000,
                 train_props_post_nms=2000,
                 train_props_nms_iou=0.7,
                 test_props_pre_nms=6000,
                 test_props_post_nms=300,
                 test_props_nms_iou=0.5,
                 props_pos_iou=0.5,
                 props_neg_iou=0.1,
                 props_max_pos=32,
                 props_max_targets=128,
                 roi_pool_size=FASTER_ROI_POOL_SIZE,
                 transfer_rcnn_fc=True
    ):
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
        self.test_props_pre_nms=test_props_pre_nms
        self.test_props_post_nms=test_props_post_nms
        self.test_props_nms_iou=test_props_nms_iou
        self.props_pos_iou=props_pos_iou
        self.props_neg_iou=props_neg_iou
        self.props_max_pos=props_max_pos
        self.props_max_targets=props_max_targets
        self.roi_pool_size=roi_pool_size

        # next init region related utilities
        self.anchor_gen = region.AnchorGenerator(anchor_scales, anchor_aspect_ratios)
        self.anchor_target_gen = region.AnchorTargetCreator(self.anchor_gen)
        self.train_props_gen = region.ProposalCreator(
            self.anchor_gen,
            self.train_props_pre_nms,
            self.train_props_post_nms,
            self.train_props_nms_iou)
        self.test_props_gen = region.ProposalCreator(
            self.anchor_gen,
            self.test_props_pre_nms,
            self.test_props_post_nms,
            self.test_props_nms_iou)
        self.props_target_gen = region.ProposalTargetCreator()
        self.roi_crop = region.ROICropping()
        self.roi_pool = region.ROIPooling(output_size=roi_pool_size)
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
    
    def forward(self, x, gt):
        img_size = x.shape[-2:]
        feat = self.backbone(x)
        feat_size = feat.shape[-2:]

        rpn_cls_out, rpn_reg_out = self.rpn(feat)
        anchor_targets = None
        if self.training:
            anchor_targets = self.anchor_target_gen.targets(img_size, feat_size, gt)

        if self.training:
            props = self.train_props_gen.proposals_filtered(
                rpn_cls_out, rpn_reg_out, img_size, feat_size)
        else:
            props = self.test_props_gen.proposals_filtered(
                rpn_cls_out, rpn_reg_out, img_size, feat_size)
                
        props_targets = None
        if self.training:
            props_targets = self.props_target_gen.targets(props, gt)

        if self.training:
            roi_crops, props_targets = self.roi_crop.crop(img_size, feat, props_targets)
            roi_pool_out = self.roi_pool(roi_crops)
            rcnn_cls_out, rcnn_reg_out = self.rcnn(roi_pool_out)
        else:
            roi_crops, props = self.roi_crop.crop(img_size, feat, props)
            roi_pool_out = self.roi_pool(roi_crops)
            rcnn_cls_out, rcnn_reg_out = self.rcnn(roi_pool_out)
        # anchor_targets, props_targets will be None for test mode
        return \
            rpn_cls_out, rpn_reg_out, rcnn_cls_out, rcnn_reg_out,\
            anchor_targets, props, props_targets
    
    # WARNING: Notice that some of the configs can not be updated
    # Only allow following configs to change:
    #   anchor_scales, only the specific scales, not the number
    #   anchor_aspect_ratios, only the specific ratios, not the number
    #   anchor_pos_iou, anchor_neg_iou,     # for training RPN
    #   anchor_max_pos, anchor_max_targets  # for training RPN
    #   train_props_pre_nms, train_props_post_nms, train_props_nms_iou  
    #   test_props_pre_nms, test_props_post_nms, test_props_nms_iou
    #   props_pos_iou, props_neg_iou, props_max_pos, props_max_targets
    def update_config(self,
                      anchor_scales=None,
                      anchor_aspect_ratios=None,
                      anchor_pos_iou=None,
                      anchor_neg_iou=None,
                      anchor_max_pos=None,
                      anchor_max_targets=None,
                      train_props_pre_nms=None,
                      train_props_post_nms=None,
                      train_props_nms_iou=None,
                      test_props_pre_nms=None,
                      test_props_post_nms=None,
                      test_props_nms_iou=None,
                      props_pos_iou=None,
                      props_neg_iou=None,
                      props_max_pos=None,
                      props_max_targets=None):
        new_anchor_scales = self.anchor_scales
        if anchor_scales is not None:
            assert len(anchor_scales) == len(self.anchor_scales)
            new_anchor_scales = anchor_scales
        new_anchor_aspect_ratios = self.anchor_aspect_ratios
        if anchor_aspect_ratios is not None:
            assert len(anchor_aspect_ratios) == len(self.anchor_aspect_ratios)
            new_anchor_aspect_ratios = anchor_aspect_ratios
        if anchor_scale is not None or anchor_aspect_ratios is not None:
            self.anchor_scales = new_anchor_scales
            self.anchor_aspect_ratios = new_anchor_aspect_ratios
            self.anchor_gen = region.AnchorGenerator(anchor_scales, anchor_aspect_ratios)

        # TODO: update the other configs, remember to re-init the affected generators
            

        pass

    def train_mode(self):
        super(FasterRCNNModule, self).train()
        self.training = True
    def eval_mode(self):
        super(FasterRCNNModule, self).eval()
        self.training = False

def ckpt_name(n):
    return 'epoch_{}.pth'.format(n)

class FasterRCNNTrain(object):
    r"""
    Provide a utility to train a faster rcnn.
    
    Args:
      faster_net  # faster rcnn net
      dataloader  # dataloader
      work_dir    # to keep log, saved epochs, saved model before crash
      logger      # Turn Python's logging on and output to this file
                  # Turn logging off if it is None
    """
    def __init__(self,
                 faster_configs,
                 dataloader,
                 work_dir,
                 max_epochs,
                 optim=torch.optim.SGD,
                 optim_kwargs=dict(lr=0.001,momentum=0.9,weight_decay=0.0005),
                 rpn_loss_lambda=10.0,
                 rcnn_loss_lambda=10.0,
                 loss_lambda=1.0,
                 log_file=None,
                 seed=None,
                 log_level=logging.INFO,
                 device=torch.device('cpu')
    ):
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
        # set random seed, for all RNGs
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        # do not init the net yet
        self.faster_rcnn = None
        self.current_epoch = 1
        self.device=device
        self.rpn_loss_lambda = rpn_loss_lambda
        self.rcnn_loss_lambda = rcnn_loss_lambda
        self.loss_lambda = loss_lambda

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
        logging.info('At epoch {}, iteration {}.'.format(epoch, iter_i))
        optimizer.zero_grad()
        img_data, bboxes_data, img_info = train_data
        img_data = img_data.to(self.device)
        bboxes_data = bboxes_data.to(self.device)
        gt = region.GroundTruth(bboxes_data)
        logging.debug('Image shape: {}'.format(img_data.shape))
        logging.debug('GT bboxes: {}'.format(bboxes_data))
        logging.debug('Image info: {}'.format(img_info))
        rpn_cls_out, rpn_reg_out, rcnn_cls_out, rcnn_reg_out, \
            anchor_targets, props, props_targets = self.faster_rcnn(img_data, gt)
        if rpn_cls_out is None:
            logging.warning('rpn_cls_out is None')
        else:
            logging.info('rpn_cls_out shape: {}'.format(rpn_cls_out.shape))
        if rpn_reg_out is None:
            logging.warning('rpn_reg_out is None')
        else:
            logging.info('rpn_reg_out shape: {}'.format(rpn_reg_out.shape))
        if rcnn_cls_out is None:
            logging.warning('rcnn_cls_out is None')
        else:
            logging.info('rcnn_cls_out shape: {}'.format(rcnn_cls_out.shape))
        if rcnn_reg_out is None:
            logging.warning('rcnn_reg_out is None')
        else:
            logging.info('rcnn_reg_out shape: {}'.format(rcnn_reg_out.shape))
        logging.info('anchor targets: {}'.format(len(anchor_targets)))
        logging.info('proposals: {}'.format(len(props)))
        logging.info('proposal targets: {}'.format(len(props_targets)))
        
        rpnloss = rpn_loss(rpn_cls_out, rpn_reg_out, anchor_targets)
        rcnnloss = rcnn_loss(rcnn_cls_out, rcnn_reg_out, props_targets)
        combloss = rpnloss + self.loss_lambda * rcnnloss
        logging.info('RPN loss: {}'.format(rpnloss.item()))
        logging.info('RCNN loss: {}'.format(rcnnloss.item()))
        logging.info('Combined loss: {}'.format(combloss.item()))
        combloss.backward()
        optimizer.step()

    def train(self):
        optimizer = self.optim(self.faster_rcnn.parameters(),
                               **(self.optim_kwargs))
        logging.info('Start a new round of training, start with epoch {}'\
                     .format(self.current_epoch))
        logging.info('Optimizer: {}'.format(optimizer))
        rpn_loss = loss.RPNLoss(self.faster_rcnn.anchor_gen, self.rpn_loss_lambda)
        rcnn_loss = loss.RCNNLoss(self.rcnn_loss_lambda)
        self.faster_rcnn.to(device=self.device)
        self.faster_rcnn.train_mode()

        dataset_size = len(self.dataloader)
        tot_iters = dataset_size * (self.max_epochs - self.current_epoch)
        eta_iters = 50
        eta_ct = 0
        iter_ct = 0
        start = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs+1):
            logging.info('Start to train epoch: {}.'.format(epoch))
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
                except:
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
            logging.info('Finished traning epoch {}, save trained model to {}'.format(
                epoch, epoch_model))
            torch.save(self.faster_rcnn.state_dict(), epoch_model)
        print('Finished Training:)!!!')



# turn RCNN output into prediction result
# apply nms here indepently to each category
def interpret_rcnn_output(rcnn_cls_out, rcnn_reg_out, props, nms_iou):
    logging.info('Interpret RCNN output(apply NMS with iou_thr={})'.format(nms_iou))
    batch_size, num_cates = rcnn_cls_out.shape

    bboxes = []
    scores = []
    cates = []
    for i, cls_out in enumerate(rcnn_cls_out):
        prop = props[i]
        max_arg = torch.argmax(cls_out)
        cate = max_arg.item()
        if cate == 0:
            continue
        cates.append(cate)
        soft = torch.softmax(cls_out, -1)
        scores.append(soft[cate])
        bbox_param = rcnn_reg_out[i][cate:cate+4]
        bboxes.append(region.param2xywh(bbox_param, prop['adj_bbox']))
    print('pos bboxes:', len(bboxes))
    bboxes_xyxy = [region.BBox(xywh=bbox).get_xyxy() for bbox in bboxes]
    keep = torchvision.ops.nms(torch.tensor(bboxes_xyxy), torch.tensor(scores), nms_iou)
    print('after NMS:', len(keep))
    print('keep:')
    ret_bboxes = [bboxes[k] for k in keep]
    ret_scores = [scores[k] for k in keep]
    ret_cates  = [cates[k] for k in keep]
    ious = []
    for i in range(len(ret_bboxes)):
        for j in range(i+1, len(ret_bboxes)):
            ious.append(region.calc_iou(region.BBox(xywh=ret_bboxes[i]),
                                        region.BBox(xywh=ret_bboxes[j])))
    print(max(ious), min(ious))
    return ret_bboxes, ret_scores, ret_cates

class FasterRCNNTest(object):
    r"""
    Utility to test a faster rcnn
    """
    def __init__(self, work_dir, faster_configs, dataloader, device):
        self.work_dir = work_dir
        self.device = device
        self.faster_configs = faster_configs
        self.faster_rcnn = FasterRCNNModule(**faster_configs)
        self.dataloader = dataloader


    def load_epoch(self, epoch):
        # TODO
        saved = osp.join(self.work_dir, ckpt_name(epoch))
        self.faster_rcnn.load_state_dict(torch.load(saved))
        self.faster_rcnn.to(self.device)

    def inference(self, img):
        self.faster_rcnn.eval_mode()
        trans = self.dataloader.dataset.transform
        img = Image.open(img)
        img_data = trans(img).unsqueeze(0)
        img_data = img_data.to(self.device)
        rpn_cls_out, rpn_reg_out, rcnn_cls_out, rcnn_reg_out,\
            anchor_targets, props, props_targets = self.faster_rcnn(img_data, None)
        print('rpn_cls_out.shape:', rpn_cls_out.shape)
        print('rpn_reg_out.shape:', rpn_reg_out.shape)
        print('rcnn_cls_out.shape:', rcnn_cls_out.shape)
        print('rcnn_reg_out.shape:', rcnn_reg_out.shape)
        print('props:', len(props))
        print('props[0]:', props[0])
        print('Next interpret the rcnn results')
        bboxes, scores, cates = interpret_rcnn_output(rcnn_cls_out,
                                                      rcnn_reg_out, props,
                                                      self.faster_rcnn.test_props_nms_iou)
        resize_rat = img_data.shape[2] / img.height
        print('resize rat:', resize_rat)
        bboxes = [[i/resize_rat for i in bbox] for bbox in bboxes]
        for i, bbox in enumerate(bboxes):
            print(bbox, scores[i], cates[i])
        
        
    pass
