import torch, logging
from . import utils

def _bbox_target(props_bbox, gt_bbox, gt_label, assigner, sampler, target_means=None, target_stds=None):
    from .registry import build_module
    device = props_bbox.device
    if isinstance(assigner, dict):
        assigner = build_module(assigner)
    if isinstance(sampler, dict):
        sampler = build_module(sampler)
    gt_bbox = gt_bbox.to(props_bbox.dtype)
    props_bbox = torch.cat([gt_bbox, props_bbox], dim=1)
    logging.debug('props_bbox after adding GT: {}'.format(props_bbox.shape))
    
    labels, overlaps_ious = assigner(props_bbox, gt_bbox)
    logging.debug('labels after assigner: -1:{}, 0:{}, >0:{}, >=0: {}'\
                  .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum(), (labels>=0).sum()))
    labels = sampler(labels)
    logging.debug('labels after sampler: -1:{}, 0:{}, >0:{}, >=0: {}'\
                  .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum(), (labels>=0).sum()))
    pos_places = (labels > 0)
    neg_places = (labels == 0)
    chosen_places = (labels>=0)

    # find out where in the propo_bbox gts are                                                                         
    n_props_bbox = props_bbox.shape[1]
    n_gts = gt_label.numel()
    is_gt = labels.new_zeros(n_props_bbox)
    is_gt[:n_gts]=1
    is_gt_chosen=is_gt[chosen_places]
    logging.debug('chosen gt: {}, number of gt: {}'.format(is_gt_chosen.sum(), n_gts))
    
    # check IoU of assigned pos props                                                                                  
    pos_iou = overlaps_ious[pos_places]
    logging.debug('IoU of positively assigned props: n={}, avg={}'\
                  .format(len(pos_iou), pos_iou.mean()))
    
    labels = labels - 1
    labels[labels<0] = 0
    label_bboxes = gt_bbox[:, labels]
    label_cls = gt_label[labels]
    # it is very important to set neg places to 0 as 0 means background                                                
    label_cls[neg_places] = 0
    
    tar_is_gt = is_gt_chosen
    tar_props = props_bbox[:, chosen_places]
    tar_label = label_cls[chosen_places] # class of each gt label, 0 means background                                  
    # logging.debug('class of each target label, 0 means background')                                                  
    # logging.debug('{}'.format(tar_label))                                                                            
    tar_bbox = label_bboxes[:, chosen_places]
    # calc target param which reg_out regress to                                                                       
    tar_param = utils.bbox2param(tar_props, tar_bbox)
    
    # for debug                                                                                                        
    pos_tar_param = tar_param[:, (tar_label>0)]
    logging.debug('mean of pos_tar_param of RCNN: {}'.format(pos_tar_param.mean(dim=1)))
    logging.debug('std  of pos_tar_param of RCNN: {}'.format(pos_tar_param.std(dim=1)))
    # for debug                                                                                                        
    if target_means is not None and target_stds is not None:
        param_mean = tar_param.new(target_means).view(4, 1)
        param_std  = tar_param.new(target_stds).view(4, 1)
        tar_param = (tar_param - param_mean) / param_std # normalize the regression values
    return tar_props, tar_bbox, tar_label, tar_param, tar_is_gt
    
def bbox_target(props, gt_bbox, gt_label, assigner, sampler, target_means=None, target_stds=None):
    with torch.no_grad():
        return _bbox_target(props, gt_bbox, gt_label, assigner, sampler, target_means, target_stds)


