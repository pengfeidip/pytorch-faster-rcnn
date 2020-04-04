from . import utils
import os.path as osp
import copy, torch, logging

class BasicTester(object):
    def __init__(self,
                 model,
                 train_cfg,
                 test_cfg,
                 device):
        self.device=device
        self.model=model
        self.train_cfg=copy.deepcopy(train_cfg)
        self.test_cfg=copy.deepcopy(test_cfg)

    def load_ckpt(self, ckpt):
        self.ckpt=ckpt
        assert self.model is not None
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        logging.info('loaded ckpt: {}'.format(ckpt))

    def inference(self, dataloader):
        self.model.eval()
        inf_res, ith = [], 0
        logging.info('Start to inference {} images...'.format(len(dataloader)))
        logging.info('Test config:')
        logging.info(str(self.test_cfg))
        with torch.no_grad():
            for test_data in dataloader:
                img_meta = test_data['img_meta'].data[0][0]
                img_data = test_data['img'].data[0]
                ori_size = img_meta['ori_shape'][:2]
                scale = img_meta['scale_factor']
                
                filename = img_meta['filename']
                filename = osp.basename(filename)
                iid = int(filename[:-4])
                logging.info('Inference {}-th image with image id: {}'.format(ith, iid))
                img_w, img_h=ori_size
                img_res={'width':img_w, 'height': img_h, 'image_id':iid, 'file_name': filename}
                bbox, score, category=self.inference_one(img_data, img_meta)
                if len(bbox)==0:
                    logging.warning('0 predictions for image {}'.format(iid))
                    continue
                img_res['bbox']=utils.xyxy2xywh(bbox).t()/scale
                img_res['score']=score
                img_res['category']=category
                logging.info('{} bbox predictions for {}-th image with image id {}'.format(bbox.shape[1], ith, iid))
                inf_res.append(img_res)
                ith += 1
            return inf_res

    def inference_one(self, img_data, img_meta):
        img_data = img_data.to(device=self.device)
        return self.model.forward_test(img_data, img_meta)
