from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
import os.path as osp
import torch, logging

class Hook(object):
    MAX_PRIORITY = 0
    MIN_PRIORITY = 10
    
    def __init__(self, priority):
        assert priority >= Hook.MAX_PRIORITY and priority <= Hook.MIN_PRIORITY, 'Invalid priority {}'.format(priority)
        self.priority = priority

    def before_train_all(self):
        pass
    def after_train_all(self):
        pass
    def before_iter(self):
        pass
    def after_iter(self):
        pass
    def before_epoch(self):
        pass
    def after_epoch(self):
        pass
    def before_step(self):
        pass
    def after_step(self):
        pass


class Hookable(object):
    def __init__(self):
        self.hooks = [[] for i in range(Hook.MAX_PRIORITY, Hook.MIN_PRIORITY+1)]

    def clear_hooks(self):
        self.__init__()

    def add_hook(self, hook):
        prio = hook.priority
        self.hooks[prio].append(hook)

    def call_hooks(self, action):
        for hk_prio in self.hooks:
            for hk in hk_prio:
                if hasattr(hk, action):
                    getattr(hk, action)()

class OptimizerHook(Hook):
    def __init__(self, trainer, priority=1):
        super(OptimizerHook, self).__init__(priority)
        self.trainer = trainer

    def before_step(self):
        grad_clip = self.trainer.optim_cfg.grad_clip
        params = self.trainer.model.parameters()
        if grad_clip is not None:
            clip_grad_norm_(list(filter(lambda x:x.requires_grad==True, params)), **grad_clip)

class LrHook(Hook):
    def __init__(self, trainer, priority=1):
        super(LrHook, self).__init__(priority)
        self.trainer = trainer
        cfg = self.trainer.lr_cfg
        self.warmup_iters=cfg.warmup_iters
        self.warmup_ratio=cfg.warmup_ratio
        self.lr_decay=cfg.lr_decay

    def before_iter(self):
        cur_iter = self.trainer.cur_iter
        init_lr = self.trainer.initial_lr
        if cur_iter > self.warmup_iters:
            pass
        elif cur_iter == self.warmup_iters:
            self.trainer.set_lr(init_lr)
        else:
            k = (1 - cur_iter / self.warmup_iters) * (1 - self.warmup_ratio)
            self.trainer.set_lr(init_lr * k)


    def before_epoch(self):
        epoch = self.trainer.cur_epoch
        if epoch in self.lr_decay:
            decay = self.lr_decay[epoch]
            lr = self.trainer.get_lr()
            lr = [x*decay for x in lr]
            self.trainer.set_lr(lr)


class CkptHook(Hook):
    def __init__(self, trainer, priority=1):
        super(CkptHook, self).__init__(priority)
        self.trainer = trainer
        self.ckpt_cfg = trainer.ckpt_cfg
    def after_epoch(self):
        epoch = self.trainer.cur_epoch
        if epoch % self.ckpt_cfg.interval == 0:
            epoch_model = osp.join(self.trainer.work_dir, 'epoch_{}.pth'.format(epoch))
            torch.save(self.trainer.model.state_dict(), epoch_model)
            logging.info('Finished training epoch {}, saved trained model to {}'.format(epoch, epoch_model))

# print loss, eta infomation on screen
import datetime
class ReportHook(Hook):
    def __init__(self, trainer, priority=1):
        super(ReportHook, self).__init__(priority)
        self.trainer = trainer
        self.report_cfg = trainer.report_cfg
        self.tic = None
        self.loss_tracker = []

    def _simple_time(self, t):
        t = str(t)
        return t[:t.rfind('.')]

    def report(self, avg_loss, remain_time, now):
        loss_str = ', '.join(['{}: {}'.format(k, v) for k, v in avg_loss.items()])
        print(self._simple_time(now)+': ' + ', '.join([
            loss_str,
            'ETA: ' + self._simple_time(remain_time)
        ]))
        pass

    def get_avg_loss(self):
        if len(self.loss_tracker) == 0:
            return {}
        loss_sum = {k:0 for k in self.loss_tracker[0].keys()}
        loss_sum['loss'] = 0
        for loss in self.loss_tracker:
            tot_loss = 0
            for k, v in loss.items():
                loss_sum[k]+=v
                tot_loss += v
            loss_sum['loss'] += tot_loss
        
        return OrderedDict({k:round(v/len(self.loss_tracker), 3) for k, v in loss_sum.items()})


    def after_iter(self):
        if self.trainer.cur_loss is not None:
            self.loss_tracker.append(self.trainer.cur_loss)
        if self.tic is None:
            self.tic = datetime.datetime.now()
        cur_iter = self.trainer.cur_iter
        if cur_iter % self.report_cfg.interval == 0:
            now = datetime.datetime.now()
            time_elapse = now - self.tic
            tot_iters = self.trainer.get_total_iters()
            remain_iters = tot_iters - cur_iter
            remain_time = time_elapse * (remain_iters / self.report_cfg.interval)
            self.report(self.get_avg_loss(), remain_time, now)
            self.tic=now
            self.loss_tracker = []

