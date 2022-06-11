import os
import os.path as osp

import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.tensorboard import SummaryWriter  

'''
the structure of model
the args parser config
tensorboard log
traing, validating and testing log
best validating result and corresponding testing result

specified by different seed
'''

def get_suffix_without_seed(args):
    suffix = 'm'+str(args.pmodel)+ '.' +args.encoder + '.' + str(args.drop)
    if args.pmodel in [3,4]:
        suffix = suffix + '.' + args.nts + '.' + args.nah + '.' + args.dropt + '.' + args.intt
    elif args.pmodel in [5]:
        suffix = suffix + '.' + args.nts + '.' + args.docsa + '.' + args.fuse + '.' + str(args.decoder)
        if args.decoder == 'rnn':
            if args.att_level == 'none':
                rnn_tag = ''
            elif args.att_level == 'sentence':
                rnn_tag = 'lvs.' 
                order_tag = 'l.' if args.att_inter_order == 'linear_first' else 'i.'
                rnn_tag = rnn_tag + order_tag + args.att_inter_type
            elif args.att_level == 'word_sentence':
                rnn_tag = 'lvws.' + args.att_inter_structure + '.' + args.att_inter_type + '.' \
                            + args.att_inter_ws + '.' +str(args.att_pred_cat_x)
            suffix = suffix + 'rd' + str(args.rd) + '.' + args.yemb + '.' + args.loss + '.' + rnn_tag
    
    if args.few_data > 0:
        suffix = suffix + 'few.' + str(args.few_data)
        
    suffix = suffix + args.special_tag

    if args.special_tag == 'no_record':
        suffix = 'no_record'

    return suffix

class SelfLogger(object):
    def __init__(self, name:str):
        super(SelfLogger, self).__init__()
        self.model_dir_ori = osp.join('logs',name)

        # used as the directory of this model
        # if exist, can mkdir directly
        if not osp.exists(self.model_dir_ori):
            os.makedirs(self.model_dir_ori)

        # used as tensorboard file name
        self.tensorboard_file = None
        self.tb_writer = None

        self.model_config_path = osp.join(self.model_dir_ori,'model_config.txt')
        self.training_logs_path = osp.join(self.model_dir_ori,'traing_logs.txt')

        print('Logger init completed')


    def write_model(self, model):
        if not osp.exists(self.model_config_path):
            with open(self.model_config_path,'w') as f:
                f.write(str(model))

    def write_traing_log(self, content):
        print(content)
        with open(self.training_logs_path,'a') as f:
            f.write(str(content)+'\n')

    def change_iter_num(self, num:int):
        self.model_dir = osp.join(self.model_dir_ori,str(num))

        self.tensorboard_file = osp.join(self.model_dir,'tb_vis')
        self.tb_writer = SummaryWriter(self.tensorboard_file)
        self.model_config_path = osp.join(self.model_dir,'model_config.txt')
        self.training_logs_path = osp.join(self.model_dir,'traing_logs.txt')


class BestHolder(object):
    def __init__(self, ):
        super(BestHolder, self).__init__()
        self.macro_f1_value = 0 # valid
        self.y_true_valid = None
        self.y_pred_valid = None

        self.y_true = None 
        self.y_pred = None
        self.weights = None

        self.epo_num = 0

    def update_valid(self, y_true_valid, y_pred_valid, pref):
        self.y_true_valid = y_true_valid
        self.y_pred_valid = y_pred_valid
        self.macro_f1_value = pref
    
    def update_test(self, y_true_test, y_pred_test, weights, epo):
        self.y_true = y_true_test 
        self.y_pred = y_pred_test
        self.weights = weights
        self.epo_num = epo



    

