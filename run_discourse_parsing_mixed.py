import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

import random
from nbformat import write

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_mixed import BaseSequenceLabelingSplitImpExp
from model_mixed import BaseSequenceLabelingSplitImpExp_SA 

from sklearn import metrics
import numpy as np
# import cPickle
import copy

import warnings
warnings.filterwarnings('ignore')

from temp_config import temp_config_model_name, temp_config_data_path, temp_config_logger

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

import time
import math

from logging_utils import SelfLogger
logger = SelfLogger(temp_config_logger)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


use_cuda = torch.cuda.is_available()
if use_cuda:
    logger.write_traing_log("Using GPU!")
else:
    logger.write_traing_log("Using CPU!")

def feed_data_cuda(data):
    if use_cuda:
        for X in data:
            for i in range(len(X)):
                X[i] = X[i].cuda()

def apply_weighted_class(Y):
    logger.write_traing_log('Weighting class...')

    weights = Variable(torch.zeros((Y[0].size(-1))), requires_grad = False)
    for y in Y:
        weights = weights + torch.sum(y.abs(),0)

    weights = (1 / weights)

    for y in Y:
        for j in range(y.size(0)):
            y[j,:] = y[j,:] * weights


def load_data(weighted_class = False):
    logger.write_traing_log('Loading Data...')
    # outfile = open(os.path.join(os.getcwd(),'data/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding.pt'),'r')
    pdtb_data = torch.load(temp_config_data_path)
    # (All the Words/POS/NER/label(both implict&explicit) and discourse unit (DU) boundary information are already transformed to Pytorch vector format)
    # outfile.close()

    dev_X,dev_Y,train_X,train_Y,test_X,test_Y = pdtb_data['dev_X'],pdtb_data['dev_Y'],pdtb_data['train_X'] ,pdtb_data['train_Y'],pdtb_data['test_X'],pdtb_data['test_Y']

    dev_X_eos_list = dev_X[2]
    dev_X_label_length_list = dev_X[1]
    dev_X = dev_X[0]

    train_X_eos_list = train_X[2]
    train_X_label_length_list = train_X[1]
    train_X = train_X[0]

    test_X_eos_list = test_X[2]
    test_X_label_length_list = test_X[1]
    test_X = test_X[0]

    if weighted_class:
        apply_weighted_class(train_Y)
        apply_weighted_class(dev_Y)
        apply_weighted_class(test_Y)

    feed_data_cuda([train_X,train_Y,test_X,test_Y])

    return dev_X,dev_X_label_length_list,dev_X_eos_list,dev_Y,train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y

#epsilon = 1.0e-6
def calculate_loss(predict,target,criterion,alpha = 1):
    #predict = torch.clamp(predict, min=np.log(epsilon), max=np.log(1-epsilon))
    if alpha == 1:
        return -torch.dot(predict.view(-1),target.abs().view(-1))
    elif alpha == 0:
        return -torch.dot(predict.view(-1),target.clamp(0,1).view(-1))
    elif alpha >= 100:
        return -torch.dot(predict.view(-1),target.clamp(-1,0).abs().view(-1))
    else:
        return -torch.dot(predict.view(-1),target.clamp(0,1).view(-1))- alpha*torch.dot(predict.view(-1),target.clamp(-1,0).abs().view(-1))


def train(model,X,Y,optimizer,criterion,alpha = 1):
    optimizer.zero_grad()
    loss = 0

    X_eos_list = X[1]
    X = X[0]

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]

        output = model(sample, sample_eos_list, target)
        loss += calculate_loss(output,target,criterion, alpha = alpha)

    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    total_norm = nn.utils.clip_grad_norm(model.parameters(), 5.0)

    optimizer.step()
    return loss.data 

def process_label(predict_Y,target_Y):
    assert predict_Y.shape == target_Y.shape
    list_1 = []
    list_2 = []

    for i in range(target_Y.shape[0]):
        real_label = target_Y[i,:]
        predict_label = predict_Y[i,:]

        #handle non-label case
        if np.sum(real_label) <= 0:
            continue

        #handle multilabel case
        if np.sum(real_label) >= 2:
            # predict one of the correct label
            if real_label[np.argmax(predict_label)] > 0:
                real_label = np.zeros(target_Y.shape[1])
                real_label[np.argmax(predict_label)] = 1

        list_1.append(real_label)
        list_2.append(predict_label)

    if len(list_1) > 0:
        real_Y = np.stack(list_1)
        predict_Y = np.stack(list_2)

        return predict_Y,real_Y
    else:
        return None,None

def print_evaluation_result(result):
    predict_Y,target_Y,loss = result[0],result[1],result[2]

    logger.write_traing_log('Confusion Metric')
    logger.write_traing_log(str(metrics.confusion_matrix(target_Y,predict_Y)))
    logger.write_traing_log('Accuracy')
    logger.write_traing_log(str(metrics.accuracy_score(target_Y, predict_Y)))
    logger.write_traing_log('loss')
    logger.write_traing_log(str(loss))
    logger.write_traing_log('Micro Precision/Recall/F-score')
    logger.write_traing_log(str(metrics.precision_recall_fscore_support(target_Y, predict_Y, average='micro') ))
    logger.write_traing_log('Macro Precision/Recall/F-score')
    logger.write_traing_log(str(metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro')) )
    logger.write_traing_log('Each-class Precision/Recall/F-score')
    logger.write_traing_log(str(metrics.precision_recall_fscore_support(target_Y, predict_Y, average=None) ))

    return metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro')[2], result

def evaluate(model,X,Y, criterion, epoch, alpha, discourse = 'implicit'):
    model.eval()
    X_eos_list = X[1]
    X = X[0]

    predict_Y_list = []
    target_Y_list = []
    loss = 0
    loss_ = 0

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]

        predict = model(sample, sample_eos_list, target)
        loss_ += calculate_loss(predict,target,criterion, alpha = alpha)

        if discourse == 'all':
            target = target.abs()
        elif discourse == 'explicit':
            target = -target

        if use_cuda:
            predict = predict.cpu()
            target = target.cpu()

        predict = predict.data.numpy()
        target = target.data.numpy()
        
        predict,target = process_label(predict,target)

        if target is not None:
            loss += -np.sum(predict*target)

            predict = np.argmax(predict,axis = 1)
            target = np.argmax(target,axis = 1)

            predict_Y_list.append(predict)
            target_Y_list.append(target)

    logger.tb_writer.add_scalar('2_test_valid_loss/'+ discourse +'loss',loss_,epoch)
    logger.tb_writer.add_scalar('3_test_valid_loss/'+ discourse +'loss',loss,epoch)

    predict_Y = np.concatenate(predict_Y_list,axis=0)
    target_Y = np.concatenate(target_Y_list,axis=0)
    model.train()

    return print_evaluation_result((predict_Y,target_Y,loss))

def average_result(each_iteration_result_list):
    all_result_list = []
    explicit_result_list = []
    implicit_result_list = []

    for each_iteration_result in each_iteration_result_list:
        all_result, explicit_result, implicit_result = each_iteration_result[0],each_iteration_result[1],each_iteration_result[2]

        all_result_list.append(all_result)
        explicit_result_list.append(explicit_result)
        implicit_result_list.append(implicit_result)

    def average_result_list(result_list):
        predict_Y_list = []
        target_Y_list = []
        loss = 0
        for result in result_list:
            predict_Y_list.append(result[0])
            target_Y_list.append(result[1])
            loss += result[2]
        predict_Y = np.concatenate(predict_Y_list,axis=0)
        target_Y = np.concatenate(target_Y_list,axis=0)
        return (predict_Y,target_Y,loss)
    
    return (average_result_list(all_result_list),average_result_list(explicit_result_list),average_result_list(implicit_result_list))

def trainEpochs(model, X, Y, valid_X, valid_Y, batch_size, n_epochs, print_every=1, evaluate_every = 1, optimizer_type = 'adam', weight_decay = 0, alpha = 1, use_scheduler = False):
    if optimizer_type == 'adadelta': 
        optimizer = optim.Adadelta(model.parameters(), lr = 0.5, weight_decay = weight_decay)  #baseseq2seq lr = 0.5
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr = 0.005, weight_decay = weight_decay)  #baseseq2seq lr = 0.005
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(),lr= 0.0005, weight_decay = weight_decay) #baseseq2seq lr= 0.0002
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = weight_decay)
    else:
        logger.write_traing_log("optimizer not recommend for the task!")
        sys.exit()

    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience = 10)

    criterion = nn.NLLLoss()

    X_label_length_list = X[1]
    X_eos_list = X[2]
    X = X[0]

    start = time.time()
    random_list = [i for i in range(len(Y))]
    print_loss_total = 0  # Reset every print_every
    best_macro_Fscore = -1
    best_result = None
    best_model = None
    logger.write_traing_log('----------------------------------------------------')
    logger.write_traing_log('Training start: ' + '#training_samples = ' + str(len(Y)))
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        logger.write_traing_log('epoch ' + str(epoch) + '/' + str(n_epochs))
        random.shuffle(random_list)

        i = 0
        target_length = 0
        batch = [] 
        while i < len(random_list):
            batch.append(random_list[i])
            target_length += X_label_length_list[random_list[i]]
            i = i + 1

            if target_length >= batch_size or i >= len(random_list):
                batch_X_eos_list = []
                batch_X = []
                batch_Y = []

                for index in batch:
                    batch_X.append(X[index])
                    batch_X_eos_list.append(X_eos_list[index])
                    batch_Y.append(Y[index])

                loss = train(model, (batch_X, batch_X_eos_list), batch_Y, optimizer, criterion, alpha = alpha)
                print_loss_total += loss

                target_length = 0
                batch = []

        with open('record_time.txt','a') as f:
            f.write('training:'+str(time.time()-start_time)+'\n')

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logger.write_traing_log('%s (%d %d%%) %.4f' % (timeSince(start, epoch*1.0/ n_epochs), epoch, epoch*1.0 / n_epochs * 100, print_loss_avg))
            logger.tb_writer.add_scalar('1_training_loss/loss',print_loss_avg,epoch)

        if epoch % evaluate_every == 0:
            logger.write_traing_log('----------------------------------------------------')
            logger.write_traing_log('Step Evaluation: #valid_samples= ' + str(len(valid_Y)))
            logger.write_traing_log('Evaluate on Explicit/Implicit discourse relation')
            logger.write_traing_log( '----------------------------------------------------')
            _, tmp_all_result = evaluate(model,valid_X,valid_Y, criterion, epoch, alpha = alpha, discourse = 'all')

            logger.write_traing_log( 'Evaluate on Explicit discourse relation')
            logger.write_traing_log( '----------------------------------------------------')
            _, tmp_explicit_result = evaluate(model,valid_X,valid_Y,  criterion, epoch, alpha = alpha, discourse = 'explicit')

            logger.write_traing_log( 'Evaluate on Implicit discourse relation')
            logger.write_traing_log( '----------------------------------------------------')
            start_time_ = time.time()
            tmp_macro_Fscore, tmp_implicit_result = evaluate(model,valid_X,valid_Y, criterion, epoch, alpha = alpha,discourse = 'implicit')
            with open('record_time.txt','a') as f:
                f.write('testing:'+str(time.time()-start_time_)+'\n')
            if tmp_macro_Fscore > best_macro_Fscore:
                best_macro_Fscore = tmp_macro_Fscore
                best_result = (tmp_all_result, tmp_explicit_result, tmp_implicit_result)
                best_model = copy.deepcopy(model)

            if use_scheduler:
                scheduler.step(tmp_macro_Fscore)

    logger.write_traing_log( 'Training completed!')
    return best_macro_Fscore,best_result,best_model

batch_size_list = [128]  # fixed 128 > 64 > 256
hidden_size_list = [300] # 600>300>100
dropout_list = [5]
l2_reg_list = [0]   # fixed 0
nb_epoch_list = [100]
encoder_sentence_embedding_type_list = ['max'] # max > mean > last
sentence_zero_inithidden_list = [False]
optimizer_type_list = ['adam']  # adam > adagrad > other
num_layers_list = [1]
parameters_list = []
for num_layers in num_layers_list:
    for sentence_embedding_type in encoder_sentence_embedding_type_list:
        for sentence_zero_inithidden in sentence_zero_inithidden_list:
            for batch_size in batch_size_list:
                for optimizer_type in optimizer_type_list:
                    for hidden_size in hidden_size_list:
                        for nb_epoch in nb_epoch_list:
                                for weight_decay in l2_reg_list:
                                    for dropout in dropout_list:
                                        parameters = {}
                                        parameters['nb_epoch'] = nb_epoch
                                        parameters['sentence_embedding_type'] = sentence_embedding_type
                                        parameters['sentence_zero_inithidden']= sentence_zero_inithidden
                                        parameters['num_layers'] = num_layers
                                        parameters['batch_size'] = batch_size
                                        parameters['hidden_size'] = hidden_size
                                        parameters['optimizer_type'] = optimizer_type
                                        parameters['dropout'] = dropout * 0.1
                                        parameters['weight_decay'] = weight_decay
                                        parameters_list.append(parameters)

if __name__ == "__main__":
    dev_X,_,dev_X_eos_list,dev_Y,train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,_,test_X_eos_list,test_Y = load_data(weighted_class= False)
    word_embedding_dimension = test_X[0].size(-1)
    number_class = test_Y[0].size(-1)

    macro_F1_list = []
    for parameters in parameters_list:
        overall_best_result = None
        overall_best_model = None
        overall_best_macro = -1
        each_iteration_result_list = []
        each_iteration_best_model_list = [] 
        each_iteration_macro_Fscore_list = []

        for iteration in range(5):
            logger.change_iter_num(iteration)
            if temp_config_model_name == 'ori':
                model = BaseSequenceLabelingSplitImpExp(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                    sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'feedforward', NTN_flag = False, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
            elif temp_config_model_name == 'sa':
                model = BaseSequenceLabelingSplitImpExp_SA(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                    sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'feedforward', NTN_flag = False, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
            logger.write_model(model)
            #model = BaseSequenceLabelingSplitImpExp_LSTMEncoder(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
            #        sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'dot', NTN_flag = True, num_layers = parameters['num_layers'], dropout = parameters['dropout'])

            if use_cuda:
                model = model.cuda()
                                                                                                                                           #(dev_X, dev_X_eos_list), dev_Y
            best_macro_Fscore, best_result, best_model = trainEpochs(model, (train_X,train_X_label_length_list,train_X_eos_list), train_Y, (test_X, test_X_eos_list), test_Y, 
                                                    batch_size = parameters['batch_size'], n_epochs = parameters['nb_epoch'], optimizer_type = parameters['optimizer_type'], 
                                                    weight_decay = parameters['weight_decay'], alpha = 1, use_scheduler = False)
            
            logger.write_traing_log( '----------------------------------------------------')
            logger.write_traing_log( 'Experiment Iteration ' +  str(iteration+1) + ' Evaluation: #test_samples= ' + str(len(test_Y)))
            logger.write_traing_log( 'Evaluate on Explicit/Implicit discourse relation')
            logger.write_traing_log( '----------------------------------------------------')
            print_evaluation_result(best_result[0])

            logger.write_traing_log( 'Evaluate on Explicit discourse relation')
            logger.write_traing_log( '----------------------------------------------------')
            print_evaluation_result(best_result[1])

            logger.write_traing_log( 'Evaluate on Implicit discourse relation')
            logger.write_traing_log( '----------------------------------------------------')
            print_evaluation_result(best_result[2])

            each_iteration_result_list.append(best_result)
            each_iteration_best_model_list.append(best_model.state_dict())
            each_iteration_macro_Fscore_list.append(best_macro_Fscore)
            if best_macro_Fscore > overall_best_macro:
                overall_best_macro = best_macro_Fscore
                overall_best_result = best_result
                overall_best_model = best_model

        logger.write_traing_log('--------------------------------------------------------------------------')
        logger.write_traing_log('Overall Best Result:')
        logger.write_traing_log('Evaluate on Explicit/Implicit discourse relation')
        logger.write_traing_log('-------------------------------------------------------------------------')
        print_evaluation_result(overall_best_result[0])

        logger.write_traing_log('Evaluate on Explicit discourse relation')
        logger.write_traing_log('-------------------------------------------------------------------------')
        print_evaluation_result(overall_best_result[1])

        logger.write_traing_log('Evaluate on Implicit discourse relation')
        logger.write_traing_log('-------------------------------------------------------------------------')
        print_evaluation_result(overall_best_result[2])


        overall_average_result = average_result(each_iteration_result_list)
        logger.write_traing_log('-------------------------------------------------------------------------')
        logger.write_traing_log('Overall Average Result:')
        logger.write_traing_log('Evaluate on Explicit/Implicit discourse relation')
        logger.write_traing_log('-------------------------------------------------------------------------')
        print_evaluation_result(overall_average_result[0])

        logger.write_traing_log('Evaluate on Explicit discourse relation')
        logger.write_traing_log('-------------------------------------------------------------------------')
        print_evaluation_result(overall_average_result[1])

        logger.write_traing_log('Evaluate on Implicit discourse relation')
        logger.write_traing_log('-------------------------------------------------------------------------')
        macro_F1_list.append(print_evaluation_result(overall_average_result[2])[0])

        logger.write_traing_log('Implicit discourse relation classification Macro F_score std:' + str(np.std(each_iteration_macro_Fscore_list)))

        logger.write_traing_log(str(parameters))
        sys.stdout.flush()

    logger.write_traing_log( '-------------------------------------------------------------------------')
    for i in range(len(parameters_list)):
        logger.write_traing_log(str(parameters_list[i]) + '       macro F-1:  ' + str(macro_F1_list[i]) + '\n')