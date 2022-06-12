import torch

pdtb_data = torch.load('data/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding.pt')
dev_X,dev_Y,train_X,train_Y,test_X,test_Y = pdtb_data['dev_X'],pdtb_data['dev_Y'],pdtb_data['train_X'] ,pdtb_data['train_Y'],pdtb_data['test_X'],pdtb_data['test_Y']

pdtb_data_t5 = torch.load('data/t5_embedding_large.pt')
dev_X_t5,dev_Y_t5,train_X_t5,train_Y_t5,test_X_t5,test_Y_t5 = \
    pdtb_data_t5['dev_X'],pdtb_data_t5['dev_Y'],pdtb_data_t5['train_X'] ,pdtb_data_t5['train_Y'],pdtb_data_t5['test_X'],pdtb_data_t5['test_Y']

def compare_X(dev_X,dev_X_t5):
    dev_X_data = dev_X[0]
    dev_X_label_length_list = dev_X[1]
    dev_X_eos_list = dev_X[2]

    dev_X_data_t5 = dev_X_t5[0]
    dev_X_label_length_list_t5 = dev_X_t5[1]
    dev_X_eos_list_t5 = dev_X_t5[2]

    assert len(dev_X_data) == len(dev_X_data_t5)

    assert len(dev_X_label_length_list) == len(dev_X_label_length_list_t5)
    for i in range(len(dev_X_label_length_list)):
        assert dev_X_label_length_list[i] == dev_X_label_length_list_t5[i] # length should be the same

    assert len(dev_X_eos_list) == len(dev_X_eos_list_t5)
    for i in range(len(dev_X_eos_list)):
        assert len(dev_X_eos_list[i]) == len(dev_X_eos_list_t5[i]) # num of eos should be the same
        assert dev_X_eos_list[i][-1] == dev_X_data[i].shape[1] # last eos should be the data length
        assert dev_X_eos_list_t5[i][-1] == dev_X_data_t5[i].shape[1] # last eos should be the data length


def compare_Y(dev_Y,dev_Y_t5):
    assert len(dev_Y) == len(dev_Y_t5)
    for i in range(len(dev_Y)):
        assert dev_Y[i].equal(dev_Y_t5[i])

compare_Y(dev_Y,dev_Y_t5)
compare_Y(train_Y,train_Y_t5)
compare_Y(test_Y,test_Y_t5)

compare_X(dev_X,dev_X_t5)
compare_X(train_X,train_X_t5)
compare_X(test_X,test_X_t5)

print('Comparison finish!')
pass