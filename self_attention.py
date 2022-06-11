from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mid_dim):
        super(SelfAttention, self).__init__()

        self.ws1 = nn.Linear(hidden_size, mid_dim)
        self.ws2 = nn.Linear(mid_dim, 1)

        self.drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_tensor, word_mask):
        '''
        typically input should be: batch_size, edu_size, word_in_edu, hidden_size
        '''
        batch_size, edu_size, word_in_edu, hidden_size = input_tensor.shape # hidden_size 400
        input_tensor = input_tensor.view(batch_size * edu_size, word_in_edu, hidden_size)
        word_mask = word_mask.view(batch_size * edu_size, word_in_edu)

        self_attention = F.tanh(self.ws1(self.drop(input_tensor)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze()
        self_attention = self_attention + -10000*(word_mask == 0).float()
        self_attention = self.softmax(self_attention)
        weighted_embedding = torch.sum(input_tensor*self_attention.unsqueeze(-1), dim=1)

        return weighted_embedding


class DocSelfAttention(nn.Module):
    def __init__(self, hidden_size, mid_dim):
        super(DocSelfAttention, self).__init__()

        '''
        self attention should be done after word and syntax are concate together
        '''

        self.ws1 = nn.Linear(hidden_size, mid_dim)
        self.ws2 = nn.Linear(mid_dim, 1)
        self.ws3 = nn.Linear(hidden_size, mid_dim)

        self.drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_all, word_weighted):

        # word_all : bs, word_all_num, dim
        # word_weighted: bs, edu_size, dim

        batch_size, word_all_num, word_dim = word_all.shape 
        batch_size, edu_size, word_dim = word_weighted.shape 

        out_holder = torch.zeros_like(word_weighted).to(device)

        for i in range(batch_size):
            # 1, 1, word_all_num, dim - 1, edu_size, 1, dim
            word_all_ = word_all[i]
            word_weighted_ = word_weighted[i]
            self_attention = word_all_.unsqueeze(0) - word_weighted_.unsqueeze(1)
            self_attention = F.tanh(self.ws1(self.drop(self_attention)))
            self_attention = self.ws2(self.drop(self_attention)).squeeze()
            self_attention = self.softmax(self_attention)
            self_attention = torch.sum(word_all_*self_attention.unsqueeze(-1), dim=1)
            out_holder[i] = self_attention

        out_holder = out_holder + word_weighted
        out_holder = self.ws3(out_holder)

        # bs, edu_size, dim
        return out_holder

if __name__ == '__main__':
    sa = SelfAttention(20,20)
    input_t = torch.rand((2,4,5,20))