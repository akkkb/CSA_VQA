import torch
import torch.nn as nn
from Attention import CSCA
from text_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

## Attended Feature Fxtraction, Fusion and Classification

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_net, mm_emb, num_hid):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_net = v_net
        self.mm_emb = mm_emb
        self.layer_norm = nn.LayerNorm(int_hid)
        
    def forward(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        v_emb = self.v_net(v)
        v_repr, q_repr = self.mm_emb(v_emb, q_emb, num_hid, batchsize, nhead)
        uemb = torch.sum(v_repr, dim = 1) + torch.sum(q_repr, dim = 1)
        joint_feat = torch.sigmoid(self.classifier(uemb))
        return joint_feat

def model(dataset, num_hid, int_hid, nhead, nblock batch_size):  
        w_emb = WordEmbedding(dataset, 300, 0.0)
        q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        v_net = FCNet([dataset.v_dim, num_hid])
        mm_emb = CSCA(num_hid, nblock, batchsize, nhead)
        classifier = SimpleClassifier(num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
        return BaseModel(w_emb, q_emb, v_net, mm_emb, classifier)
