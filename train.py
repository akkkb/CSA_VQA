import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.autograd import Variable
import _pickle as cPickle
import numpy as np
import statistics

## Score Computation Function
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

## Categorywise Score Computation Function
def compute_score_with_logits_inclass(logits, labels, type_, m):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    match = (type_ == m)
    match = match.cuda()
    scores = scores[match,:]
    labels = labels[match,:]
    return scores.sum(), labels.sum()

def init_dict(names_list):
    out_dict = {}
    for nm in names_list:
        out_dict[nm] = [0,0]
    return out_dict

## Model Training Module
def train(model, train_loader, eval_loader, num_epochs, output):

    question_type= ['absurd','activity_recognition','attribute','color','counting','object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']

    utils.create_dir(output)
    params = list(model.parameters())

    filename = 'result'
    optim = torch.optim.Adamax(params)
    logger = utils.Logger(os.path.join(output, '%s.txt' %(filename)))
    BCEL = torch.nn.BCELoss(reduction='sum').cuda()

    for epoch in range(num_epochs):
        t = time.time()
        print("\n-----------------------------> Epoch is %d <------------------------------" %(epoch))
        train_score = init_dict(question_type)
        train_score_ques = 0
        total_loss = 0

        for i, (v, q, a, type_) in enumerate(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, q)      ## Prediction
            loss = BCEL(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()
            print("\r[epoch %2d][step %4d/%4d] loss: %.4f" % ( epoch + 1, i, int( len(train_loader.dataset) / a.shape[0]), loss.cpu().data.numpy()), end='          ')

            batch_score = compute_score_with_logits(pred, a).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score
  
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        model.train(False)
        eval_score, eval_ls = evaluate(model, eval_loader)
        model.train(True)

        ## Categorywise Accuracy
        for qtp in question_type:
                count = eval_ls[qtp][0]/eval_ls[qtp][1]
                logger.write('\t%s is: \t\t %d / %d  -----> %.2f'%(qtp, eval_ls[qtp][0], eval_ls[qtp][1], count*100))

        logger.write('\n\teval score: \t\t ----->  %.2f' % (eval_score * 100))

        ## Best Model Save
        if eval_score > best_eval_score:
                epc = epoch
                model_path = os.path.join(output, '%s_model.pth' %(filename))
                torch.save(img_att, model_path)
                best_eval_score = eval_score

    logger.write('\n\t Best Evaluation score is obtained at epoch %d : %.2f' %(epc,best_eval_score * 100))

## Model Evaluation Code
def evaluate(ques_cat, ques_att, img_att, Model,dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    question_type = ['absurd','activity_recognition','attribute','color','counting', 'object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']

    val_score = init_dict(question_type)

    for v, q, a, type_ in iter(dataloader):
        with torch.no_grad():
            v = v.cuda()
            a = a.cuda()
            q = q.cuda()
 
        pred = model(v, q)
        batch_score = compute_score_with_logits(pred, a).sum()

        for m in range(len(qtype)):
               bs_p, bs_a = compute_score_with_logits_inclass(pred, a, type, m)
               val_score[qtype[m]][0] += bs_p
               val_score[qtype[m]][0] += bs_a

    num = 0
    den = 0

    ## Categorwise Questions Evaluation
    for itm in val_score:
        num += val_score[itm][0]
        den += val_score[itm][1]

    score = num / den
    return score, val_score
