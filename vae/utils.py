import torch

def hardmax(logits, dim=-1):
    index = logits.argmax(dim=dim, keepdim=True)
    y_soft = torch.nn.functional.softmax(logits, dim=dim)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft

def hardmax_bernoulli(logits, threshold=0.5):
    y_soft = torch.sigmoid(logits)
    y_hard = torch.zeros_like(logits)
    y_hard[y_soft > threshold] = 1.
    return y_hard - y_soft.detach() + y_soft

def ent_categorical(logits, dim=-1):
    prob = torch.nn.functional.softmax(logits, dim=dim)
    log_prob = torch.nn.functional.log_softmax(logits, dim=dim)
    ent = - torch.sum(prob * log_prob, dim=dim)
    return ent

def ent_bernoulli(logits):
    prob = torch.sigmoid(logits)
    ent = - (prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob))
    return ent