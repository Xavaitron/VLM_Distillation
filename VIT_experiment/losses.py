import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def kd_loss(student_logits, teacher_logits, temperature=1.0):
    loss_kl = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    return loss_kl

def adaad_inner_loss(model, teacher_model, x_natural, step_size=2/255.0, steps=10, epsilon=8/255.0):
    criterion_kl = nn.KLDivLoss(reduction='none')
    model.eval()
    teacher_model.eval()

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()

    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv

def igdm_inner_loss(model, teacher_model, x_natural, step_size=2/255.0, steps=10, epsilon=8/255.0):
    # This matches the adaad_inner_loss5 from the ICLR_IGDM codebase
    criterion_kl = nn.KLDivLoss(reduction='none')
    model.eval()
    teacher_model.eval()

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()

    for _ in range(steps):
        delta = x_adv - x_natural
        x_adv.requires_grad_()
        with torch.enable_grad():
            teacher_minus = teacher_model(x_adv - 2 * delta)
            student_minus = model(x_adv - 2 * delta)

            student_plus = model(x_adv)
            teacher_plus = teacher_model(x_adv)
            
            loss_kl = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), 
                                   F.softmax(teacher_plus - teacher_minus, dim=1))
            loss_kl = torch.sum(loss_kl)
            
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv
