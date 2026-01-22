import os
import argparse
import torch
import time
from rslad_loss import *
from cifar100_models import *
import torchvision
from torchvision import datasets, transforms
import time
# we fix the random seed to 0, this method can keep the results consistent in the same conputer. 
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
from robustbench.utils import load_model

#########################################################################################################

from argparse import ArgumentParser
from status import ProgressBar
from args import create_parser
try:
    import wandb
except ImportError:
    wandb = None
from autoattack import AutoAttack


parser = create_parser()
args = parser.parse_known_args()[0]

print(args)

basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not args.nowand:
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    wandb.init(dir="./wandbtmp", project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=args.wandb_name, tags=[args.wandb_tags])
    args.wandb_url = wandb.run.get_url()
    wandb.save(basepath+'/resnet18_'+str(args.method)+'cifar100.py', base_path=basepath)

##########################################################################################################################################


prefix = 'resnet18-CIFAR100_RSLAD'
epochs = args.epochs
batch_size = args.batch
epsilon = 8/255.0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR100(root='../dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


if args.student == "RES-18":
    student = resnet18()
    student = torch.nn.DataParallel(student)
    student = student.cuda()
elif args.student == "MN-V2":
    student = mobilenet_v2()
    student = torch.nn.DataParallel(student)
    student = student.cuda()

student.train()
optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)




if args.teacher == "LTD":
    teacher = load_model(model_name='Chen2021LTD_WRN34_10', dataset='cifar100', threat_model='Linf')
elif args.teacher == "DEC":
    teacher = load_model(model_name='Cui2023Decoupled_WRN-28-10', dataset='cifar100', threat_model='Linf')
elif args.teacher == "BDM":
    teacher = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar100', threat_model='Linf')
elif args.teacher == "Rice2020Overfitting":
    teacher = load_model(model_name='Rice2020Overfitting', dataset='cifar100', threat_model='Linf')
elif args.teacher == "Sitawarin2020Improving":
    teacher = load_model(model_name='Sitawarin2020Improving', dataset='cifar100', threat_model='Linf')
elif args.teacher == "Chen2020Efficient":
    teacher = load_model(model_name='Chen2020Efficient', dataset='cifar100', threat_model='Linf')
elif args.teacher == "Cui2020Learnable_34_10_LBGAT0":
    teacher = load_model(model_name='Cui2020Learnable_34_10_LBGAT0', dataset='cifar100', threat_model='Linf')
elif args.teacher == "Hendrycks2019Using":
    teacher = load_model(model_name='Hendrycks2019Using', dataset='cifar100', threat_model='Linf')
elif args.teacher == "Rade2021Helper_R18_ddpm":
    teacher = load_model(model_name='Rade2021Helper_R18_ddpm', dataset='cifar100', threat_model='Linf')
elif args.teacher == "Cui2020Learnable_34_10_LBGAT6":
    teacher = load_model(model_name='Cui2020Learnable_34_10_LBGAT6', dataset='cifar100', threat_model='Linf')
elif args.teacher == "SELF":
    teacher = resnet18()
    teacher = torch.nn.DataParallel(teacher)
    state_dict = torch.load('./result_models/adaad_IGDM_60_1_0_100_BDM2024-03-25RES-18.pt')
    teacher.load_state_dict(state_dict)
else:
    pass
teacher = teacher.cuda()
#teacher = teacher.half()
teacher.eval()

progress_bar = ProgressBar()

criterion_kl = nn.KLDivLoss(reduction="batchmean")
for epoch in range(1,epochs+1):
    for step,(X,y) in enumerate(trainloader):
        N,_,_,_ = X.shape
        student.train()
        X = X.float().cuda()
        y = y.cuda()
        inputs_adv = adaad_inner_loss(student, teacher, X, step_size=2/255,
                                        steps=10, epsilon=8/255)

        optimizer.zero_grad()
        with torch.no_grad():
            delta = inputs_adv - X
            teacher_plus = teacher(X + args.beta * delta)
        
            teacher_logits = teacher(X)
            
            teacher_minus = teacher(X - args.gamma * delta)
        
        student_plus = student(X + args.beta *delta) 
        student_logits = student(X) 
        student_minus = student(X - args.gamma * delta) 
        ours_loss = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), F.softmax((teacher_plus - teacher_minus).detach(), dim=1))

        adv_outputs = student(inputs_adv)

        with torch.no_grad():
            t_adv_outputs = teacher(inputs_adv)


        kl_loss1 = (1/len(adv_outputs))*torch.sum(nn.KLDivLoss(reduce=False)(
            F.log_softmax(adv_outputs, dim=1), F.softmax(t_adv_outputs.detach(), dim=1)))
        kl_loss2 = (1/len(adv_outputs))*torch.sum(nn.KLDivLoss(reduce=False)(
            F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1)))

        loss = 1*kl_loss1 + (1-1)*kl_loss2 + args.alpha * (epoch/200) * ours_loss

        loss.backward()
        optimizer.step()
        progress_bar.prog(step, len(trainloader), epoch, loss.item())




    if epoch > 90 :
        test_accs = []
        test_accs_adv = []
        student.eval()
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=2.0/255.0,epsilon=8.0/255.0)
            with torch.no_grad():
                logits = student(test_batch_data)
                logits_adv = student(test_ifgsm_data)
            
            predictions_adv = np.argmax(logits_adv.cpu().detach().numpy(),axis=1)
            predictions_adv = predictions_adv - test_batch_labels.cpu().detach().numpy()
            
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            
            test_accs = test_accs + predictions.tolist()
            test_accs_adv = test_accs_adv + predictions_adv.tolist()
        test_accs = np.array(test_accs)
        test_accs_adv = np.array(test_accs_adv)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        test_acc_adv = np.sum(test_accs_adv==0)/len(test_accs_adv)
        print('PGD20 acc',test_acc_adv)


        if not args.nowand:
            d2={'clean_acc': test_acc, 'robust_acc': test_acc_adv}
            wandb.log(d2)



        #torch.save(student.state_dict(),'./models/'+prefix+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
    #if epoch in [215,260,285]:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] *= 0.1
    if epochs > 150 : 
        if epoch in [100, 150]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    else:
        if epoch in [70, 90]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                
save_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
torch.save(student.state_dict(),'./result_models/'+ args.wandb_name + save_time+ str(args.student) + '.pt')

student.eval()
autoattack = AutoAttack(student, norm='Linf', eps=8/255.0, version='standard')
x_total = [x for (x, y) in testloader]
y_total = [y for (x, y) in testloader]
x_total = torch.cat(x_total, 0)
y_total = torch.cat(y_total, 0)
_, robust_acc = autoattack.run_standard_evaluation(x_total, y_total)
print('final AA',robust_acc)
if not args.nowand:
    AA_d = {'RESULT_AA': robust_acc}
    wandb.log(AA_d)

for i in range(6):
    time.sleep(10)

if not args.nowand:
    wandb.finish()