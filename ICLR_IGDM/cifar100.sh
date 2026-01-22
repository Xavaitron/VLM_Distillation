GPU_ID=0

method=ard_IGDM
alpha=20
beta=1
gamma=1
epochs=200
teacher=BDM
student=RES-18

nowand=1
wandb_project=wandb_entity
wandb_entity=wandb_entity


for teacher in BDM 
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${method}_cifar100.py --wandb_name ${method}_${alpha}_${beta}_${gamma}_${epochs}_${teacher} --alpha $alpha --beta $beta --gamma $gamma --nowand $nowand --wandb_project $wandb_project --wandb_entity $wandb_entity --method $method --epochs $epochs --teacher $teacher --student $student 
done