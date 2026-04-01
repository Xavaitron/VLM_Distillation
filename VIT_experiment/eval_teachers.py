import os
import time
import csv
import argparse
import torch
from dataset import get_cifar100_dataloaders
from models import get_cnn_teacher, get_vit_teacher
from attacks import eval_robustness

def main():
    parser = argparse.ArgumentParser(description='Evaluate Teacher Models and log to training_results.csv')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, testloader = get_cifar100_dataloaders(batch_size=args.batch_size, img_size=32)

    teachers = [
        {
            'name': 'Wang2023Better_WRN-28-10',
            'arch': 'CNN',
            'method': 'TEACHER',
            'loader': lambda: get_cnn_teacher('Wang2023Better_WRN-28-10'),
        },
        {
            'name': 'Debenedetti2022Light_XCiT-S12',
            'arch': 'VIT',
            'method': 'TEACHER',
            'loader': lambda: get_vit_teacher('Debenedetti2022Light_XCiT-S12'),
        },
    ]

    csv_file = os.path.join('result_models', 'training_results.csv')
    os.makedirs('result_models', exist_ok=True)

    for t in teachers:
        print(f"\n{'='*60}")
        print(f"Evaluating {t['arch']} Teacher: {t['name']}")
        print(f"{'='*60}")

        model = t['loader']().to(device)
        model.eval()

        clean_acc, pgd_acc, fgsm_acc, cw_acc, aa_acc = eval_robustness(model, testloader, device)

        print(f"  Clean: {clean_acc*100:.2f}%  FGSM: {fgsm_acc*100:.2f}%  "
              f"PGD-20: {pgd_acc*100:.2f}%  C&W: {cw_acc*100:.2f}%  AA: {aa_acc*100:.2f}%")

        # Append to CSV
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Date', 'Architecture', 'Method', 'Teacher', 'Batch Size', 'Epochs',
                                 'Clean Acc (%)', 'FGSM Acc (%)', 'PGD20 Acc (%)', 'C&W Acc (%)', 'AA Acc (%)'])
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                t['arch'],
                t['method'],
                t['name'],
                args.batch_size,
                '-',
                f"{clean_acc*100:.2f}",
                f"{fgsm_acc*100:.2f}",
                f"{pgd_acc*100:.2f}",
                f"{cw_acc*100:.2f}",
                f"{aa_acc*100:.2f}",
            ])

        # Free GPU memory before loading next teacher
        del model
        torch.cuda.empty_cache()

    print(f"\n✅ Teacher results appended to {csv_file}")

if __name__ == '__main__':
    main()
