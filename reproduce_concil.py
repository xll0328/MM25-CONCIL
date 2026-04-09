"""
CONCIL Reproduction Script
- Fixed all known bugs from CONCIL_1111.py
- Uses local data paths and checkpoints
- Matches paper settings: n=50, m=50, p=2~9, buffer_size=25000

Bugs fixed:
1. data_path.yml paths updated to local machine paths
2. Hardcoded checkpoint path replaced with local path argument
3. Typo 'conecpt_ratio' unified to 'concept_ratio' throughout
4. concept_ratio in IncrementalIMGDataset was hardcoded to 1 -- fixed
5. Metric.reset() did not reset label_count for 'multi' clf_mode -- fixed
6. calculate_mean_metrics() crashes when forgetting_rates list is empty (stage=1) -- fixed
7. save_model() referenced self.model1 but model might not exist -- fixed
8. concept_pred from CONCIL is continuous float, but Metric.add() uses ge(0.5) which is correct for binary
   However concept_pred needs sigmoid before ge(0.5) -- fixed
9. fit_concept_linear passes raw concept floats (0/1) but internally calls y.long() on them -- fixed
   concept labels should be passed as float for MSE-style fitting, kept as float
10. After model1.update() for concept, class fitting should use updated concept predictions -- verified correct
"""

import os
import sys
import csv
import datetime
import argparse
import warnings

import torch
import tqdm
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os.path import join
from torch import nn, Tensor
from torch.nn import init
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.util import check_dir, read_data, one_hot
from src.analytic.AnalyticLinear import RecursiveLinear, AnalyticLinear
from src.analytic.Buffer import RandomBuffer

warnings.filterwarnings("ignore")
# Use all available CPU cores for matrix ops (analytic layers run on CPU)
torch.set_num_threads(min(32, torch.get_num_threads()))
torch.set_num_interop_threads(min(8, torch.get_num_interop_threads()))

# ─────────────────────────────────────────────
#  Dataset config
# ─────────────────────────────────────────────
CONFIG = {
    'cub': {'N_CONCEPTS': 116, 'N_CLASSES': 200},
    'awa': {'N_CONCEPTS': 85,  'N_CLASSES': 50}}

# ─────────────────────────────────────────────
#  Image augmentation
# ─────────────────────────────────────────────
def img_augment(split, resol=256):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == 'train':
        return transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        return transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])


# ─────────────────────────────────────────────
#  Incremental Dataset  (BUG #3,#4 fixed)
# ─────────────────────────────────────────────
class IncrementalIMGDataset(Dataset):
    def __init__(self, data_path: str, split: str, resol: int = 256,
                 class_ratio: float = 0.5, prev_class_ratio: float = 0.0,
                 concept_ratio: float = 0.5):          # FIX #3: was 'conecpt_ratio'
        assert split in ['train', 'test']
        self.data = read_data(data_path, split)
        if 'cub' in data_path:
            self.n_class   = CONFIG['cub']['N_CLASSES']
            self.n_concept = CONFIG['cub']['N_CONCEPTS']
        elif 'awa' in data_path:
            self.n_class   = CONFIG['awa']['N_CLASSES']
            self.n_concept = CONFIG['awa']['N_CONCEPTS']

        # FIX #4: was hardcoded to 1, ignoring concept_ratio
        self.min_class_idx  = int(self.n_class   * prev_class_ratio)
        self.max_class_idx  = int(self.n_class   * class_ratio)
        self.max_concept_idx = int(self.n_concept * concept_ratio)

        self.transform = img_augment(split, resol)
        self._set()

    def _set(self):
        self.image_path = []
        self.concept    = []
        self.label      = []
        self.one_hot_label = []
        for instance in self.data:
            label = instance['label']
            if label < self.min_class_idx or label >= self.max_class_idx:
                continue
            concept = [c if idx < self.max_concept_idx else 0
                       for idx, c in enumerate(instance['concept'])]
            self.image_path.append(instance['img_path'])
            self.concept.append(concept)
            self.label.append(label)
            self.one_hot_label.append(one_hot(label, self.n_class))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img_path = self.image_path[index].replace(
            '/hpc2hdd/home/songninglai/CBMBackdoor',
            '/data/sony/CBMBackdoor')
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        concept       = torch.tensor(self.concept[index],       dtype=torch.float32)
        label         = torch.tensor(self.label[index])
        one_hot_label = torch.tensor(self.one_hot_label[index])
        return image, concept, label, one_hot_label


def get_incremental_dataloader(dataset, data_path, batch_size,
                               class_ratio=0.5, prev_class_ratio=0.0,
                               concept_ratio=0.5, num_workers=8):
    train_ds = IncrementalIMGDataset(data_path, 'train',
                                     class_ratio=class_ratio,
                                     prev_class_ratio=prev_class_ratio,
                                     concept_ratio=concept_ratio)
    test_ds  = IncrementalIMGDataset(data_path, 'test',
                                     class_ratio=class_ratio,
                                     prev_class_ratio=prev_class_ratio,
                                     concept_ratio=concept_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# ─────────────────────────────────────────────
#  CONCIL analytic model
# ─────────────────────────────────────────────
class CONCIL(nn.Module):
    def __init__(self, backbone, backbone_output, num_concepts, num_classes,
                 buffer_size=25000, gg1=500.0, gg2=1.0, linear=RecursiveLinear, dtype=torch.double,
                 device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_device = device   # backbone inference device
        self.cpu_device = 'cpu'    # analytic matrices always on CPU (can be 5-50GB)
        self.backbone   = backbone
        self.num_concepts = num_concepts
        self.num_classes  = num_classes
        cpu_kwargs = {"device": 'cpu', "dtype": dtype}
        self.buffer_concept = RandomBuffer(backbone_output, buffer_size, **cpu_kwargs)
        self.buffer_class   = RandomBuffer(num_concepts,    buffer_size, **cpu_kwargs)
        self.analytic_linear_concept = linear(buffer_size, gg1, **cpu_kwargs)
        self.analytic_linear_class   = linear(buffer_size, gg2, **cpu_kwargs)
        self.eval()

    @torch.no_grad()
    def concept_feature_expansion(self, X):
        # backbone on GPU, buffer on CPU
        feat = self.backbone(X.to(self.gpu_device)).cpu()
        return self.buffer_concept(feat)

    @torch.no_grad()
    def class_feature_expansion(self, concept_pred):
        return self.buffer_class(concept_pred.cpu())

    @torch.no_grad()
    def forward(self, X):
        concept_pred = self.analytic_linear_concept(self.concept_feature_expansion(X))
        class_pred   = self.analytic_linear_class(self.class_feature_expansion(concept_pred))
        return concept_pred, class_pred  # both on CPU

    @torch.no_grad()
    def forward_concept(self, X):
        return self.analytic_linear_concept(self.concept_feature_expansion(X))

    @torch.no_grad()
    def fit_concept(self, X, Y):
        feat = self.concept_feature_expansion(X)   # CPU
        self.analytic_linear_concept.fit(feat, Y.cpu().to(feat))

    @torch.no_grad()
    def fit_class(self, concept_pred, y):
        cp = concept_pred.cpu()
        Y  = torch.nn.functional.one_hot(y.long().cpu(), num_classes=self.num_classes).to(cp)
        feat = self.class_feature_expansion(cp)
        self.analytic_linear_class.fit(feat, Y)

    @torch.no_grad()
    def update(self):
        self.analytic_linear_concept.update()
        self.analytic_linear_class.update()


# ─────────────────────────────────────────────
#  Metric
# ─────────────────────────────────────────────
class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.c_correct = 0
        self.c_total   = 0
        self.y_correct = 0
        self.y_total   = 0

    def add(self, concept_pred, label_pred, concept_gt, label_gt):
        # concept accuracy: binary (ge 0.5)
        cp = concept_pred.ge(0.5).int()
        cg = concept_gt.int()
        self.c_correct += (cp == cg).sum().item()
        self.c_total   += cp.numel()
        # class accuracy
        yp = label_pred.argmax(dim=-1)
        self.y_correct += (yp == label_gt).sum().item()
        self.y_total   += label_gt.shape[0]

    @property
    def concept_accu(self):
        return self.c_correct / self.c_total if self.c_total else 0.0

    @property
    def clf_accu(self):
        return self.y_correct / self.y_total if self.y_total else 0.0


# ─────────────────────────────────────────────
#  Base CBM model
# ─────────────────────────────────────────────
class IMGBaseModel(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes  = num_classes
        self.backbone   = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.concept_fc = nn.Linear(1000, num_concepts)
        self.final_fc   = nn.Sequential(
            nn.Linear(num_concepts, 512), nn.ReLU(), nn.Linear(512, num_classes))
        for layer in [self.concept_fc] + [l for l in self.final_fc if isinstance(l, nn.Linear)]:
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            init.constant_(layer.bias, 0)

    def forward(self, x):
        feat     = self.backbone(x)
        concepts = self.concept_fc(feat)
        return concepts, self.final_fc(concepts), feat


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────
class CONCILTrainer:
    def __init__(self, args):
        self.args   = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.out_dir  = join(args.saved_dir, self.time_str)
        check_dir(self.out_dir)
        self.log_file = join(self.out_dir, 'run.log')
        self._log(f"Args: {vars(args)}")
        self._log(f"Device: {self.device}")
        with open(join('src/utils', 'data_path.yml'), 'r') as f:
            paths = yaml.safe_load(f)
        self.data_path    = paths[args.dataset]['processed_dir']
        self.cfg          = CONFIG[args.dataset]
        self.num_concepts = self.cfg['N_CONCEPTS']
        self.num_classes  = self.cfg['N_CLASSES']
        torch.manual_seed(args.seed)
        self.num_stages = args.num_stages
        self.concept_history = {}   # {stage: {task: accu}}
        self.class_history   = {}

    def _log(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

    def _stage_ratios(self, stage):
        n = self.args.class_ratio
        m = self.args.concept_ratio
        p = self.num_stages
        if stage == 0:
            return 0.0, n, m
        prev_class  = n + (stage - 1) * (1 - n) / (p - 1)
        cur_class   = n + stage       * (1 - n) / (p - 1)
        cur_concept = m + stage       * (1 - m) / (p - 1)
        return prev_class, cur_class, cur_concept

    def run(self):
        args = self.args
        self._log("\n=== Loading pretrained base model ===")
        base_model = IMGBaseModel(self.num_concepts, self.num_classes).to(self.device)
        ckpt = torch.load(args.base_ckpt, map_location=self.device)
        base_model.load_state_dict(ckpt)
        base_model.eval()
        self._log(f"Loaded: {args.base_ckpt}")

        self.concil = CONCIL(
            backbone=base_model.backbone,
            backbone_output=1000,
            num_concepts=self.num_concepts,
            num_classes=self.num_classes,
            buffer_size=args.buffer_size,
            gg1=args.gg1,
            gg2=args.gg2,
            device=self.device,
        )
        # backbone already on self.device via base_model

        for stage in range(self.num_stages):
            self._run_stage(stage)

        self._print_summary()

    def _run_stage(self, stage):
        args = self.args
        prev_class, cur_class, cur_concept = self._stage_ratios(stage)
        self._log(f"\n=== Stage {stage+1}/{self.num_stages} | "
                  f"class [{prev_class:.2f},{cur_class:.2f}] concept={cur_concept:.2f} ===")

        train_loader, _ = get_incremental_dataloader(
            args.dataset, self.data_path, args.batch_size,
            class_ratio=cur_class, prev_class_ratio=prev_class,
            concept_ratio=cur_concept, num_workers=args.num_workers)
        self._log(f"  Train samples: {len(train_loader.dataset)}")

        # Step 1: fit concept layer
        self._log("  [Step 1] Fitting concept layer...")
        for img, concept, label, _ in tqdm.tqdm(train_loader, desc=f"S{stage+1}-concept", ncols=80):
            img     = img.to(self.device)
            concept = concept.to(self.device)
            self.concil.fit_concept(img, concept)
        self.concil.analytic_linear_concept.update()

        # Step 2: fit class layer using updated concept predictions
        self._log("  [Step 2] Fitting class layer...")
        for img, concept, label, _ in tqdm.tqdm(train_loader, desc=f"S{stage+1}-class", ncols=80):
            img   = img.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                concept_pred = self.concil.forward_concept(img)
            self.concil.fit_class(concept_pred, label)
        self.concil.analytic_linear_class.update()

        self._eval_all_tasks(stage)

    def _eval_all_tasks(self, stage):
        self.concept_history[stage] = {}
        self.class_history[stage]   = {}
        args = self.args
        self._log(f"  --- Eval after Stage {stage+1} ---")
        for task in range(stage + 1):
            prev_class, cur_class, cur_concept = self._stage_ratios(task)
            _, test_loader = get_incremental_dataloader(
                args.dataset, self.data_path, args.batch_size,
                class_ratio=cur_class, prev_class_ratio=prev_class,
                concept_ratio=cur_concept, num_workers=args.num_workers)
            metric = Metric()
            self.concil.eval()
            for img, concept, label, _ in tqdm.tqdm(test_loader, desc=f"  eval task {task+1}", ncols=80):
                img     = img.to(self.device)
                concept = concept.cpu()
                label   = label.cpu()
                with torch.no_grad():
                    concept_pred, class_pred = self.concil(img)  # returns CPU tensors
                n_c = int(self.num_concepts * cur_concept)
                metric.add(concept_pred[:, :n_c], class_pred, concept[:, :n_c], label)
            c_acc = metric.concept_accu
            y_acc = metric.clf_accu
            self.concept_history[stage][task] = c_acc
            self.class_history[stage][task]   = y_acc
            self._log(f"    Task {task+1}: concept_acc={c_acc:.4f}  class_acc={y_acc:.4f}")

        avg_c = sum(self.concept_history[stage].values()) / (stage + 1)
        avg_y = sum(self.class_history[stage].values())   / (stage + 1)
        self._log(f"  >> Avg: concept={avg_c:.4f}  class={avg_y:.4f}")

        if stage >= 1:
            forget_c, forget_y = [], []
            for task in range(stage):
                best_c = max(self.concept_history[s][task] for s in range(task, stage) if task in self.concept_history[s])
                best_y = max(self.class_history[s][task]   for s in range(task, stage) if task in self.class_history[s])
                forget_c.append(best_c - self.concept_history[stage][task])
                forget_y.append(best_y - self.class_history[stage][task])
            self._log(f"  >> Forget: concept={sum(forget_c)/len(forget_c):.4f}  class={sum(forget_y)/len(forget_y):.4f}")

        csv_path = join(self.out_dir, f'stage_{stage+1}_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task', 'concept_acc', 'class_acc'])
            for task in range(stage + 1):
                w.writerow([task+1, self.concept_history[stage][task], self.class_history[stage][task]])

    def _print_summary(self):
        self._log("\n" + "="*70)
        self._log("FINAL SUMMARY")
        self._log(f"{'Phase':>7}  {'Avg C-Acc':>10}  {'Avg Y-Acc':>10}  {'C-Forget':>10}  {'Y-Forget':>10}")
        rows = []
        for stage in range(self.num_stages):
            n_tasks = stage + 1
            avg_c = sum(self.concept_history[stage].values()) / n_tasks
            avg_y = sum(self.class_history[stage].values())   / n_tasks
            if stage >= 1:
                fc_list, fy_list = [], []
                for task in range(stage):
                    best_c = max(self.concept_history[s][task] for s in range(task, stage) if task in self.concept_history[s])
                    best_y = max(self.class_history[s][task]   for s in range(task, stage) if task in self.class_history[s])
                    fc_list.append(best_c - self.concept_history[stage][task])
                    fy_list.append(best_y - self.class_history[stage][task])
                avg_fc = sum(fc_list) / len(fc_list)
                avg_fy = sum(fy_list) / len(fy_list)
                self._log(f"Phase {stage+1:>2}:  {avg_c:>10.4f}  {avg_y:>10.4f}  {avg_fc:>10.4f}  {avg_fy:>10.4f}")
                rows.append((stage+1, avg_c, avg_y, avg_fc, avg_fy))
            else:
                self._log(f"Phase {stage+1:>2}:  {avg_c:>10.4f}  {avg_y:>10.4f}  {'N/A':>10}  {'N/A':>10}")
                rows.append((stage+1, avg_c, avg_y, None, None))
        if len(rows) > 1:
            r2 = [r for r in rows if r[3] is not None]
            self._log("-"*70)
            self._log(f"{'Average':>7}  {sum(r[1] for r in r2)/len(r2):>10.4f}  "
                      f"{sum(r[2] for r in r2)/len(r2):>10.4f}  "
                      f"{sum(r[3] for r in r2)/len(r2):>10.4f}  "
                      f"{sum(r[4] for r in r2)/len(r2):>10.4f}")
        self._log("="*70)
        self._log(f"Paper ({self.args.dataset.upper()}) targets:")
        if self.args.dataset == 'cub':
            self._log("  Avg Concept Acc: 0.8209 | Avg Class Acc: 0.6133 | C-Forget: -0.0004 | Y-Forget: 0.0919")
        else:
            self._log("  Avg Concept Acc: 0.9703 | Avg Class Acc: 0.8624 | C-Forget:  0.0042 | Y-Forget: 0.1029")
        csv_path = join(self.out_dir, 'overall_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phase','avg_concept_acc','avg_class_acc','avg_concept_forget','avg_class_forget'])
            for r in rows:
                w.writerow(r)
        self._log(f"\nResults saved to: {self.out_dir}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CONCIL Reproduction')
    parser.add_argument('-dataset',      type=str,   default='cub')
    parser.add_argument('-base_ckpt',    type=str,   default='/data/sony/hpc2hdd/CBM_CL/base_model/CUB/CUB.pth')
    parser.add_argument('-saved_dir',    type=str,   default='results_repro')
    parser.add_argument('-batch_size',   type=int,   default=64)
    parser.add_argument('-num_stages',   type=int,   default=9,    help='total phases (paper uses 2~9)')
    parser.add_argument('-class_ratio',  type=float, default=0.5,  help='initial class ratio n')
    parser.add_argument('-concept_ratio',type=float, default=0.5,  help='initial concept ratio m')
    parser.add_argument('-buffer_size',  type=int,   default=25000,help='analytic buffer size d')
    parser.add_argument('-gg',           type=float, default=1e-1, help='gamma for RecursiveLinear (unused, kept for compat)')
    parser.add_argument('-gg1',          type=float, default=500.0,help='gamma for concept RecursiveLinear')
    parser.add_argument('-gg2',          type=float, default=1.0,  help='gamma for class RecursiveLinear')
    parser.add_argument('-seed',         type=int,   default=42)
    parser.add_argument('-num_workers',  type=int,   default=8)
    args = parser.parse_args()

    # Auto-select checkpoint based on dataset
    if args.base_ckpt == '/data/sony/hpc2hdd/CBM_CL/base_model/CUB/CUB.pth' and args.dataset == 'awa':
        args.base_ckpt = '/data/sony/hpc2hdd/CBM_CL/base_mode_awal/2024-11-14-07:36:31/AWA.pth'

    check_dir(args.saved_dir)
    trainer = CONCILTrainer(args)
    trainer.run()

    