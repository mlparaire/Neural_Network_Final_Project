from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import kagglehub
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict
import numpy as np
import cv2
import seaborn as sns
import os, random, numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, accuracy_score, precision_score, recall_score, 
f1_score,classification_report,roc_curve,precision_recall_fscore_support)
import time, copy,datetime
from sklearn.preprocessing import label_binarize


from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import json

from tqdm import tqdm
##Phase 1
from PIL import Image
import hashlib

from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torchvision import transforms


##Phase 2
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim

class Loader():
    ## We load the dataset 
    def __init__(self,location_training:str = './Dataset/train',location_testing:str='./Dataset/test',seed:int|None = 42):
        try:
            assert os.path.isdir(location_training),'No training dataset found'
        except AssertionError as e:
            print(f'{e}, downloading training dataset')
            kagglehub.dataset_download("josephrynkiewicz/diabetic-retinopathy-test-unzipped")
        finally:
            try:
                assert os.path.isdir(location_testing),'No testing dataset found'
            except AssertionError as e:
                print(f'{e}, Download testing dataset')
                kagglehub.dataset_download("josephrynkiewicz/diabetic-retinopathy-test-unzipped")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"Torch Version: {torch.__version__}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device.type == 'cuda', "Cuda is not available, Please install torch with cu121"
        print("GPU Available with Cuda:", self.device)
        print(f"{torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")


        if not os.path.isdir('./outputs'):
            print("Creating output files directory")
            os.makedirs('./outputs')
            os.makedirs('./outputs/images')
            os.makedirs('./outputs/tensors')
            os.makedirs('./outputs/params')
            os.makedirs('./outputs/report')
        self.training_set = pd.read_csv(
            os.path.join(os.path.abspath(location_training),'trainLabels.csv')
            ).rename(
            columns={"image": "id_code", "level": "label"})
        
        self.training_set['filepath'] = self.training_set['id_code'].apply(lambda x: './Dataset/train/'+str(x)+'.jpeg')
        
        self.SAMPLE_PARAMTERS = {
            "sample": 3500,
            "image_resolution": 224,
            'value_split' : 0.2
        }

        self.HYPER_PARAMETERS = {
            "batch_size": 32, 
            "total_epochs": 15,
            "hp_epochs":2,
            "n_workers":3,
            "lr": 5e-5,
            'dr': 0.5,
            "embed_dim": 256,
            "n_layers": 12,
            'lr':3e-4,
            "wd":1e-4,
            'patience':4
        }

        self.label_map = {
            '0' : 'No DR',
            '1' : 'Mild',
            '2' : 'Moderate',
            '3' :'Severe',
            '4' :'Proliferative DR'
        }

        if not os.path.isdir('./Dataset/train_reshaped'):
            os.makedirs('./Dataset/train_reshaped')

        max_per_class = self.SAMPLE_PARAMTERS['sample'] // 5

        subsets = []
        for label in range(5):
            n_sample = min(len(self.training_set[self.training_set['label'] == label]), max_per_class)
            subsets.append(self.training_set[self.training_set['label'] == label].sample(n=n_sample, random_state=42))

        self.training_set = pd.concat(subsets).reset_index(drop=True)
        print(f"Subset class distribution:\n{self.training_set['label'].value_counts().sort_index()}")
        print(f"Total subset size: {len(self.training_set)}")

        self.class_count = self.training_set['label'].value_counts().sort_index()


        needed = set(self.training_set['id_code'].values+'.jpeg') - set(os.listdir('./Dataset/train_reshaped'))

        if needed == 0:
            print("No resize needed")
        else:
            print(f"{needed} resize(s) needed")
        for name in list(needed):
            img = Image.open('./Dataset/train/'+name).convert('RGB')
            img = img.resize((self.SAMPLE_PARAMTERS['image_resolution'], self.SAMPLE_PARAMTERS['image_resolution']), Image.LANCZOS)
            img.save('./Dataset/train_reshaped/'+name, 'JPEG', quality=100)
            needed.discard(name)
        print('Loading Class successfully loaded')
    @property
    def visualize_random_sample(self) -> Axes:
        samples = self.training_set.sample(6).values
        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 10))
        axs = axs.ravel()
        for n in range(6):
            axs[n].imshow(plt.imread(f"./Dataset/train_reshaped/{samples[n][0]}.jpeg"))
            axs[n].set_title(self.label_map.get(str(samples[n][1])))
            axs[n].axis("off")
        plt.savefig('./outputs/images/random_sample.png')
    
    @property
    def visualize_sample_cat(self) -> Axes:
        samples = []

        for key in self.label_map.keys():
            samples = np.append(samples,self.training_set[self.training_set['label'] == int(key)].sample(1).values)
        samples = samples.reshape(5,3)

        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 10))
        axs = axs.ravel()
        for n in range(5):
            axs[n].imshow(plt.imread(f"./Dataset/train_reshaped/{samples[n][0]}.jpeg"))
            axs[n].set_title(self.label_map.get(str(samples[n][1])))
            axs[n].axis("off")
        plt.savefig('./outputs/images/sample_cat.png',dpi=200)


    def get_size_per_catogery(self) -> Axes:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        ax.bar(self.label_map.values(), self.class_count.values, color=sns.color_palette('turbo', 5))
        ax.set_title('Class Distribution – Full Dataset')
        ax.set_xlabel('DR Severity')
        ax.set_ylabel('Count')
        plt.savefig('./outputs/images/size_per_category.png')

    def get_images_properties(self) -> Axes:
        self.images_address = self.training_set['id_code'].apply(lambda x: './Dataset/train_reshaped/'+str(x)+'.jpeg').values
        widths, heights, aspects, sizes_kb = [], [], [], []
        for p in self.images_address:
            try:
                img = Image.open(p)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w / h)
                sizes_kb.append(os.path.getsize(p) / 1024)
            except Exception:
                continue
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        axes[0].hist(widths, bins=50)
        axes[0].set_title('Image Width Distribution')
        axes[0].set_xlabel('Width (px)')
        axes[0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
        axes[0].legend()

        axes[1].hist(heights, bins=50)
        axes[1].set_title('Image Height Distribution')
        axes[1].set_xlabel('Height (px)')
        axes[1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('./outputs/images/image_properties.png', dpi=150, bbox_inches='tight')
        plt.show()

        if (np.min(widths) - np.max(widths)) <= 1:
            print(f'Width is similar accross all images')
        if (np.min(heights) - np.max(heights)) <= 1:
            print(f'Height is similar accross all images')
        


class Preprocessing(Loader):

    def crop_black_borders(self,img,level:int=5):
        """Remove black padding around the circular retinal image."""
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        mask = gray > level
        if not mask.any():
            return img

        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        if img.ndim == 3:
            return img[y0:y1, x0:x1]
        return img[y0:y1, x0:x1]

    def gamma_correction(self,img, gamma=1.4):
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ]).astype("uint8")
        return cv2.LUT(img, table)
 
    def ben_graham_preprocess(self,img, sigma=8) -> np.array:
        """
        Ben Graham's preprocessing: crop borders, resize, then
        subtract Gaussian-blurred local average to enhance lesions.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = self.crop_black_borders(img)
        img = self.gamma_correction(img)
        
        img = cv2.addWeighted(
            img, 4,
            cv2.GaussianBlur(img, (0, 0), sigma), -4,
            128
        )
        return img
    
    def clahe_preprocess(self,img,cliplimit:float = 2.0):
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = self.crop_black_borders(img)
        img = self.gamma_correction(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img)

        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8,8))
        l = clahe.apply(l)

        img = cv2.merge((l,a,b))
        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def __init__(self):
        super().__init__()
        if not os.path.isdir('./Dataset/train_processed'):
            os.makedirs('./Dataset/train_processed/Ben_Graham')
            os.makedirs('./Dataset/train_processed/Clahe')

        needed_graham = set(self.training_set['id_code'].values+'.jpeg') - set(os.listdir('./Dataset/train_processed/Ben_Graham')) 
        needed_Clahe =  set(self.training_set['id_code'].values+'.jpeg') - set(os.listdir('./Dataset/train_processed/Clahe')) 

        for name in list(needed_graham):
            img = Image.open('./Dataset/train_reshaped/'+name).convert('RGB')
            img = self.ben_graham_preprocess(img)
            img = Image.fromarray(img)
            img.save('./Dataset/train_processed/Ben_Graham/'+name, 'JPEG', quality=100)
            needed_graham.discard(name)

        for name in list(needed_Clahe):
            img = Image.open('./Dataset/train_reshaped/'+name).convert('RGB')
            img = self.clahe_preprocess(img)
            img = Image.fromarray(img)
            img.save('./Dataset/train_processed/Clahe/'+name, 'JPEG', quality=100)
            needed_Clahe.discard(name)

        self.train_df, self.val_df = train_test_split(
            self.training_set, test_size=self.SAMPLE_PARAMTERS['value_split'],
            stratify=self.training_set['label'], random_state=42
        )
        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df = self.val_df.reset_index(drop=True)

        class_counts = self.train_df['label'].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * 5 # Number of classes
        self.class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        print(f"\nThe dataset is balanced, therefore all class have the same weights):")

        sample_weights = [class_weights[label] for label in self.train_df['label'].values]
        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def plot_comparison_with_non_processed(self) -> Axes:
        samples = []
        for label in range(5):
            samples.append(self.training_set[self.training_set['label'] == label].sample(1, random_state=42))

        samples = pd.concat(samples).reset_index(drop=True)
        fig, axs = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(16, 10))
        for col, address in enumerate(samples['id_code']):
            axs[0,col].imshow(plt.imread(f"./Dataset/train_reshaped/{address}.jpeg"))
            axs[0,col].set_title(f'{self.label_map.get(str(samples.iloc[col,1]))}')
            axs[1,col].imshow(plt.imread(f"./Dataset/train_processed/Ben_Graham/{address}.jpeg"))
            axs[1,col].set_title(f'Ben Graham')
            axs[2,col].imshow(plt.imread(f"./Dataset/train_processed/Clahe/{address}.jpeg"))
            axs[2,col].set_title(f'Clahe')
        plt.savefig('./outputs/images/sample_comparison_processed.png')


class Image_Processing():
    def __init__(self,image_addresses:pd.DataFrame,tensors:str='Train',transformation:'str'=None,size_of_image:int=224):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.image_addresses = image_addresses
        self.tensors = tensors
        self.transformation=transformation
        self.train_transforms = transforms.Compose([
            transforms.Resize((size_of_image, size_of_image)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((size_of_image,size_of_image)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_addresses)

        
    def __getitem__(self, idx):
            row = self.image_addresses.iloc[idx]
            filepath,label  = row['id_code'],row['label']

            if self.transformation == 'Ben_Graham':
                img = Image.open("./Dataset/train_processed/Ben_Graham/"+filepath+'.jpeg').convert('RGB')
            elif self.transformation == 'Clahe':
                img = Image.open("./Dataset/train_processed/Clahe/"+filepath+'.jpeg').convert('RGB')
            else:
                img = Image.open("./Dataset/train_reshaped/"+filepath+'.jpeg').convert('RGB')


            if self.tensors == "train":
                img = self.train_transforms(img)
            else:
                img = self.val_transforms(img)

            return img, label
        
class EarlyStopping:
    """Stop training when validation metric stops improving."""
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
 

class ResNet50(Preprocessing):
    def __init__(self,chosen_transformation:str=None,dropout:float=0.5,freeze_backbone:bool=False):
        super().__init__()
        self.chosen_transformation = chosen_transformation
        train_dataset = Image_Processing(self.train_df, tensors='Train',transformation=self.chosen_transformation,size_of_image=self.SAMPLE_PARAMTERS['image_resolution'])
        val_dataset = Image_Processing(self.val_df, tensors='test',transformation=self.chosen_transformation,size_of_image=self.SAMPLE_PARAMTERS['image_resolution'])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.HYPER_PARAMETERS['batch_size'], sampler=self.sampler,
            num_workers=self.HYPER_PARAMETERS['n_workers'], pin_memory=True, drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False,
            num_workers=2, pin_memory=True
        )
        print(f"Train DataLoader successfully loaded: {len(self.train_loader)} batches x {32} = {len(self.train_loader)*32} samples/epoch")
        print(f"Val DataLoader: {len(self.val_loader)} batches x {32}")

        print("Build ResNet50")
        self.model_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in self.model_resnet50.parameters():
                param.requires_grad = False
        self.in_features_resnet = self.model_resnet50.fc.in_features
        self.model_resnet50.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features_resnet, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 5)
        )

        print("Build EfficientNet-B3")
        self.model_efficientnet_b3  = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in self.model_efficientnet_b3.parameters():
                param.requires_grad = False
        self.in_features_efficient = self.model_efficientnet_b3.classifier[1].in_features
        self.model_efficientnet_b3.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features_efficient, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 5)
       )
 
        print(self.model_resnet50.fc)
        print(f"Total parameters: {sum(p.numel() for p in self.model_resnet50.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model_resnet50.parameters() if p.requires_grad):,}")

        print("\n=== EfficientNet-B3 Architecture (classifier head) ===")
        print(self.model_efficientnet_b3.classifier)
        print(f"Total parameters: {sum(p.numel() for p in self.model_efficientnet_b3.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model_efficientnet_b3.parameters() if p.requires_grad):,}")

    def plot_training_batch(self) -> Axes:
        batch_imgs, batch_labels = next(iter(self.train_loader))
        fig, axes = plt.subplots(2, 8, figsize=(20, 6))
        for i in range(min(16, len(batch_imgs))):
            ax = axes[i // 8, i % 8]
            img = batch_imgs[i].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(self.label_map.get(str(batch_labels[i].item())), fontsize=9)
            ax.axis('off')
            fig.suptitle('Sample Training Batch', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_testing_batch(self) -> Axes:
        batch_imgs, batch_labels = next(iter(self.val_loader))
        fig, axes = plt.subplots(2, 8, figsize=(20, 6))
        for i in range(min(16, len(batch_imgs))):
            ax = axes[i // 8, i % 8]
            img = batch_imgs[i].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(self.label_map.get(str(batch_labels[i].item())), fontsize=9)
            ax.axis('off')
            fig.suptitle('Sample Testing Batch', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def train_one_epoch(self,model, loader, criterion, optimizer, scaler):
        """Train for one epoch with mixed precision."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        return running_loss / total, correct / total


    @torch.no_grad()
    def validate(self,model, loader, criterion):
        """Validate and return predictions + probabilities."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs.float(), dim=1)
            _, preds = probs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        return (
            running_loss / total,
            correct / total,
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs)
        )

    def train_model(self,
                    model_name:str,
                    chosen_transformation:str,
                    lr:float = None,
                    weight_decay:float = None,
                    epochs:int = None,
                    best_model:bool = False
                    ):
        """Full training loop with logging, scheduling, and early stopping."""
        
        if best_model:
            parameters = json.load(open(f'./outputs/params/best_params_ResNet_{chosen_transformation}_{hashlib.sha256(datetime.date.today().strftime("%A %b %d %Y").encode()).hexdigest()[:5]}.json','r'))
            lr = parameters['lr']
            weight_decay = parameters['weight_decay']
        else:
            lr = lr if lr is not None else self.HYPER_PARAMETERS['lr']
            weight_decay = weight_decay if weight_decay is not None else self.HYPER_PARAMETERS['wd']
        
        epochs,patience= epochs if epochs is not None else self.HYPER_PARAMETERS['total_epochs'],self.HYPER_PARAMETERS['patience']
        print(f"\n{'='*60}")
        model = getattr(self,f'{model_name}')
        print(f"  Training {model_name}")
        print(f"  LR={lr}, WD={weight_decay}, Epochs={epochs}, Patience={patience}")
        print(f"{'='*60}")

        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        scaler = GradScaler("cuda")
        early_stopping = EarlyStopping(patience=patience, mode='max')

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_kappa': []
        }
        best_kappa = -1
        best_model_state = None
        best_epoch = 0

        for epoch in range(epochs):
            start_time = time.time()
            

            train_loss, train_acc = self.train_one_epoch(
                model, self.train_loader, criterion, optimizer, scaler
            )
            val_loss, val_acc, val_preds, val_labels,val_probs  = self.validate(
                model, self.val_loader, criterion
            )
            val_kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')

            scheduler.step()
            elapsed = time.time() - start_time

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_kappa'].append(val_kappa)

            print(f"  Epoch {epoch+1:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} QWK: {val_kappa:.4f} | "
                  f"{elapsed:.1f}s")

            if val_kappa > best_kappa:
                best_kappa = val_kappa
                best_epoch = epoch + 1
                if isinstance(model, nn.DataParallel):
                    best_model_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                else:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if early_stopping(val_kappa):
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

        print(f"\n  Best epoch: {best_epoch} with QWK = {best_kappa:.4f}")

        if best_model_state is not None:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(best_model_state)
            else:
                model.load_state_dict(best_model_state)

        if best_model:
            resnet_save_path = os.path.join('./outputs/tensors/', f'{model._get_name()+"_"+self.chosen_transformation}_best.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), resnet_save_path)
            else:
                torch.save(model.state_dict(), resnet_save_path)
            print(f"ResNet50 model saved to {resnet_save_path}")
        
        self.model = model
                    
        return model, history, best_kappa
    

    def tune_model(self,model_name:str,config:Dict) -> None:

        print(f"Hyperparameter search for {getattr(self,f'{model_name}')._get_name()} ({len(config)} configs x {self.HYPER_PARAMETERS['total_epochs']} epochs):")
        print("-" * 70)

        hp_results_resnet = []
        for i, hp in enumerate(config):
            print(f"\nConfig {i+1}/{len(config)}: {hp}")
            model, hist, best_kappa = self.train_model(model_name,lr = hp['lr'],weight_decay=hp['weight_decay'])
            hp_results_resnet.append({**hp, 'cohen_kappa': best_kappa})
            del model
            torch.cuda.empty_cache()

        hp_df_resnet = pd.DataFrame(hp_results_resnet)
        print(f"\n\nHyperparameter Search Results {getattr(self,f'{model_name}')._get_name()}:")
        print(hp_df_resnet.to_string(index=False))

        self.best_params = best_hp_resnet = max(hp_results_resnet, key=lambda x: x['cohen_kappa'])
        print(f"\nBest hyperparameters for {getattr(self,model_name)._get_name()+'_'+self.chosen_transformation}: {best_hp_resnet}")
        with open(f'./outputs/params/best_params_{getattr(self,model_name)._get_name()+"_"+self.chosen_transformation}_{hashlib.sha256(datetime.date.today().strftime("%A %b %d %Y").encode()).hexdigest()[:5]}.json','w') as f:
            json.dump(self.best_params,f)
    
    def plot_training_diagnosis(self,history:Dict,model_name:str)-> Axes:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=4)
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=4)
        axes[0].set_title(f'{getattr(self,model_name)._get_name()} - Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=4)
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s', markersize=4)
        axes[1].set_title(f'{getattr(self,model_name)._get_name()} - Accuracy', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(history['val_kappa'], label='Val QWK', marker='D', markersize=4, color='green')
        axes[2].set_title(f'{getattr(self,model_name)._get_name()} - Cohen Quadratic Weighted Kappa', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('QWK')
        axes[2].legend()
        axes[2].grid(True)

        plt.suptitle(f'{getattr(self,model_name)._get_name()} Training Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join('./outputs/images', 'efficientnet_b3_training_curves.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, model_name:str):
        criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        val_loss, val_acc, preds, labels,probs  = self.validate(getattr(self,model_name), self.val_loader, criterion)

        qwk = cohen_kappa_score(labels, preds, weights='quadratic')
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, labels=list(range(5))
        )
        macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
        weighted_f1 = precision_recall_fscore_support(labels, preds, average='weighted')[2]

        print(f"\n{'='*60}")
        print(f"  {model_name} - Evaluation Results")
        print(f"{'='*60}")
        print(f"  Validation Loss:     {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Quadratic W. Kappa:  {qwk:.4f}")
        print(f"  Macro F1-Score:      {macro_f1:.4f}")
        print(f"  Weighted F1-Score:   {weighted_f1:.4f}")

        self.val_loss = val_loss
        self.val_acc = val_acc,
        self.qwk = qwk
        self.macro_f1 = macro_f1,
        self.weighted_f1 = weighted_f1,
        self.preds = preds
        self.labels =labels
        self.probs = probs,
        self.per_class_precision = precision,
        self.per_class_recall = recall
        self.per_class_f1 =  f1
    
    def compute_confusion_matrix(self, model_name:str):
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        cm = confusion_matrix(self.labels, self.preds, labels=list(range(5)))
        cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=self.label_map.values(), yticklabels=self.label_map.values(),
            cbar_kws={'label': 'Count'})
        for i in range(5):
            for j in range(5):
                ax.text(j + 0.5, i + 0.75, f'({cm_pct[i, j]:.0f}%)',
                    ha='center', va='center', fontsize=8, color='gray')
        ax.set_title(f"{getattr(self,model_name)._get_name()}\nQWK={self.qwk:.2f} | Acc={self.val_acc:.2f}",
                 fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join('./outputs.reports/', f'confusion_matrices_{getattr(self,model_name)._get_name()}.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def compute_reports(self, model_name:str):
        print(f"  Classification Report - {getattr(self,model_name)._get_name()}")
        print(classification_report(
        self.labels, self.preds,
        target_names=self.label_map.values(), digits=4,
        labels=list(range(5))
        ))

class Predict_with_model(Preprocessing):
    def __init__(self, model, images_dir: str, preprocess_method: str = "ben_graham"):
        self.model = model
        self.preprocess_method = preprocess_method
        self.images_dir = images_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, img):
        if self.preprocess_method == "ben_graham":
            img = self.ben_graham_preprocess(img)
        elif self.preprocess_method == "clahe":
            img = self.clahe_preprocess(img)
        return img

    def predict_from_images(self):
        image_paths = [
            './Dataset/test/'+p for p in os.listdir(self.images_dir)
        ]

        self.model.eval()
        results = {'image' : [],'level':[]}

        with torch.no_grad():
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                img = self.preprocess(img)          # Ben Graham / CLAHE
                img = Image.fromarray(img)          # back to PIL
                tensor = self.transform(img).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                output = self.model(tensor)
                pred = torch.argmax(output, dim=1).item()
                results['image'].append(path.replace('./Dataset/test/','').replace('.jpeg',''))
                results['level'].append(pred)

        return pd.DataFrame(results)