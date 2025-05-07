# %% [markdown]
# # local 実行！

# %%
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
# または環境変数を完全に無効化
os.environ["PJRT_DEVICE"] = "CUDA"

# %%
import os
import sys
import re
import copy
import random
import glob
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image


import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch import seed_everything as pl_seed_everything
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import QuantileTransformer

import transformers

import wandb

from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as AF

# from joai_toolkit.src.cv import image_augumentation



# %%
class CFG : 
    exp_name = "a01_gem_deberta_anotherseed"
    random_seed = 496
    
    # model info
    model = "resnet50.a1h_in1k"
    use_custom_pooling = False
    dropout_ratio = 0.3
    
    # train info
    epochs = 100
    batch_size = 64
    learning_rate = 1e-3
    num_workers = 8
    num_classes = 4
    device = "cuda"
    criterion =  nn.BCEWithLogitsLoss
    optimizer = optim.AdamW 
    scheduler = transformers.get_cosine_schedule_with_warmup
    warmup_prop = 0.1
    patience = 10
    precision = "16-mixed" 
    do_tta = False
    smoothing = 0.1
    
    # run info
    debug_one_epoch = False
    debug_one_fold = False
    only_infer = False
    do_wandb = True
    
    # data info
    imput_img_size = 224
    train_img_size = 224
    data_dir = "../dataset"
    output_dir = "../outputs"
    fold = 5
    
config = CFG()

# %%
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    pl_seed_everything(seed)
    
seed_everything(config.random_seed)

# %%
def class_to_dict(obj):
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("__") and not callable(getattr(obj, k))}
class_dict = class_to_dict(config)

# %%
class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Citation : This code is copied from and based on the original code from https://www.kaggle.com/code/haqishen/gridmask
    Thanks to the author.
    
    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    
    
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)
                
    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        # 新しい配列を作成して返す
        new_image = image.copy()  # 画像のコピーを作成
        new_image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return new_image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


# %%
train_transform = A.Compose([
    A.Resize(config.train_img_size, config.train_img_size),
    A.RandomRotate90(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(config.imput_img_size, config.imput_img_size),
    A.Normalize(),
    ToTensorV2()
])

# %%
def add_degrees_celcius(df) -> pd.DataFrame:
    """
    Add a column to the DataFrame indicating if the temperature is in Celsius.
    """
    df["Caption"] = df["Caption"].astype(str) 
    df['is_degrees_celsius'] = df["Caption"].str.contains("°C", na=False)
    df['is_degrees_celsius'] = df['is_degrees_celsius'].astype(int)
    # x-y°C の形式になっている
    df["min_temp"] = -1
    df["max_temp"] = -1
    for i, row in df.iterrows():
        text = row["Caption"]
        assert isinstance(text, str)
        if "°C" in text:
            # x-y°C の形式になっている
            match = re.search(r'(\d+)-(\d+)°C', text)
            if match:
                min_temp, max_temp = match.groups()
                df.at[i, "min_temp"] = int(min_temp)
                df.at[i, "max_temp"] = int(max_temp)
            
            # x°C と y°C の形式になっている
            match = re.findall(r'(\d+)°C', text)
            if match:
                temp_list = [int(t) for t in match]
                df.at[i, "min_temp"] = min(temp_list)
                df.at[i, "max_temp"] = max(temp_list)
            
        if "°F" in text:    
            # x-y°F の形式になっている
            match = re.search(r'(\d+)-(\d+)°F', text)
            if match:
                min_temp, max_temp = match.groups()
                df.at[i, "min_temp"] = int(min_temp)
                df.at[i, "max_temp"] = int(max_temp)
                
            # x°F と y°F の形式になっている
            match = re.findall(r'(\d+)°F', text)
            if match:
                temp_list = [int(t) for t in match]
                df.at[i, "min_temp"] = min(temp_list)
                df.at[i, "max_temp"] = max(temp_list)
                
            # Convert Fahrenheit to Celsius
            if df.at[i, "min_temp"] != -1:
                df.at[i, "min_temp"] = (df.at[i, "min_temp"] - 32) * 5 / 9
            if df.at[i, "max_temp"] != -1:
                df.at[i, "max_temp"] = (df.at[i, "max_temp"] - 32) * 5 / 9
        
    df["dff_tmp"] = df["max_temp"] - df["min_temp"]
    return df
                

# %%

class joai_dataset(torch.utils.data.Dataset):
    def __init__(self, df, embedding, attention_mask, target_class,transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.embedding = embedding
        self.attention_mask = attention_mask
        self.target_class = target_class

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rgb_img_path = os.path.join(config.data_dir, "images", self.df.iloc[idx]['image_path_uuid'])
        gray_img_path = os.path.join(config.data_dir, "images_reverse", self.df.iloc[idx]['image_path_uuid'].replace("RGB", "Gray"))
        # それぞれつなげて 4ch にする
        rgb_img = Image.open(rgb_img_path).convert("RGB")
        gray_img = Image.open(gray_img_path).convert("L")
        gray_img = gray_img.resize((config.train_img_size, config.train_img_size))
        gray_img = np.array(gray_img)
        image = np.concatenate([np.array(rgb_img), gray_img[:, :, np.newaxis]], axis=2)
                    
        if self.transform:
            if self.is_test is False and np.random.rand() < 0.5:
                # 先に gridmask を行う
                # gridmask = image_augumentation.GridMask(p=0.5, num_grid=5) # 変更前
                gridmask = GridMask(p=0.5, num_grid=5) # 変更後
                params = gridmask.get_params_dependent_on_targets({'image': image})
                image = gridmask.apply(image.copy(), **params)
            image = self.transform(image=image)['image']
        
        num_feature_list = []
        for col in ["MQ8", "MQ5", "is_degrees_celsius", "min_temp", "max_temp", "dff_tmp"]:
            num_feature_list.append(float(self.df.iloc[idx][col]))
            
        num_feature = np.array(num_feature_list, dtype=np.float32)
        num_feature = torch.tensor(num_feature, dtype=torch.float32)
        
        embedding = torch.tensor(self.embedding[idx], dtype=torch.float32)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.float32)
        
        if self.is_test:
            return image, num_feature, embedding, attention_mask, -1
        
        else :
            label = self.df.iloc[idx]["Gas"] == self.target_class
            label = torch.tensor(label, dtype=torch.float32)
            # soft labe にする
            label = label * (1 - config.smoothing) + (1 - label) * config.smoothing
            return image, num_feature, embedding, attention_mask, label

# %%
name_to_label = {
    "Mixture": 0,
    "NoGas": 1,
    "Perfume": 2,
    "Smoke": 3
}

# %%
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


# %%
class joaiModel(nn.Module):
    def __init__(self, model_name, num_classes, numerical_feature_count=2):
        super(joaiModel, self).__init__()
        
        # 画像部分 
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, in_chans=4)
        self.model.global_pool = nn.Identity()
        self.gem = GeM()
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.fc_img = nn.Linear(self.model.num_features, 256)
        
        # 数値部分
        self.fc_num = nn.Linear(numerical_feature_count, 128)
        
        # テキスト部分
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_text = nn.Linear(768, 256)
        
        self.film_generator = nn.Sequential(
            nn.Linear(128+256, 256),
            nn.ReLU(),
            nn.Linear(256, 256*2),
        )
        
        self.fc_final = nn.Linear(256+256+128, 1)
        
    def forward(self, x, numerical_features, embedding, attention_mask):
        x = self.model(x)
        x = self.gem(x)
        x = x.view(x.size(0), -1)
        x = self.fc_img(x)
        
        numerical_features = self.fc_num(numerical_features)
        numerical_features = F.relu(numerical_features)
        
        embedding = embedding.transpose(1, 2)
        embedding = self.max_pool(embedding)
        embedding = embedding.squeeze(-1)
        embedding = self.fc_text(embedding)
        embedding = F.relu(embedding)
        
        film_input = torch.cat([embedding, numerical_features], dim=1)
        film_params = self.film_generator(film_input)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        image_features = x * (1 + gamma) + beta
        image_features = F.relu(image_features)
        
        combined_features = torch.cat([image_features, embedding, numerical_features], dim=1)
        combined_features = self.fc_final(combined_features)
        return combined_features
        

# %%
class joai_pl_model(pl.LightningModule):
    def __init__(self, model, len_train_loader):
        super(joai_pl_model, self).__init__()
        self.model = model
        self.criterion = config.criterion()
        self.do_wandb = config.do_wandb
        self.len_train_loader = len_train_loader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, numerical_features, embedding, attention_mask, labels = batch
        outputs = self.model(images, numerical_features, embedding, attention_mask)
        # 二値分類なのでoutputsは[batch_size, 1]の形状
        outputs = outputs.squeeze(-1)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, numerical_features, embedding, attention_mask, labels = batch
        if config.do_tta:
            image_0 = images.clone()
            image_1 = torch.flip(image_0, dims=[-1])
            image_2 = torch.flip(image_0, dims=[-2])
            image_3 = torch.flip(image_0, dims=[-1, -2])
            output_0 = self.model(image_0, numerical_features, embedding, attention_mask).squeeze(-1)
            output_1 = self.model(image_1, numerical_features, embedding, attention_mask).squeeze(-1)
            output_2 = self.model(image_2, numerical_features, embedding, attention_mask).squeeze(-1)
            output_3 = self.model(image_3, numerical_features, embedding, attention_mask).squeeze(-1)
            outputs = (output_0 + output_1 + output_2 + output_3) / 4
        else: 
            outputs = self.model(images, numerical_features, embedding, attention_mask).squeeze(-1)
            
        loss = self.criterion(outputs, labels)
        # 確率に変換して閾値0.5で二値化
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        hard_labels = (labels > 0.5).float()    
        
        # 二値分類用の評価指標
        val_f1 = f1_score(hard_labels.cpu().numpy(), preds.cpu().numpy(), average='binary')
        val_precision = precision_score(hard_labels.cpu().numpy(), preds.cpu().numpy(), average='binary')
        val_recall = recall_score(hard_labels.cpu().numpy(), preds.cpu().numpy(), average='binary')
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('val_f1', torch.tensor(val_f1, device=self.device), prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('val_precision', torch.tensor(val_precision, device=self.device), prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('val_recall', torch.tensor(val_recall, device=self.device), prog_bar=True, on_step=False, on_epoch=True, logger=True)
        
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels
        }
    
    def predict_step(self, batch, batch_idx):
        images, numerical_features, embedding, attention_mask, labels = batch
        if config.do_tta:
            image_0 = images.clone()
            image_1 = torch.flip(image_0, dims=[-1])
            image_2 = torch.flip(image_0, dims=[-2])
            image_3 = torch.flip(image_0, dims=[-1, -2])
            output_0 = self.model(image_0, numerical_features, embedding, attention_mask).squeeze(-1)
            output_1 = self.model(image_1, numerical_features, embedding, attention_mask).squeeze(-1)
            output_2 = self.model(image_2, numerical_features, embedding, attention_mask).squeeze(-1)
            output_3 = self.model(image_3, numerical_features, embedding, attention_mask).squeeze(-1)
            outputs = (output_0 + output_1 + output_2 + output_3) / 4
        else: 
            outputs = self.model(images, numerical_features, embedding, attention_mask).squeeze(-1)
            
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        return {"probs": probs, "preds": preds}
    
    def configure_optimizers(self):
        optimizer_for_pl = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)
        
        # Calculate total training steps and warmup steps
        num_training_steps = self.len_train_loader * config.epochs 
        num_warmup_steps = int(config.warmup_prop * num_training_steps)  # 10% of total steps for warmup
        scueduler_obj = transformers.get_cosine_schedule_with_warmup(
                optimizer_for_pl,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
        )
        # Create scheduler with proper parameters
        scheduler = {
            "scheduler": scueduler_obj,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer_for_pl], [scheduler]

# %%
def run_train_cv_pl(train, test):    
    train = add_degrees_celcius(train)
    test = add_degrees_celcius(test)
    train_embedding = np.load("../dataset/deberta_embeddings_train.npy")
    test_embedding = np.load("../dataset/deberta_embeddings_test.npy")
    train_attention_mask = np.load("../dataset/deberta_attention_masks_train.npy")
    test_attention_mask = np.load("../dataset/deberta_attention_masks_test.npy")
        
    train_pred = pd.DataFrame(None, index=train.index, columns=name_to_label.keys())
    test_pred = pd.DataFrame(None, index=test.index, columns=name_to_label.keys())
    train_pred[list(name_to_label.keys())] = 0
    test_pred[list(name_to_label.keys())] = 0
    
    for fold in range(config.fold):
        print(f"==========================fold {fold+1}==========================")
        if config.debug_one_fold and fold != 0:
            break
        print(f"Fold {fold+1} / {config.fold}")
        seed_everything(config.random_seed)
        
        train_df = train[train['fold'] != fold].reset_index(drop=True)
        val_df = train[train['fold'] == fold].reset_index(drop=True)
        test_df = test.copy()
        val_indices = train[train['fold'] == fold].index.tolist()
        processor = QuantileTransformer(
            n_quantiles=max(min(len(train_df)//10, 1000), 10),
            output_distribution="normal",
            subsample=int(1e9),
            random_state=config.random_seed
        )
        to_process = ["MQ8", "MQ5", "max_temp", "min_temp", "dff_tmp"]
        for col in to_process:
            train_df[col] = processor.fit_transform(train_df[[col]])
            val_df[col] = processor.transform(val_df[[col]])
            test_df[col] = processor.transform(test_df[[col]])
        
        train_tmp_embedding = train_embedding[train[train['fold'] != fold].index]
        val_tmp_embedding = train_embedding[train[train['fold'] == fold].index]
        train_tmp_attention_mask = train_attention_mask[train[train['fold'] != fold].index]
        val_tmp_attention_mask = train_attention_mask[train[train['fold'] == fold].index]
        
        for label_name in name_to_label.keys():
            train_dataset = joai_dataset(train_df, transform=train_transform, embedding=train_tmp_embedding, attention_mask=train_tmp_attention_mask, target_class=label_name)
            val_dataset = joai_dataset(val_df, transform=test_transform, embedding=val_tmp_embedding, attention_mask=val_tmp_attention_mask, target_class=label_name)
            test_dataset = joai_dataset(test_df, transform=test_transform, embedding=test_embedding, attention_mask=test_attention_mask, target_class=label_name, is_test=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
            len_train_loader = len(train_loader)    
            model = joaiModel(config.model, config.num_classes, numerical_feature_count=6)
            pl_model = joai_pl_model(model, len_train_loader)
        
        
            wandb_logger = None
            if config.do_wandb:
                wandb_logger = WandbLogger(
                    project="JOAI", 
                    group=config.exp_name, 
                    name=f"fold{fold}_{label_name}", 
                    config=class_dict, save_dir="logs"
                    )
                # wandb に画像を保存
                # 最初のバッチを保存
                first_batch = next(iter(train_loader))
                for i, image in enumerate(first_batch[0]):
                    wandb_logger.log_image(
                        key=f"train_image_{i}",
                        images=[wandb.Image(image.permute(1, 2, 0).cpu().numpy())],
                        caption=[f"train_image_{i}"]
                    )
                    if i== 16:
                        break
            
            tensorboard_logger = TensorBoardLogger("logs", name=f"{config.exp_name}_fold{fold}")
        
            checkpoint_callback = ModelCheckpoint(
                monitor='val_f1',
                dirpath=os.path.join(config.output_dir, f"{config.exp_name}_fold{fold}_{label_name}"),
                filename='best-checkpoint',
                save_top_k=1,
                mode='max'
            )

            early_stopping_callback = EarlyStopping(
                monitor='val_f1',
                patience=config.patience,
                mode='max'
            )

            trainer = pl.Trainer(
                max_epochs=config.epochs,
                accelerator="gpu",
                devices=1,
                logger=wandb_logger if config.do_wandb else tensorboard_logger,
                callbacks=[checkpoint_callback, early_stopping_callback],
                precision=config.precision,
                log_every_n_steps=10,
            )
        
            trainer.fit(pl_model, train_loader, val_loader)
            best_model_path = checkpoint_callback.best_model_path
            best_pl_model = joai_pl_model.load_from_checkpoint(
                best_model_path,
                model=model,
                len_train_loader=len_train_loader
            )
        
        
            val_preds_list = trainer.predict(best_pl_model, val_loader)
            val_probs = np.concatenate([x['probs'].cpu().numpy() for x in val_preds_list], axis=0)
            train_pred.loc[val_indices, label_name] += val_probs
            
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
            test_preds_list = trainer.predict(best_pl_model, test_loader)
            test_probs = np.concatenate([x['probs'].cpu().numpy() for x in test_preds_list], axis=0)
            test_pred.loc[:, label_name] += test_probs / config.fold
            del model, trainer, train_loader, val_loader, test_loader, best_pl_model, best_model_path
            if config.do_wandb:
                wandb.finish()
                del wandb_logger

            gc.collect()
            torch.cuda.empty_cache()
        if config.debug_one_fold:
            break
        
    return {
        "oof": train_pred.values,
        "predictions": test_pred.values
    }

# %%
def main():
    if config.only_infer:
        raise ValueError("Inference mode is not implemented yet.")
    else:
        train = pd.read_csv(os.path.join(config.data_dir, "train_folds_another_seed.csv"))
        test = pd.read_csv(os.path.join(config.data_dir, "test.csv"))
        name_to_label = {
            "Mixture": 0,
            "NoGas": 1,
            "Perfume": 2,
            "Smoke": 3
        }
        label_to_name = {v: k for k, v in name_to_label.items()}
        pred_dict = run_train_cv_pl(train, test)
        oof_preds = pred_dict["oof"]
        # 後処理として、 oof_preds の is_degrees_celsius が 1 だった場合に Perfume と Smoke のラベル以外の確率を 0 にする
        train_tmp = add_degrees_celcius(train)
        for i in range(len(oof_preds)):
            if train_tmp.iloc[i]["is_degrees_celsius"] == 1 and "°f" not in train_tmp.iloc[i]["Caption"].lower():
                oof_preds[i][0] = 0
                oof_preds[i][3] = 0
                
         
        test_preds = pred_dict["predictions"]
        test_tmp = add_degrees_celcius(test)
        for i in range(len(test_preds)):
            if test_tmp.iloc[i]["is_degrees_celsius"] == 1 and np.argmax(test_preds[i]) not in [1, 2]:
                print("there's a possibility of miss classification")
                print(i, test_preds[i])
            # Checking if it's in Celsius and doesn't contain Fahrenheit
            if test_tmp.iloc[i]["is_degrees_celsius"] == 1 and "°f" not in test_tmp.iloc[i]["Caption"].lower():
                test_preds[i][0] = 0
                test_preds[i][3] = 0
            
        print("oof_preds shape:", oof_preds.shape)
        print("test_preds shape:", test_preds.shape)
        
        for i, class_name in enumerate(label_to_name.values()):
            print(f"{class_name}: {i}")
            train[class_name] = oof_preds[:, i]
            
        train.to_csv(os.path.join(config.output_dir, f"{config.exp_name}_oof.csv"), index=False)
        
        for i, class_name in enumerate(label_to_name.values()):
            test[class_name] = test_preds[:, i]
            
        test.to_csv(os.path.join(config.output_dir, f"{config.exp_name}_test_probs.csv"), index=False)
        
        # oof_preds は予測の確率が格納された配列
        oof_preds = np.argmax(oof_preds, axis=1)
        print(f"valid_total_f1: {f1_score(train['Gas'].apply(lambda x: name_to_label[x]).values, oof_preds, average='weighted')}")
        test_preds = np.argmax(test_preds, axis=1)
        sample_submission = pd.read_csv(os.path.join(config.data_dir, "sample_submission.csv"))
        sample_submission['Gas'] = test_preds
        sample_submission['Gas'] = sample_submission['Gas'].map(label_to_name)
        sample_submission.to_csv(os.path.join(config.output_dir, f"{config.exp_name}_submission.csv"), index=False)
        print("Inference completed. Submission file saved.")
        
if __name__ == "__main__":
    main()


