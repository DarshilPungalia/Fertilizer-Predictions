from ingestion import LoadData
from eval import eval
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

cache = {}

class Ensemble:
    def __init__(self):
        self.models = {}
        self.data = LoadData()
        self.utils = eval()

        self.ensemble()
    
    @staticmethod
    def train(build_classifier, X, y, splits, model_name=None):
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)
        oof_preds = np.zeros((X.shape[0], len(np.unique(y))))  

        models = []
        for i, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), desc=f'Training {splits} {model_name}...', total=splits)):
            model = build_classifier()
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict_proba(X[val_idx])
            oof_preds[val_idx] = pred  
            models.append(model)

        return models, oof_preds
    
    @staticmethod
    def build_nn(input_shape, output_classes=7):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        class NeuralNetwork(nn.Module):
            def __init__(self, input_shape, output_classes=7):
                super(NeuralNetwork, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_shape, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.4),
                    
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    
                    nn.Linear(32, output_classes)
                )
                
            def forward(self, x):
                return self.network(x)
        
        model = NeuralNetwork(input_shape, output_classes).to(device)  
        return model

    
    def ensemble(self):
        x, y, self.x_test = self.data.get_preprocessed_data()

        xgb_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'multi:softprob',
            'num_class': 7,
            'random_state': 12
        } 

        xgb, oof_level_1 = self.train(lambda: XGBClassifier(**xgb_params),
                         x,
                         y,
                         splits=5,
                         model_name='XGBoost')
        
        self.models['xgb'] = xgb
        print(f'Level 1 OOF shape: {oof_level_1.shape}\n{oof_level_1[0]}')
        
        light_params = {
            'objective': 'multiclass',
            'boosting_type': 'gbdt',
            'device': 'gpu', 
            'verbosity': -1,
            'max_bin': 255,
            'num_class': 7
        }
        
        lgb, oof_level_2 = self.train(lambda: LGBMClassifier(**light_params),
                         oof_level_1,
                         y,
                         splits=3,
                         model_name='LightGBM')
        
        self.models['lgb'] = lgb
        print(f'Level 2 OOF shape: {oof_level_2.shape}\n{oof_level_2[0]}')

        model = self.build_nn(input_shape=oof_level_2.shape[1], output_classes=7)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.Tensor(oof_level_2).to(device)  
        y_tensor = torch.LongTensor(y).to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)  
        
        print(f"Level 2 OOF shape: {oof_level_2.shape}")
        print(f"Level 2 OOF stats: min={oof_level_2.min():.4f}, max={oof_level_2.max():.4f}, mean={oof_level_2.mean():.4f}")
        print(f"Labels shape: {y.shape}, unique labels: {len(np.unique(y))}")
        print(f"Label distribution: {np.bincount(y)}")
        
        if np.isnan(oof_level_2).any():
            print("WARNING: NaN values found in features!")
            oof_level_2 = np.nan_to_num(oof_level_2)
                
        X_train, X_val, y_train, y_val = train_test_split(
            oof_level_2, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_tensor = torch.Tensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.Tensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        print('Training Meta Model...')
        print(f'{"Epoch":>5} | {"Loss":>8} | {"Val Loss":>8} | {"Acc":>7} | {"MAP@3":>7} | {"LR":>8}')
        print('-' * 80)
                
        for epoch in range(50):  
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_true = y_val_tensor.cpu().numpy()
                
                accuracy = accuracy_score(val_true, val_predictions)
                map3 = self.utils.map3(val_true, val_outputs.cpu().numpy())
                
                avg_loss = epoch_loss / len(train_dataloader)
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f'{epoch+1:>5} | {avg_loss:>8.4f} | {val_loss:>8.4f} | {accuracy:>7.4f} | {map3:>7.4f} | {current_lr:>8.6f}')
                
                scheduler.step(val_loss)
                
        self.models['nn'] = model
    
    def get_submission(self, path):
        level_1 = self.models['xgb']
        level_2 = self.models['lgb']
        meta = self.models['nn']

        print('Predicting from Level 1 (XGBoost ensemble)...')
        level_1_preds = np.zeros((self.x_test.shape[0], 7))  
        
        for xgb_model in level_1:
            pred = xgb_model.predict_proba(self.x_test)
            level_1_preds += pred
        
        level_1_preds /= len(level_1)  
        
        print('Predicting from Level 2 (LightGBM ensemble)...')
        level_2_preds = np.zeros((self.x_test.shape[0], 7))
        
        for lgb_model in level_2:
            pred = lgb_model.predict_proba(level_1_preds)
            level_2_preds += pred
        
        level_2_preds /= len(level_2)  
        
        print('Predicting from Meta Model (Neural Network)...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        meta.eval()
        
        with torch.no_grad():
            level_2_tensor = torch.Tensor(level_2_preds).to(device)
            meta_outputs = meta(level_2_tensor)
            meta_probs = torch.softmax(meta_outputs, dim=1)
            meta_preds = meta_probs.cpu().numpy()
        
        print('Generating top-3 predictions...')
        top3_fertilizers = self.utils.top3(meta_preds, encoder=self.data.get_label_encoder())
        
        submission = pd.DataFrame({
            'id': self.data.get_test_data()['id'],
            'Fertilizer Name': top3_fertilizers
        })
        
        print(f'Saving submission to {path}...')
        submission.to_csv(path, index=False)
        print(f'Submission saved successfully with shape: {submission.shape}')
                
        return submission
    

if __name__ == '__main__':
    model = Ensemble()
    model.get_submission('submissions/Ensemble2.csv')