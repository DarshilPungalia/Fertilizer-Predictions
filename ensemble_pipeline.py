import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

cache = {}

class Ensemble:
    def __init__(self):
        self.train_data = pd.read_csv(r'data\train.csv')
        self.test_data = pd.read_csv(r'data\test.csv')
        self.X = self.train_data.drop(columns=['Fertilizer Name', 'id'])
        self.Y = self.train_data['Fertilizer Name']
        self.models = {}
        self.label_encoder = None
        self.preprocessor_pipeline = None

        self.ensemble()

    @staticmethod
    def get_transformer(num_columns=None, cat_columns=None, for_label: bool = False):
        if for_label:
            return LabelEncoder()
        
        scaler = MinMaxScaler()
        encoder = OneHotEncoder(handle_unknown='ignore')

        num_pipeline = Pipeline(steps=[
            ('normalization', scaler)
        ])

        cat_pipeline = Pipeline(steps=[
            ('encoder', encoder)
        ])

        combined = ColumnTransformer([
            ('nums', num_pipeline, num_columns),
            ('cat', cat_pipeline, cat_columns)
        ], remainder='passthrough')

        return Pipeline(steps=[
            ('preprocessor', combined)
        ])
    
    def preprocessor(self):
        x = self.X
        y = self.Y

        num_columns = x.select_dtypes(include=['int64', 'float64']).columns
        cat_columns = x.select_dtypes(include=['object', 'category']).columns

        pipeline = self.get_transformer(num_columns=num_columns, cat_columns=cat_columns)
        label_pipeline = self.get_transformer(for_label=True)

        print('Transforming Training Data...')
        values = pipeline.fit_transform(x)
        labels = label_pipeline.fit_transform(y)
        print(f'shape of X: {values.shape}, shape of Y: {labels.shape}')

        print('Transforming Test Data...')
        test_values = pipeline.transform(self.test_data.drop(columns=['id']))

        # Store pipelines for later use
        self.preprocessor_pipeline = pipeline
        self.label_encoder = label_pipeline

        return values, labels, test_values
    
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
        x, y, self.x_test = self.preprocessor()

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
                map3 = self.map3(val_true, val_outputs.cpu().numpy())
                
                avg_loss = epoch_loss / len(train_dataloader)
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f'{epoch+1:>5} | {avg_loss:>8.4f} | {val_loss:>8.4f} | {accuracy:>7.4f} | {map3:>7.4f} | {current_lr:>8.6f}')
                
                scheduler.step(val_loss)
                
        self.models['nn'] = model

    def predict_top3(self, probabilities, get_indices: bool = False):
        top3_predictions = []
        top3_indices = []
        
        for prob_row in probabilities:
            top3_indice = np.argsort(prob_row)[-3:][::-1]
            top3_indices.append(top3_indice)

            if not get_indices:
                top3_labels = self.label_encoder.inverse_transform(top3_indice)
                
                top3_string = " ".join(top3_labels)
                top3_predictions.append(top3_string)
        
        return top3_indices if get_indices else top3_predictions
    
    def map3(self, y_true, y_predicted):
        y_predicted = self.predict_top3(y_predicted, get_indices=True)
        scores = []

        for preds, true in zip(y_predicted, y_true):
            position_scores = {preds[0]: 1.0, preds[1]: 1/2, preds[2]: 1/3}
            scores.append(position_scores.get(true, 0.0))
        
        return np.mean(scores)
    
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
        top3_fertilizers = self.predict_top3(meta_preds)
        
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Fertilizer Name': top3_fertilizers
        })
        
        print(f'Saving submission to {path}...')
        submission.to_csv(path, index=False)
        print(f'Submission saved successfully with shape: {submission.shape}')
                
        return submission
    

if __name__ == '__main__':
    model = Ensemble()
    model.get_submission('submissions/Ensemble2.csv')