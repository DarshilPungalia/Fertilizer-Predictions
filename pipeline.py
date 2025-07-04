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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

cache = {}

class Fertilizer:
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
        
        # Define model with  precision (float64)
        class NeuralNetwork(nn.Module):
            def __init__(self, input_shape, output_classes=7):
                super(NeuralNetwork, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_shape, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.4),
                                        
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
        print(f'Level 1 OOF shape: {oof_level_1.shape}')
        
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
        print(f'Level 2 OOF shape: {oof_level_2.shape}')

        model = self.build_nn(input_shape=oof_level_2.shape[1], output_classes=7)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.Tensor(oof_level_2).to(device)  
        y_tensor = torch.LongTensor(y).to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=2046, shuffle=True)  
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        X_train, X_val, y_train, y_val = train_test_split(
            oof_level_2, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_tensor = torch.Tensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.Tensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        print('Training Meta Model...')
        print(f'{"Epoch":>5} | {"Loss":>8} | {"Acc":>7} | {"Prec":>7} | {"Rec":>7} | {"F1":>7}')
        print('-' * 55)
        
        best_f1 = 0.0
        for epoch in range(100):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_true = y_val_tensor.cpu().numpy()
                
                accuracy = accuracy_score(val_true, val_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_true, val_predictions, average='weighted', zero_division=0
                )
                
                avg_loss = epoch_loss / len(train_dataloader)
                
                print(f'{epoch+1:>5} | {avg_loss:>8.4f} | {accuracy:>7.4f} | {precision:>7.4f} | {recall:>7.4f} | {f1:>7.4f}')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_state = model.state_dict().copy()
        
        model.load_state_dict(best_model_state)
        print(f'\nBest F1 Score: {best_f1:.4f}')
        
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
        
        final_predictions = np.argmax(meta_preds, axis=1)
        
        final_labels = self.label_encoder.inverse_transform(final_predictions)
        
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Fertilizer Name': final_labels
        })
        
        print(f'Saving submission to {path}...')
        submission.to_csv(path, index=False)
        print(f'Submission saved successfully with shape: {submission.shape}')
        
        return submission
