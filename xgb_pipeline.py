import numpy as np
from xgboost  import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import optuna
from ingestion import LoadData
from eval import eval
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')

class XGBoostPipeline:
    def __init__(self):
        self.config = {'tree_method': 'hist',
                        'device': 'cuda',
                        'objective': 'multi:softprob',
                        'num_class': 7,
                        'random_state': 12
                        }
        self.model = None
        self.best_params = None
        self.data = LoadData()
        self.utils = eval()

        self.xgb()
    
    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 128, 1024),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(self.x_sub, self.y_sub):
            x_train, x_val = self.x_sub[train_idx], self.x_sub[val_idx]
            y_train, y_val = self.y_sub[train_idx], self.y_sub[val_idx]
            
            model = XGBClassifier(**self.config ,**params)
            model.fit(x_train, y_train, verbose=False)
            
            pred = model.predict_proba(x_val)
            score = self.utils.map3(y_val, pred, encoder=self.data.get_label_encoder())
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_hp(self, n_trials=25):
        
        study = optuna.create_study(direction='maximize')
        print('Optimizing HyperParameter...')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        print(f'Best trial was with Params: {study.best_params}')
        
        self.best_params = study.best_params
    
    def train(self, x, y):
        self.model = XGBClassifier(**self.config, **self.best_params)
        print('Starting Model Training...')
        self.model.fit(x, y, verbose=True)

        return self.model
        
    def xgb(self):
        self.x_train, self.y_train, self.x_test = self.data.get_preprocessed_data()
        self.x_sub, self.y_sub = self.x_train[:50000], self.y_train[:50000]

        self.optimize_hp()
        self.model = self.train(self.x_train, self.y_train)

    def get_submission(self, path):
        print('Generating model predictions...')
        preds = self.model.predict_proba(self.x_test)

        print('Generating top-3 predictions...')
        top3_fertilizers = self.utils.top3(preds, encoder=self.data.get_label_encoder())
        
        submission = pd.DataFrame({
            'id': self.data.get_test_data()['id'],
            'Fertilizer Name': top3_fertilizers
        })
        
        print(f'Saving submission to {path}...')
        submission.to_csv(path, index=False)
        print(f'Submission saved successfully with shape: {submission.shape}')
                
        return submission
    
if __name__ == "__main__":
    pipeline = XGBoostPipeline()
    pipeline.get_submission(r'submissions\XGB.csv')
        