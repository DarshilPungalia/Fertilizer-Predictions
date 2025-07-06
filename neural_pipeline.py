import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import optuna
from ingestion import LoadData
from eval import eval
import pandas as pd
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings(action='ignore')

class NeuralPipeline:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.data = LoadData()
        self.utils = eval()

        self.Neural()

    @staticmethod
    def build_model(trial):
        model = tf.keras.models.Sequential()
        
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "selu", "tanh"])

        layers = trial.suggest_int("no_of_layers", 2, 6)

        for i in range(layers):
            units = trial.suggest_int(f'units_{i+1}', 64, 512, step=32)
            model.add(tf.keras.layers.Dense(units, activation=activation))
            
            if trial.suggest_categorical(f'batchnorm_{i+1}', [True, False]):
                model.add(tf.keras.layers.BatchNormalization())

            dropout = trial.suggest_float(f'droprate_{i+1}', 0.1, 0.5, log=True)
            model.add(tf.keras.layers.Dropout(dropout))
        
        model.add(tf.keras.layers.Dense(7, activation='softmax'))

        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model

    
    def objective(self, trial):
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4)]

        for train_idx, val_idx in skf.split(self.x_sub, self.y_sub):
            x_train, x_val = self.x_sub[train_idx], self.x_sub[val_idx]
            y_train, y_val = self.y_sub[train_idx], self.y_sub[val_idx]
            
            model = self.build_model(trial)
            model.fit(x_train, 
                      y_train,
                      validation_data=(x_val, y_val),
                      epochs=10, 
                      batch_size=2048,
                      callbacks=callbacks,
                      verbose=0)
            
            pred = model.predict(x_val)
            score = self.utils.map3(y_val, pred, encoder=self.data.get_label_encoder())
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_hp(self, n_trials=25):
        
        study = optuna.create_study(direction='maximize')
        print('Optimizing HyperParameter...')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        print(f'Best trial was with Params: {study.best_params}')
        
        self.best_trial = study.best_trial
    
    def train(self, x, y):
        self.model = self.build_model(self.best_trial)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=12)
        trainData = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1024).prefetch(tf.data.AUTOTUNE)
        valData = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(512).prefetch(tf.data.AUTOTUNE)
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4)]
        print('Starting Model Training...')
        self.model.fit(trainData.repeat(),
                       validation_data=valData.repeat(),
                       epochs=32,
                       steps_per_epoch=16,
                       validation_steps=10,
                       callbacks=callbacks)

        return self.model
        
    def Neural(self):
        self.x_train, self.y_train, self.x_test = self.data.get_preprocessed_data()
        self.x_sub, self.y_sub = self.x_train[:50000], self.y_train[:50000]

        self.optimize_hp()
        self.model = self.train(self.x_train, self.y_train)

    def get_submission(self, path):
        print('Generating model predictions...')
        preds = self.model.predict(self.x_test)

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
    pipeline = NeuralPipeline()
    pipeline.get_submission(r'submissions\NN.csv')
        