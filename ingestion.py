import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class LoadData:
    def __init__(self):
        self.train_data = pd.read_csv(r'data\train.csv')
        self.test_data = pd.read_csv(r'data\test.csv')
        self.X = self.train_data.drop(columns=['Fertilizer Name', 'id'])
        self.Y = self.train_data['Fertilizer Name']
        self.label_encoder = None
        self.preprocessor_pipeline = None

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
    
    def get_preprocessed_data(self):
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
    
    def get_encoder(self):
        return self.preprocessor_pipeline
    
    def get_label_encoder(self):
        return self.label_encoder
    
    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data