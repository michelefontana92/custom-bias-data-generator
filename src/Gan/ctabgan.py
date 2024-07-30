"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
from .pipeline.data_preparation import DataPrep
from .synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 raw_csv_path,
                 categorical_columns, 
                 mixed_columns,
                 general_columns,
                 problem_type,
                 **kwargs):

        self.__name__ = 'CTABGAN'
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.synthesizer = CTABGANSynthesizer(epochs=self.num_epochs)
        self.raw_df = pd.read_csv(raw_csv_path)
        
        self.categorical_columns = categorical_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.problem_type = problem_type
        assert isinstance(self.mixed_columns,dict), "mixed_columns should be a dictionary"
        assert isinstance(self.general_columns,list), "general_columns should be a list"
        assert isinstance(self.categorical_columns,list), "categorical_columns should be a list"
        assert isinstance(self.problem_type,dict), "problem_type should be a dictionary"
        assert list(self.problem_type.keys())[0] in ["Classification","Regression"], "problem_type should have a key 'type' with value 'Classification' or 'Regression'"
    
        self.log_columns = kwargs.get('log_columns', [])
        self.non_categorical_columns = kwargs.get('non_categorical_columns', [])
        self.integer_columns = kwargs.get('integer_columns', [])
        self.test_ratio = kwargs.get('test_ratio', 0.2)

        
                
    def fit(self):    
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,
                                  self.log_columns,self.mixed_columns,
                                  self.general_columns,
                                  self.non_categorical_columns,
                                  self.integer_columns,
                                  self.problem_type,
                                  self.test_ratio)
        
        self.synthesizer.fit(train_data=self.data_prep.df, 
                             categorical = self.data_prep.column_types["categorical"],
                             mixed = self.data_prep.column_types["mixed"],
                             general = self.data_prep.column_types["general"],
                             non_categorical = self.data_prep.column_types["non_categorical"],
                             type=self.problem_type)
       
    def generate_samples(self,num_samples):
        sample = self.synthesizer.sample(num_samples) 
        sample_df = self.data_prep.inverse_prep(sample)
        return sample_df
