"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
from .pipeline.data_preparation import DataPrep
from .synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings
import pickle
import os 
warnings.filterwarnings("ignore")


class CTABGAN:
   
    def __init__(self,
                 raw_csv_path,
                 categorical_columns, 
                 mixed_columns,
                 general_columns,
                 problem_type,
                 **kwargs):
        
        """
        CTABGAN (Conditional Table-based Generative Adversarial Network) class for generating synthetic data.

        :param raw_csv_path: The file path of the raw CSV data.
        :type raw_csv_path: str
        :param categorical_columns: List of column names that are categorical.
        :type categorical_columns: list
        :param mixed_columns: Dictionary where the keys are column names and the values are the corresponding mixed types.
        :type mixed_columns: dict
        :param problem_type: Dictionary specifying the problem type. The key can be either 'Classification' or 'Regression'. The value is the name of the target variable.
        :type problem_type: dict
        :param **kwargs: Additional keyword arguments.
        :type **kwargs: dict
        :keyword num_epochs: Number of epochs for training the synthesizer (default: 10).
        :type num_epochs: int
        :keyword general_columns: List of column names that are general types (default: []).
        :type general_columns: list
        :keyword log_columns: List of column names that are log-transformed (default: []).
        :type log_columns: list
        :keyword non_categorical_columns: List of column names that are non-categorical types (default: []).
        :type non_categorical_columns: list
        :keyword integer_columns: List of column names that are integer types (default: []).
        :type integer_columns: list
        :keyword test_ratio: Ratio of test data to split from the raw data (default: 0.2).
        :type test_ratio: float
        """

        self.__name__ = 'CTABGAN'
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.synthesizer = CTABGANSynthesizer(epochs=self.num_epochs)
        self.raw_df = pd.read_csv(raw_csv_path)
        
        self.categorical_columns = categorical_columns
        self.mixed_columns = mixed_columns
        
        self.problem_type = problem_type
        assert isinstance(self.mixed_columns,dict), "mixed_columns should be a dictionary"
        assert isinstance(self.categorical_columns,list), "categorical_columns should be a list"
        assert isinstance(self.problem_type,dict), "problem_type should be a dictionary"
        assert list(self.problem_type.keys())[0] in ["Classification","Regression"], "problem_type should have a key 'type' with value 'Classification' or 'Regression'"
    
        self.log_columns = kwargs.get('log_columns', [])
        self.non_categorical_columns = kwargs.get('non_categorical_columns', [])
        self.integer_columns = kwargs.get('integer_columns', [])
        self.test_ratio = kwargs.get('test_ratio', 0.2)
        self.general_columns = kwargs.get('general_columns', [])
        
                
    def fit(self):    
        """
        Fit the CTABGAN model by performing data preprocessing and training the synthesizer.
        """
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
        """
        Generate synthetic samples using the trained synthesizer.

        :param num_samples: Number of synthetic samples to generate.
        :type num_samples: int

        :return: DataFrame containing the generated synthetic samples.
        :rtype: pandas.DataFrame
        """
        sample = self.synthesizer.sample(num_samples) 
        sample_df = self.data_prep.inverse_prep(sample)
        return sample_df

    def save(self, path):
        """
        Save the CTABGAN instance to a file.

        :param path: The file path to save the CTABGAN instance.
        :type path: str
        """
        dir_name = os.path.dirname(path)
        if dir_name != '' and not os.path.exists(dir_name):
            os.makedirs(dir_name)
       
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """
        Load a saved CTABGAN instance from a file.

        :param path: The file path to load the CTABGAN instance from.
        :type path: str

        :return: The loaded CTABGAN instance.
        :rtype: CTABGAN
        """
        with open(path, 'rb') as f:
            return pickle.load(f)