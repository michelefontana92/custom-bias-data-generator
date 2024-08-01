import pytest
from custom_bias_generator import CTABGAN, stat_sim
import pandas as pd
import numpy as np
import os

@pytest.fixture
def gan():
    real_path = "data/king.csv"
    gan =  CTABGAN(raw_csv_path = real_path,
    categorical_columns = ['bedrooms', "floors", 'waterfront', 'view', 'condition', 'grade','zipcode'],   
    mixed_columns= {"sqft_basement":[0.0],  "yr_renovated":[0.0]},
    general_columns= ["bathrooms", "sqft_living", "sqft_above", "yr_built", "long", "sqft_living15"],
    problem_type= {"Regression": "price"},
    num_epochs=20
    ) 
    return gan

@pytest.fixture
def adult_gan():
    real_path = "data/adult.csv"
    adult_gan =  CTABGAN(raw_csv_path = real_path,
                  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                  mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                  general_columns = ['age'],
                  integer_columns = ['age','capital-gain', 'capital-loss','hours-per-week'],
                  problem_type= {'Classification': 'income'},
                  num_epochs=1
    ) 
    return adult_gan

  
@pytest.fixture(scope='session')
def teardown():
    os.remove("data/adult_synthetic.csv")
    os.remove("data/adult_stat_results.csv")
    os.remove("data/adult_gan.pkl")

def test_gan_initialization(gan):
    assert gan is not None
    assert isinstance(gan, CTABGAN)
    assert gan.raw_df is not None
    

def test_adult_gan_training(adult_gan):
   
   adult_gan.fit()
   assert adult_gan.synthesizer is not None
   df_synthetic = adult_gan.generate_samples(10)
   assert df_synthetic is not None
   assert len(df_synthetic) == 10
   assert len(df_synthetic.columns) == len(adult_gan.raw_df.columns)
   df_synthetic.to_csv("data/adult_synthetic.csv", index=False)
   
   real_path = 'data/adult.csv'
   fake_path = 'data/adult_synthetic.csv'
   adult_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
   stat_res_avg = []
   
   stat_res = stat_sim(real_path,fake_path,adult_categorical)
   stat_res_avg.append(stat_res)

   stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
   stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
   stat_results.to_csv("data/adult_stat_results.csv", index=False)

def test_gan_save_load(adult_gan):
   adult_gan.fit()
   adult_gan.save("data/adult_gan.pkl")
   gan = CTABGAN.load("data/adult_gan.pkl")
   assert gan is not None
   assert isinstance(gan, CTABGAN)
   gan.generate_samples(10)
 

# Execute teardown after all tests
def test_teardown(teardown):
    pass