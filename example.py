from custom_bias_generator import CTABGAN, BiasInjector,stat_sim
import pandas as pd 
import warnings 

warnings.filterwarnings("ignore")

# Load the "adult" dataset
data_path = 'data/adult.csv'
fake_data_path = 'data/adult_synthetic.csv'
# Specify the sensitive attribute and the target attribute
sensitive_attribute_list = ['gender','race']

target_attribute = 'income'

# List of all the categorical attributes in the dataset
categorical_columns = ['workclass', 'education', 'marital-status', 
                       'occupation', 'relationship', 'race', 
                       'gender', 'native-country', 'income']

# List of all the integer attributes in the dataset. 
# This says to CTABGAN  that these columns are integer and should be treated as such 
integer_columns = ['age','capital-gain', 'capital-loss','hours-per-week']

# List of all the mixed attributes in the dataset.
# This are all the columns that are not categorical or integer.
# In other words, they are:
# - continuous attributes with missing values (represented with a particular value, in this case 0.0)
# - categorical attributes with missing values
# - attributes with both categorical and continuous values
# In the list we have to specify what are the non-continuous values.
mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]}

#These columns have to be encoded with a "GeneralEncoder". 
# Useful if we have Gaussian-distributed columns to produce better samples. 
general_columns = ['age']

# We say that the problem is a classification problem 
# and the target attribute is "income"
problem_type = {'Classification': 'income'}

# Number of epochs for the training of the GAN
num_epochs = 100

# Number of synthetic samples to generate
num_fake_samples = 50000

#------------------------------------------------------------
# NOTE:
# IMPORTANT TO SPECIFY CORRECTLY THE CATEGORICAL COLUMNS 
# AND THE MIXED COLUMNS. The other columns are not so crucial
# -----------------------------------------------------------

# Train the CTABGAN model
ctgan = CTABGAN(raw_csv_path = data_path,
                  categorical_columns = categorical_columns, 
                  mixed_columns= mixed_columns,
                  general_columns = general_columns,
                  integer_columns = integer_columns,
                  problem_type= problem_type,
                  num_epochs=num_epochs
)
ctgan.fit()
ctgan.save('ctgan.pkl')
# Load the saved CTABGAN model, if we have already trained it
ctgan:CTABGAN = CTABGAN.load('ctgan.pkl')

# Generate synthetic data
synthetic_data = ctgan.generate_samples(num_fake_samples)
synthetic_data.to_csv(fake_data_path, index=False)

# Evaluate the synthetic data, in terms of distance of the distributions 
# from the real data. 
evaluations = stat_sim(data_path, fake_data_path, categorical_columns)
print(evaluations)

# Inject Fairness problems into the synthetic data

# Instantiate the BiasInjector
bias_injector = BiasInjector(
    data_path=fake_data_path,
    target_label=target_attribute,
    positive_label_value='>50K'
)

# Specify the conditional probability mass function for the sensitive attributes.
# For example, we say that we want that 
# - Pr(gender=Male, race=White | income='<=50K') = 0.2
# - Pr(gender=Male, race=White | income='>50K') = 0.6
pmf = {'<=50K':[(['Male','White'],0.2),
                (['Male','Black'],0.1),
                (['Female','White'],0.3),
                (['Female','Black'],0.4)
                ],
        '>50K':[(['Male','White'],0.6),
                (['Male','Black'],0.2),
                (['Female','White'],0.1),
                (['Female','Black'],0.1)
                ],   
        }

# Prior probability of class >50K.
prior_y = 0.4
# Number of records in the final dataset
num_samples = 10000

# Build the biased dataset with the specified pmf
biased_data = bias_injector.inject_bias(prior_y=prior_y,
                                        n_samples=num_samples,
                                        sensitive_attribute_list=sensitive_attribute_list,
                                        pmf_dict=pmf)

# Save the biased dataset to a file
biased_data.to_csv('data/biased_data.csv', index=False)
columns = [target_attribute] + sensitive_attribute_list
print('\nDistribution of the class <=50K\n:',
      round(biased_data[biased_data[target_attribute]=='<=50K'][columns].value_counts(normalize=True),2))
print('\n\nDistribution of the class >50K\n:',
      round(biased_data[biased_data[target_attribute]=='>50K'][columns].value_counts(normalize=True),2))