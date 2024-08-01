import pytest
from custom_bias_generator import BiasInjector
import numpy as np

@pytest.fixture
def bias_injector():
    return BiasInjector(data_path="data/adult.csv",
                        target_label="income",
                        positive_label_value=">50K"
                        )

def test_bias_injector_init(bias_injector):
    assert bias_injector is not None

def test_bias_injection(bias_injector):
    prior_y = 0.3
    n_samples = 30000
    sensitive_attribute_list = ['gender']
    pmf_0 = [(['Male'],0.2),(['Female'],0.8)]
    pmf_1 = [(['Male'],0.5),(['Female'],0.5)]
    pmf = {'<=50K':pmf_0,'>50K':pmf_1}
    df = bias_injector.inject_bias(prior_y,n_samples,
                                   sensitive_attribute_list,
                                   pmf)
    assert df.shape[0] == n_samples
    columns = [bias_injector.target_label] + sensitive_attribute_list
    
    print('After injection:\n',round(df[columns].value_counts(normalize=True),3))
    
    df_0 = df[df[bias_injector.target_label]=='<=50K']
    df_1 = df[df[bias_injector.target_label]=='>50K']
    
    print('Negative:\n',round(df_0[sensitive_attribute_list].value_counts(normalize=True),2))
    print('Positive:\n',round(df_1[sensitive_attribute_list].value_counts(normalize=True),2))
    #assert np.isclose(dict(round(df[sensitive_attribute].value_counts(normalize=True),3))['Male'],0.2,atol=0.1)
    #assert np.isclose(dict(round(df[sensitive_attribute].value_counts(normalize=True),3))['Female'],0.8,atol=0.1)


