import numpy as np 
import pandas as pd


class BiasInjector: 
    def __init__(self,data_path,target_label,
                 positive_label_value
                 ):
        """
        Initialize the BiasInjector class.

        :param data_path: The path to the data file.
        :type data_path: str
        :param target_label: The label of the target variable.
        :type target_label: str
        :param positive_label_value: The value of the positive label.
        :type positive_label_value: str
        """
        self.df = pd.read_csv(data_path)
        self.target_label = target_label
        self.positive_label_value = positive_label_value
        
    def _data_sampling_step(self,n_samples,y_prior):
        """
        Perform the data sampling step.

        :param n_samples: The number of samples to generate.
        :type n_samples: int
        :param y_prior: The prior probability of the target variable being positive.
        :type y_prior: float

        :return: The sampled biased data.
        :rtype: pd.DataFrame
        """
        n_target_positive=min(int(n_samples*y_prior),len(self.df[self.df[self.target_label]==self.positive_label_value]))
        n_target_negative=n_samples-n_target_positive
        data_biased_positive = self.df[self.df[self.target_label]==self.positive_label_value].sample(n_target_positive)   
        data_biased_negative = self.df[self.df[self.target_label]!=self.positive_label_value].sample(n_target_negative)
        data_biased = pd.concat([data_biased_positive,data_biased_negative],axis=0).sample(frac=1).reset_index().drop(columns=['index'],axis=1)
        return data_biased
    
    def _data_mutation_step(self,data_biased:pd.DataFrame,
                            sensitive_attribute_list:list[str],
                            target_value:str,
                            pmf:list[tuple]):
        """
        Perform the data mutation step.

        :param data_biased: The biased data.
        :type data_biased: pd.DataFrame
        :param sensitive_attribute_list: The list of sensitive attributes.
        :type sensitive_attribute_list: list[str]
        :param target_value: The value of the target variable.
        :type target_value: str
        :param pmf: The probability mass function for sampling sensitive attributes.
        :type pmf: list[tuple]

        :return: The mutated data.
        :rtype: pd.DataFrame
        """
        def _sample(pmf):
            p = np.random.uniform(0,1)
            total = np.sum([prob for _,prob in pmf])
            assert np.isclose(total,1,atol=1e-5),f"Total probability is not 1.0 but {total}"
            
            total = 0
            for value,prob in pmf:
                if p <= prob +total:
                    return value    
                total += prob
        
        data_mutated = data_biased[data_biased[self.target_label]==target_value].copy()
        data_mutated[sensitive_attribute_list] = data_mutated[sensitive_attribute_list].apply(lambda _: _sample(pmf),
                                                                                              axis=1,
                                                                                              result_type='expand')
        return data_mutated
    
    def inject_bias(self,prior_y:float,
                    n_samples:int,
                    sensitive_attribute_list:list[str],
                    pmf_dict:dict[str,list[tuple]]):
        """
        Inject bias into the data.

        :param prior_y: The prior probability of the target positive class.
        :type prior_y: float
        :param n_samples: The number of samples in the final dataset.
        :type n_samples: int
        :param sensitive_attribute_list: The list of sensitive attributes.
        :type sensitive_attribute_list: list[str]
        :param pmf_dict: The dictionary of target values and their corresponding probability mass functions. For example: 
        >>> pmf_0 = [(['Male'],0.2),(['Female'],0.8)]
        >>> pmf_1 = [(['Male'],0.5),(['Female'],0.5)]
        >>> pmf_dict = {'<=50K':pmf_0,'>50K':pmf_1}
        
        :type pmf_dict: dict[str,list[tuple]]

        :return: The mutated data with injected bias.
        :rtype: pandas.DataFrame
        """
        data_biased = self._data_sampling_step(n_samples,prior_y)
        
        data_mutated = None
        for target_value,pmf in pmf_dict.items():
            data = self._data_mutation_step(data_biased,
                                            sensitive_attribute_list,
                                            target_value,
                                            pmf)
            if data_mutated is None:
                data_mutated = data
            else:
                data_mutated = pd.concat([data_mutated,data],axis=0).sample(frac=1).reset_index().drop(columns=['index'],axis=1)
        data_mutated = data_mutated.sample(frac=1).reset_index().drop(columns=['index'],axis=1)
        return data_mutated
    
