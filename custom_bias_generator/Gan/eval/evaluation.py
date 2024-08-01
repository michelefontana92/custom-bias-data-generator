import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings

warnings.filterwarnings("ignore")

def stat_sim(real_path,fake_path,cat_cols=None):
    
    Stat_dict={}
    
    real = pd.read_csv(real_path)
    fake = pd.read_csv(fake_path)

    really = real.copy()
    fakey = fake.copy()

    real_corr = associations(real, nominal_columns=cat_cols,compute_only=True)['corr']

    fake_corr = associations(fake, nominal_columns=cat_cols,compute_only=True)['corr']

    corr_dist = np.linalg.norm(real_corr - fake_corr)
    
    cat_stat = []
    num_stat = []
    
    for column in real.columns:
        
        if column in cat_cols:

            real_pdf=(really[column].value_counts()/really[column].value_counts().sum())
            fake_pdf=(fakey[column].value_counts()/fakey[column].value_counts().sum())
            categories = (fakey[column].value_counts()/fakey[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)
            
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                real_pdf_values.append(real_pdf[i])
                fake_pdf_values.append(fake_pdf[i])
            
            if len(real_pdf)!=len(fake_pdf):
                zero_cats = set(really[column].value_counts().keys())-set(fakey[column].value_counts().keys())
                for z in zero_cats:
                    real_pdf_values.append(real_pdf[z])
                    fake_pdf_values.append(0)
            Stat_dict[column]=(distance.jensenshannon(real_pdf_values,fake_pdf_values, 2.0))
            cat_stat.append(Stat_dict[column])        
        else:
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1,1))
            l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
            l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            Stat_dict[column]= (wasserstein_distance(l1,l2))
            num_stat.append(Stat_dict[column])

    return [np.mean(num_stat),np.mean(cat_stat),corr_dist]
