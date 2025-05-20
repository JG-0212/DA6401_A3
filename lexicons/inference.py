import argparse

import pandas as pd
from tensorflow import keras

from vanilla import Char2CharModel
from attention import Char2CharModelAttention

def get_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("-a","--attention", action = "store_true") 
    args = parser.parse_args()  
    return args

if __name__ == '__main__':
    test_file = "lexicons/ta.translit.sampled.test.tsv"
    test_data = pd.read_csv (test_file, header=None, sep='\t')
    args = get_args()
    print("Evaluation started")
    if args.attention == False:
        agent = Char2CharModel()
        agent.model = keras.models.load_model("s2s_model.keras")
        agent.predictor_setup()
        _,acc = agent.evaluate(test_data.iloc[:,0], test_data.iloc[:,1])
        print(f"Accuracy on test set : {acc}")
    else:
        agent = Char2CharModelAttention()
        agent.model = keras.models.load_model("s2s_model.keras")
        agent.predictor_setup()
        _,acc = agent.evaluate(test_data.iloc[:,0], test_data.iloc[:,1])
        print(f"Accuracy on test set : {acc}")
