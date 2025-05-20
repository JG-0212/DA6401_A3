import argparse

import pandas as pd

from vanilla import Char2CharModel
from attention import Char2CharModelAttention

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs", default = 20, type=int)
    parser.add_argument("-b","--batch_size", default = 64, type=int)  
    parser.add_argument("-nehl","--num_encoder_layers", default = 2, type=int)
    parser.add_argument("-ndhl","--num_decoder_layers", default = 2, type=int)
    parser.add_argument("-d","--dropout", default = 0, type=float)
    parser.add_argument("-sz","--hidden_size", default = 256, type=int)    
    parser.add_argument("-a","--attention", action="store_true") 
    args = parser.parse_args()  
    return args

if __name__ == '__main__':
    train_file = "lexicons/ta.translit.sampled.train.tsv"
    dev_file = "lexicons/ta.translit.sampled.dev.tsv"

    train_data = pd.read_csv (train_file, header=None, sep='\t')
    dev_data = pd.read_csv (dev_file, header=None, sep='\t')
    args = get_args()
    if args.attention == False:
        print("Starting to train model without attention")
        agent = Char2CharModel(hidden_size=args.hidden_size, epochs=args.epochs, 
        batch_size=args.batch_size, dropout=args.dropout, num_encoder_layers=args.num_encoder_layers, 
        num_decoder_layers=args.num_decoder_layers)
        agent.train(train_data, dev_data)
        agent.model.save("s2s_model.keras")
        print("Training finished") 
    else:
        print("Starting to train model with attention")
        agent = Char2CharModelAttention(hidden_size=args.hidden_size, epochs=args.epochs, 
        batch_size=args.batch_size, dropout=args.dropout, num_encoder_layers=args.num_encoder_layers, 
        num_decoder_layers=args.num_decoder_layers)
        agent.train(train_data, dev_data)
        agent.model.save("s2s_model_attn.keras")
        print("Training finished")       

