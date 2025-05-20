# DA6401_A3

Wandb report link: [https://api.wandb.ai/links/jayagowtham-indian-institute-of-technology-madras/n5iyxtmf](https://api.wandb.ai/links/jayagowtham-indian-institute-of-technology-madras/n5iyxtmf)

## Steps to Reproduce

### Training Example

```bash
python3 train.py --e 20 --bs 64 --a
```

### Command Line Arguments for `train.py`

- `-e`, `--epochs`  
  Default: `20`  
  Type: `int`  

- `-b`, `--batch_size`  
  Default: `64`  
  Type: `int`  

- `-nehl`, `--num_encoder_layers`  
  Default: `2`  
  Type: `int`  

- `-ndhl`, `--num_decoder_layers`  
  Default: `2`  
  Type: `int`  

- `-d`, `--dropout`  
  Default: `0`  
  Type: `float`  

- `-sz`, `--hidden_size`  
  Default: `256`  
  Type: `int`  

- `-a`, `--attention`  
  Type: `flag` (use `--a` to enable attention)

---

### Inference Example

```bash
python3 inference.py --a
```

### Command Line Arguments for `inference.py`

- `-a`, `--attention`  
  Type: `flag` (use `--a` to enable attention)
