#!/bin/bash

# M
python -u predict_informer.py --model informer --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --des 'Exp'  --root_path ../../dataset --pred_idx 192 --data ECL --inverse


# S
ython -u predict_informer.py --model informer --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 3 --d_layers 2 --attn prob --des 'Exp'  --root_path ../../dataset --data ECL --inverse --pred_idx 500