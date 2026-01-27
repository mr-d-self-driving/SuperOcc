## Training
Train SuperOcc with 4 GPUs:
```
bash tools/dist_train.sh projects/configs/superocc/superocc-t_r50_704_seq_nui_48e.py 4 --work-dir work_dirs/superocc-t/default
```

## Evaluation
```
bash tools/dist_test.sh projects/configs/superocc/superocc-t_r50_704_seq_nui_48e.py work_dirs/superocc-t/default/iter_168768.pth 4 --eval map
```

