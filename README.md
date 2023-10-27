# Projeto
Banco-Imagem

### Nome do aluno
Enio Kilder Oliveira da Silva

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|
|--|--|--|
Classificação de Objetos |YOLOv5|PyTorch|

## Performance

O modelo treinado possui performance de **81%**.

### Output do bloco de treinamento

<details>
  <summary>Expandir Conteúdo!</summary>
  
  ```text
  ###Você deve colar aqui a saída do bloco de treinamento do notebook, contendo todas as épocas e saídas do treinamento
   %cd ../yolov5
  !python classify/train.py --model yolov5n-cls.pt --data $DATASET_NAME --epochs 320 --img 128 --pretrained weights/yolov5n-cls.pt

/content/yolov5
2023-10-27 22:41:18.507423: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2023-10-27 22:41:18.507489: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2023-10-27 22:41:18.507530: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
classify/train: model=yolov5n-cls.pt, data=Banco-Imagem-1, epochs=320, batch_size=64, imgsz=128, nosave=False, cache=None, device=, workers=8, project=runs/train-cls, name=exp, exist_ok=False, pretrained=weights/yolov5n-cls.pt, optimizer=Adam, lr0=0.001, decay=5e-05, label_smoothing=0.1, cutoff=None, dropout=None, verbose=False, seed=0, local_rank=-1
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

TensorBoard: Start with 'tensorboard --logdir runs/train-cls', view at http://localhost:6006/
albumentations: RandomResizedCrop(p=1.0, height=128, width=128, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1), HorizontalFlip(p=0.5), ColorJitter(p=0.5, brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[0, 0]), Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), ToTensorV2(always_apply=True, p=1.0, transpose_mask=False)
Model summary: 149 layers, 1218405 parameters, 1218405 gradients, 3.0 GFLOPs
optimizer: Adam(lr=0.001) with parameter groups 32 weight(decay=0.0), 33 weight(decay=5e-05), 33 bias
Image sizes 128 train, 128 test
Using 1 dataloader workers
Logging results to runs/train-cls/exp2
Starting yolov5n-cls.pt training on Banco-Imagem-1 dataset with 5 classes for 320 epochs...

     Epoch   GPU_mem  train_loss   test_loss    top1_acc    top5_acc
     1/320    0.497G         1.6         1.6       0.222           1: 100% 4/4 [00:01<00:00,  2.43it/s]
     2/320    0.499G        1.52        1.59       0.222           1: 100% 4/4 [00:01<00:00,  2.61it/s]
     3/320    0.499G        1.39        1.59       0.222           1: 100% 4/4 [00:02<00:00,  1.69it/s]
     4/320    0.499G        1.41         1.6       0.222           1: 100% 4/4 [00:01<00:00,  2.09it/s]
     5/320    0.499G        1.37        1.61       0.222           1: 100% 4/4 [00:01<00:00,  2.44it/s]
     6/320    0.499G        1.27        1.63       0.194           1: 100% 4/4 [00:01<00:00,  2.47it/s]
     7/320    0.499G        1.31        1.63       0.194           1: 100% 4/4 [00:01<00:00,  2.47it/s]
     8/320    0.499G        1.35        1.63       0.222           1: 100% 4/4 [00:01<00:00,  2.40it/s]
     9/320    0.499G        1.28        1.61       0.222           1: 100% 4/4 [00:01<00:00,  2.44it/s]
    10/320    0.499G        1.27        1.67       0.222           1: 100% 4/4 [00:02<00:00,  1.50it/s]
    11/320    0.499G        1.27         1.7       0.222           1: 100% 4/4 [00:01<00:00,  3.36it/s]
    12/320    0.499G        1.27        1.63       0.222           1: 100% 4/4 [00:01<00:00,  2.71it/s]
    13/320    0.499G        1.25        1.61       0.222           1: 100% 4/4 [00:01<00:00,  2.55it/s]
    14/320    0.499G        1.26        1.64       0.222           1: 100% 4/4 [00:01<00:00,  2.43it/s]
    15/320    0.499G        1.17        1.69       0.222           1: 100% 4/4 [00:01<00:00,  2.44it/s]
    16/320    0.499G        1.15        1.74       0.222           1: 100% 4/4 [00:02<00:00,  2.00it/s]
    17/320    0.499G        1.21        1.83       0.222           1: 100% 4/4 [00:02<00:00,  1.66it/s]
    18/320    0.499G        1.15        1.92       0.222           1: 100% 4/4 [00:01<00:00,  2.61it/s]
    19/320    0.499G        1.15        2.04       0.222           1: 100% 4/4 [00:02<00:00,  1.58it/s]
    20/320    0.499G        1.15        2.14       0.222           1: 100% 4/4 [00:01<00:00,  2.62it/s]
    21/320    0.499G         1.1        2.06       0.167           1: 100% 4/4 [00:02<00:00,  1.60it/s]
    22/320    0.499G         1.1        2.14       0.194           1: 100% 4/4 [00:02<00:00,  1.33it/s]
    23/320    0.499G        1.13        1.93       0.306           1: 100% 4/4 [00:02<00:00,  1.93it/s]
    24/320    0.499G        1.11        1.84        0.25           1: 100% 4/4 [00:01<00:00,  2.38it/s]
    25/320    0.499G        1.11        1.73       0.278           1: 100% 4/4 [00:01<00:00,  2.07it/s]
    26/320    0.499G        1.12         1.7       0.222           1: 100% 4/4 [00:01<00:00,  2.44it/s]
    27/320    0.499G        1.05        1.72       0.333           1: 100% 4/4 [00:01<00:00,  2.47it/s]
    28/320    0.499G        1.06        1.84       0.222           1: 100% 4/4 [00:01<00:00,  2.21it/s]
    29/320    0.499G        1.06        1.61       0.306           1: 100% 4/4 [00:02<00:00,  1.42it/s]
    30/320    0.499G        1.07        1.54       0.361           1: 100% 4/4 [00:01<00:00,  2.37it/s]
    31/320    0.499G        1.06        1.52       0.333           1: 100% 4/4 [00:01<00:00,  2.54it/s]
    32/320    0.499G         1.1        1.55       0.417           1: 100% 4/4 [00:01<00:00,  2.48it/s]
    33/320    0.499G       0.983        1.36       0.583           1: 100% 4/4 [00:01<00:00,  2.36it/s]
    34/320    0.499G        1.03        1.46       0.472           1: 100% 4/4 [00:01<00:00,  2.64it/s]
    35/320    0.499G        1.03        1.51       0.417           1: 100% 4/4 [00:02<00:00,  1.80it/s]
    36/320    0.499G        1.04        1.38       0.583           1: 100% 4/4 [00:02<00:00,  1.71it/s]
    37/320    0.499G        1.03        1.26       0.611           1: 100% 4/4 [00:02<00:00,  1.79it/s]
    38/320    0.499G        1.06        1.22       0.639           1: 100% 4/4 [00:02<00:00,  1.85it/s]
    39/320    0.499G        1.02        1.18       0.611           1: 100% 4/4 [00:01<00:00,  2.52it/s]
    40/320    0.499G        1.01        1.28         0.5           1: 100% 4/4 [00:01<00:00,  2.33it/s]
    41/320    0.499G       0.959        1.08       0.639           1: 100% 4/4 [00:02<00:00,  1.77it/s]
    42/320    0.499G       0.982        1.04       0.639           1: 100% 4/4 [00:02<00:00,  1.86it/s]
    43/320    0.499G           1        1.08       0.528           1: 100% 4/4 [00:01<00:00,  2.41it/s]
    44/320    0.499G       0.981        1.17         0.5           1: 100% 4/4 [00:01<00:00,  2.39it/s]
    45/320    0.499G        1.02         1.3         0.5           1: 100% 4/4 [00:01<00:00,  2.43it/s]
    46/320    0.499G        1.02        1.25       0.583           1: 100% 4/4 [00:01<00:00,  2.40it/s]
    47/320    0.499G       0.944        1.24       0.583           1: 100% 4/4 [00:01<00:00,  2.39it/s]
    48/320    0.499G       0.986        1.21       0.556           1: 100% 4/4 [00:02<00:00,  1.52it/s]
    49/320    0.499G       0.962        1.15       0.611           1: 100% 4/4 [00:02<00:00,  1.92it/s]
    50/320    0.499G       0.956         1.4       0.444           1: 100% 4/4 [00:01<00:00,  2.41it/s]
    51/320    0.499G        0.94        1.28       0.583           1: 100% 4/4 [00:01<00:00,  2.34it/s]
    52/320    0.499G       0.924        1.05       0.639           1: 100% 4/4 [00:01<00:00,  2.29it/s]
    53/320    0.499G       0.921        1.19       0.639           1: 100% 4/4 [00:01<00:00,  2.32it/s]
    54/320    0.499G       0.905        1.28       0.583           1: 100% 4/4 [00:01<00:00,  2.28it/s]
    55/320    0.499G       0.951        1.16       0.694           1: 100% 4/4 [00:02<00:00,  1.43it/s]
    56/320    0.499G       0.984        1.12       0.556           1: 100% 4/4 [00:01<00:00,  2.51it/s]
    57/320    0.499G       0.919        1.37       0.472           1: 100% 4/4 [00:01<00:00,  2.34it/s]
    58/320    0.499G       0.965         1.2       0.583           1: 100% 4/4 [00:01<00:00,  2.38it/s]
    59/320    0.499G       0.908        1.13       0.639           1: 100% 4/4 [00:01<00:00,  2.33it/s]
    60/320    0.499G       0.908        1.11       0.583           1: 100% 4/4 [00:01<00:00,  2.41it/s]
    61/320    0.499G       0.919        1.11       0.583           1: 100% 4/4 [00:02<00:00,  1.89it/s]
    62/320    0.499G       0.905        1.18       0.583           1: 100% 4/4 [00:02<00:00,  1.60it/s]
    63/320    0.499G       0.844        1.37       0.528           1: 100% 4/4 [00:01<00:00,  2.32it/s]
    64/320    0.499G        0.87        1.13       0.583           1: 100% 4/4 [00:01<00:00,  2.35it/s]
    65/320    0.499G       0.852        1.06       0.611           1: 100% 4/4 [00:01<00:00,  2.46it/s]
    66/320    0.499G       0.839       0.923       0.722           1: 100% 4/4 [00:01<00:00,  2.33it/s]
    67/320    0.499G       0.804           1       0.639           1: 100% 4/4 [00:01<00:00,  2.66it/s]
    68/320    0.499G       0.848        1.29         0.5           1: 100% 4/4 [00:02<00:00,  1.58it/s]
    69/320    0.499G       0.906        1.39       0.528           1: 100% 4/4 [00:01<00:00,  2.60it/s]
    70/320    0.499G       0.877        1.12       0.611           1: 100% 4/4 [00:01<00:00,  2.31it/s]
    71/320    0.499G       0.843        1.01        0.75           1: 100% 4/4 [00:01<00:00,  2.53it/s]
    72/320    0.499G       0.878        1.05       0.667           1: 100% 4/4 [00:01<00:00,  2.41it/s]
    73/320    0.499G       0.807        1.17       0.583           1: 100% 4/4 [00:01<00:00,  2.29it/s]
    74/320    0.499G       0.912        1.25         0.5           1: 100% 4/4 [00:01<00:00,  2.30it/s]
    75/320    0.499G       0.807         1.2       0.472           1: 100% 4/4 [00:02<00:00,  1.44it/s]
    76/320    0.499G       0.871        0.93        0.75           1: 100% 4/4 [00:01<00:00,  2.46it/s]
    77/320    0.499G        0.82        1.08       0.667           1: 100% 4/4 [00:01<00:00,  2.42it/s]
    78/320    0.499G       0.846        1.22       0.556           1: 100% 4/4 [00:01<00:00,  2.37it/s]
    79/320    0.499G        0.84        1.15       0.556           1: 100% 4/4 [00:01<00:00,  2.43it/s]
    80/320    0.499G       0.852        1.19       0.611           1: 100% 4/4 [00:01<00:00,  2.30it/s]
    81/320    0.499G       0.826        1.21       0.667           1: 100% 4/4 [00:01<00:00,  2.03it/s]
    82/320    0.499G       0.867        1.23       0.611           1: 100% 4/4 [00:02<00:00,  1.58it/s]
    83/320    0.499G       0.862        1.14       0.611           1: 100% 4/4 [00:01<00:00,  2.33it/s]
    84/320    0.499G       0.783        1.02       0.639           1: 100% 4/4 [00:01<00:00,  2.40it/s]
    85/320    0.499G       0.795        1.19       0.639           1: 100% 4/4 [00:01<00:00,  2.35it/s]
    86/320    0.499G       0.799        1.12       0.528           1: 100% 4/4 [00:01<00:00,  2.09it/s]
    87/320    0.499G       0.766        1.13       0.556           1: 100% 4/4 [00:01<00:00,  2.50it/s]
    88/320    0.499G       0.826        1.11       0.639           1: 100% 4/4 [00:02<00:00,  1.59it/s]
    89/320    0.499G       0.799        1.24       0.583           1: 100% 4/4 [00:02<00:00,  1.99it/s]
    90/320    0.499G       0.795        1.15       0.583           1: 100% 4/4 [00:01<00:00,  2.48it/s]
    91/320    0.499G       0.734       0.946       0.722           1: 100% 4/4 [00:01<00:00,  2.40it/s]
    92/320    0.499G       0.738       0.893        0.75           1: 100% 4/4 [00:01<00:00,  2.55it/s]
    93/320    0.499G       0.798        1.05       0.611           1: 100% 4/4 [00:01<00:00,  2.37it/s]
    94/320    0.499G       0.829        1.05       0.611           1: 100% 4/4 [00:01<00:00,  2.49it/s]
    95/320    0.499G       0.754       0.985       0.639           1: 100% 4/4 [00:02<00:00,  1.42it/s]
    96/320    0.499G       0.765        1.02       0.667           1: 100% 4/4 [00:01<00:00,  2.15it/s]
    97/320    0.499G       0.709        1.04       0.694           1: 100% 4/4 [00:01<00:00,  2.33it/s]
    98/320    0.499G        0.77        1.07       0.556           1: 100% 4/4 [00:01<00:00,  2.53it/s]
    99/320    0.499G       0.719        1.06       0.694           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   100/320    0.499G       0.774        1.01       0.694           1: 100% 4/4 [00:01<00:00,  2.42it/s]
   101/320    0.499G       0.745        1.15       0.639           1: 100% 4/4 [00:01<00:00,  2.15it/s]
   102/320    0.499G       0.781        1.36       0.611           1: 100% 4/4 [00:02<00:00,  1.41it/s]
   103/320    0.499G       0.683        1.24       0.667           1: 100% 4/4 [00:01<00:00,  2.51it/s]
   104/320    0.499G       0.734        1.22       0.611           1: 100% 4/4 [00:01<00:00,  2.05it/s]
   105/320    0.499G       0.746         1.1       0.611           1: 100% 4/4 [00:02<00:00,  1.50it/s]
   106/320    0.499G       0.693         1.2       0.583           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   107/320    0.499G       0.732        1.21       0.611           1: 100% 4/4 [00:01<00:00,  2.05it/s]
   108/320    0.499G        0.78        1.12       0.639           1: 100% 4/4 [00:02<00:00,  1.57it/s]
   109/320    0.499G       0.659        1.07       0.694           1: 100% 4/4 [00:01<00:00,  2.50it/s]
   110/320    0.499G       0.769        1.08       0.611           1: 100% 4/4 [00:01<00:00,  2.40it/s]
   111/320    0.499G       0.729        1.06       0.694           1: 100% 4/4 [00:01<00:00,  2.52it/s]
   112/320    0.499G       0.693        1.14       0.639           1: 100% 4/4 [00:01<00:00,  2.62it/s]
   113/320    0.499G       0.696        1.45       0.528           1: 100% 4/4 [00:01<00:00,  2.53it/s]
   114/320    0.499G       0.757        1.41       0.583           1: 100% 4/4 [00:02<00:00,  1.84it/s]
   115/320    0.499G       0.711        1.15       0.639           1: 100% 4/4 [00:02<00:00,  1.66it/s]
   116/320    0.499G       0.711        1.14       0.611           1: 100% 4/4 [00:01<00:00,  2.52it/s]
   117/320    0.499G        0.71        1.18       0.639           1: 100% 4/4 [00:01<00:00,  2.48it/s]
   118/320    0.499G       0.733           1       0.694           1: 100% 4/4 [00:01<00:00,  2.49it/s]
   119/320    0.499G       0.703        1.02       0.694           1: 100% 4/4 [00:01<00:00,  2.54it/s]
   120/320    0.499G       0.656        1.16       0.611           1: 100% 4/4 [00:01<00:00,  2.34it/s]
   121/320    0.499G       0.709        1.18       0.528           1: 100% 4/4 [00:02<00:00,  1.60it/s]
   122/320    0.499G       0.699         1.1       0.639           1: 100% 4/4 [00:03<00:00,  1.27it/s]
   123/320    0.499G       0.689        1.15       0.583           1: 100% 4/4 [00:02<00:00,  1.94it/s]
   124/320    0.499G       0.696        1.33       0.556           1: 100% 4/4 [00:01<00:00,  2.43it/s]
   125/320    0.499G       0.674        1.42       0.556           1: 100% 4/4 [00:01<00:00,  2.43it/s]
   126/320    0.499G       0.713        1.49       0.583           1: 100% 4/4 [00:01<00:00,  2.13it/s]
   127/320    0.499G       0.689        1.33       0.639           1: 100% 4/4 [00:01<00:00,  2.42it/s]
   128/320    0.499G        0.63        1.16       0.639           1: 100% 4/4 [00:02<00:00,  1.74it/s]
   129/320    0.499G       0.643        1.14       0.667           1: 100% 4/4 [00:02<00:00,  1.73it/s]
   130/320    0.499G       0.681        1.21       0.639           1: 100% 4/4 [00:01<00:00,  2.55it/s]
   131/320    0.499G       0.672        1.17       0.556           1: 100% 4/4 [00:01<00:00,  2.46it/s]
   132/320    0.499G       0.664        1.16       0.694           1: 100% 4/4 [00:01<00:00,  2.56it/s]
   133/320    0.499G       0.641        1.17       0.583           1: 100% 4/4 [00:01<00:00,  2.32it/s]
   134/320    0.499G       0.692        1.16         0.5           1: 100% 4/4 [00:01<00:00,  2.55it/s]
   135/320    0.499G       0.667        1.17       0.528           1: 100% 4/4 [00:02<00:00,  1.59it/s]
   136/320    0.499G       0.717        1.08       0.667           1: 100% 4/4 [00:02<00:00,  1.95it/s]
   137/320    0.499G       0.714        1.13       0.639           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   138/320    0.499G       0.664           1       0.722           1: 100% 4/4 [00:01<00:00,  2.56it/s]
   139/320    0.499G       0.703       0.964       0.778           1: 100% 4/4 [00:01<00:00,  2.61it/s]
   140/320    0.499G       0.692       0.924        0.75           1: 100% 4/4 [00:01<00:00,  2.40it/s]
   141/320    0.499G       0.634        1.02       0.667           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   142/320    0.499G       0.635        1.11       0.667           1: 100% 4/4 [00:02<00:00,  1.38it/s]
   143/320    0.499G       0.659        1.07        0.75           1: 100% 4/4 [00:01<00:00,  2.39it/s]
   144/320    0.499G       0.698        1.03       0.778           1: 100% 4/4 [00:01<00:00,  2.23it/s]
   145/320    0.499G       0.776        1.01       0.611           1: 100% 4/4 [00:01<00:00,  2.43it/s]
   146/320    0.499G       0.669        1.03       0.639           1: 100% 4/4 [00:01<00:00,  2.40it/s]
   147/320    0.499G        0.67        1.01       0.667           1: 100% 4/4 [00:01<00:00,  2.47it/s]
   148/320    0.499G       0.693        1.11       0.667           1: 100% 4/4 [00:02<00:00,  1.81it/s]
   149/320    0.499G       0.677        1.23       0.583           1: 100% 4/4 [00:02<00:00,  1.66it/s]
   150/320    0.499G       0.639        1.07       0.611           1: 100% 4/4 [00:01<00:00,  2.58it/s]
   151/320    0.499G       0.615        1.02       0.611           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   152/320    0.499G       0.685        1.04       0.667           1: 100% 4/4 [00:01<00:00,  2.41it/s]
   153/320    0.499G       0.645        1.12       0.694           1: 100% 4/4 [00:01<00:00,  2.45it/s]
   154/320    0.499G       0.619        1.16       0.611           1: 100% 4/4 [00:01<00:00,  2.57it/s]
   155/320    0.499G       0.675        1.17       0.639           1: 100% 4/4 [00:02<00:00,  1.74it/s]
   156/320    0.499G       0.649         1.2       0.639           1: 100% 4/4 [00:02<00:00,  1.82it/s]
   157/320    0.499G        0.63        1.13       0.639           1: 100% 4/4 [00:01<00:00,  2.40it/s]
   158/320    0.499G        0.65        1.14       0.583           1: 100% 4/4 [00:01<00:00,  2.34it/s]
   159/320    0.499G       0.689        1.23       0.528           1: 100% 4/4 [00:01<00:00,  2.50it/s]
   160/320    0.499G       0.669        1.15       0.611           1: 100% 4/4 [00:01<00:00,  2.52it/s]
   161/320    0.499G       0.697         1.3       0.611           1: 100% 4/4 [00:01<00:00,  2.31it/s]
   162/320    0.499G       0.633        1.24       0.667           1: 100% 4/4 [00:02<00:00,  1.52it/s]
   163/320    0.499G       0.664        1.09       0.583           1: 100% 4/4 [00:01<00:00,  2.04it/s]
   164/320    0.499G       0.671        1.01       0.694           1: 100% 4/4 [00:01<00:00,  2.38it/s]
   165/320    0.499G       0.637        1.08       0.667           1: 100% 4/4 [00:01<00:00,  2.35it/s]
   166/320    0.499G       0.632        1.17       0.556           1: 100% 4/4 [00:01<00:00,  2.31it/s]
   167/320    0.499G        0.64        1.03       0.694           1: 100% 4/4 [00:01<00:00,  2.41it/s]
   168/320    0.499G       0.616       0.976       0.722           1: 100% 4/4 [00:01<00:00,  2.22it/s]
   169/320    0.499G       0.632        1.01        0.75           1: 100% 4/4 [00:02<00:00,  1.47it/s]
   170/320    0.499G       0.634        1.05       0.722           1: 100% 4/4 [00:01<00:00,  2.23it/s]
   171/320    0.499G       0.623        1.03        0.75           1: 100% 4/4 [00:01<00:00,  2.51it/s]
   172/320    0.499G       0.653        1.09       0.722           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   173/320    0.499G       0.631        1.14       0.694           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   174/320    0.499G       0.596        1.27       0.611           1: 100% 4/4 [00:01<00:00,  2.33it/s]
   175/320    0.499G       0.608        1.29       0.611           1: 100% 4/4 [00:02<00:00,  1.99it/s]
   176/320    0.499G       0.645        1.06       0.722           1: 100% 4/4 [00:02<00:00,  1.52it/s]
   177/320    0.499G       0.555        1.02       0.694           1: 100% 4/4 [00:01<00:00,  2.35it/s]
   178/320    0.499G       0.646        1.08       0.583           1: 100% 4/4 [00:01<00:00,  2.33it/s]
   179/320    0.499G       0.649        1.11       0.639           1: 100% 4/4 [00:01<00:00,  2.42it/s]
   180/320    0.499G       0.634        1.26       0.611           1: 100% 4/4 [00:01<00:00,  2.32it/s]
   181/320    0.499G       0.621        1.41       0.556           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   182/320    0.499G       0.591        1.34       0.528           1: 100% 4/4 [00:02<00:00,  1.70it/s]
   183/320    0.499G       0.644        1.11       0.694           1: 100% 4/4 [00:02<00:00,  1.71it/s]
   184/320    0.499G       0.649        1.08       0.722           1: 100% 4/4 [00:01<00:00,  2.19it/s]
   185/320    0.499G       0.607        1.02        0.75           1: 100% 4/4 [00:01<00:00,  2.30it/s]
   186/320    0.499G       0.635       0.979       0.722           1: 100% 4/4 [00:01<00:00,  2.40it/s]
   187/320    0.499G       0.647        1.07       0.667           1: 100% 4/4 [00:01<00:00,  2.52it/s]
   188/320    0.499G       0.618        1.21       0.583           1: 100% 4/4 [00:02<00:00,  1.69it/s]
   189/320    0.499G       0.579        1.15       0.667           1: 100% 4/4 [00:03<00:00,  1.27it/s]
   190/320    0.499G       0.647        1.01        0.75           1: 100% 4/4 [00:02<00:00,  1.80it/s]
   191/320    0.499G       0.579       0.998        0.75           1: 100% 4/4 [00:01<00:00,  2.38it/s]
   192/320    0.499G       0.631       0.999        0.75           1: 100% 4/4 [00:01<00:00,  2.43it/s]
   193/320    0.499G       0.646        1.08       0.667           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   194/320    0.499G       0.621        1.23       0.583           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   195/320    0.499G       0.588        1.25       0.583           1: 100% 4/4 [00:01<00:00,  2.31it/s]
   196/320    0.499G       0.612        1.18       0.611           1: 100% 4/4 [00:02<00:00,  1.51it/s]
   197/320    0.499G       0.634        1.19       0.528           1: 100% 4/4 [00:01<00:00,  2.01it/s]
   198/320    0.499G       0.594         1.2       0.528           1: 100% 4/4 [00:01<00:00,  2.55it/s]
   199/320    0.499G       0.594        1.22       0.583           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   200/320    0.499G       0.591        1.22       0.667           1: 100% 4/4 [00:01<00:00,  2.49it/s]
   201/320    0.499G       0.588        1.09       0.722           1: 100% 4/4 [00:01<00:00,  2.41it/s]
   202/320    0.499G       0.585        1.05       0.778           1: 100% 4/4 [00:01<00:00,  2.00it/s]
   203/320    0.499G       0.559        1.05       0.667           1: 100% 4/4 [00:02<00:00,  1.43it/s]
   204/320    0.499G       0.643        1.09       0.583           1: 100% 4/4 [00:02<00:00,  1.64it/s]
   205/320    0.499G       0.577        1.06       0.639           1: 100% 4/4 [00:02<00:00,  1.83it/s]
   206/320    0.499G       0.554        1.01       0.722           1: 100% 4/4 [00:01<00:00,  2.42it/s]
   207/320    0.499G       0.615        1.01       0.722           1: 100% 4/4 [00:01<00:00,  2.30it/s]
   208/320    0.499G       0.642       0.974       0.806           1: 100% 4/4 [00:01<00:00,  2.03it/s]
   209/320    0.499G       0.595        1.13       0.639           1: 100% 4/4 [00:02<00:00,  1.54it/s]
   210/320    0.499G       0.554        1.34       0.611           1: 100% 4/4 [00:01<00:00,  2.41it/s]
   211/320    0.499G       0.573        1.38       0.611           1: 100% 4/4 [00:01<00:00,  2.32it/s]
   212/320    0.499G       0.618        1.33       0.556           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   213/320    0.499G       0.603        1.29       0.611           1: 100% 4/4 [00:01<00:00,  2.34it/s]
   214/320    0.499G       0.611        1.27       0.556           1: 100% 4/4 [00:01<00:00,  2.28it/s]
   215/320    0.499G       0.648        1.12       0.611           1: 100% 4/4 [00:02<00:00,  1.70it/s]
   216/320    0.499G       0.518        1.03       0.667           1: 100% 4/4 [00:02<00:00,  1.77it/s]
   217/320    0.499G       0.546       0.996       0.694           1: 100% 4/4 [00:01<00:00,  2.34it/s]
   218/320    0.499G       0.591       0.947       0.722           1: 100% 4/4 [00:01<00:00,  2.53it/s]
   219/320    0.499G       0.632        0.92        0.75           1: 100% 4/4 [00:01<00:00,  2.39it/s]
   220/320    0.499G       0.554       0.914        0.75           1: 100% 4/4 [00:01<00:00,  2.52it/s]
   221/320    0.499G       0.559       0.944       0.694           1: 100% 4/4 [00:01<00:00,  2.48it/s]
   222/320    0.499G       0.562       0.951       0.722           1: 100% 4/4 [00:02<00:00,  1.50it/s]
   223/320    0.499G        0.52       0.909       0.694           1: 100% 4/4 [00:02<00:00,  1.97it/s]
   224/320    0.499G       0.564       0.873        0.75           1: 100% 4/4 [00:01<00:00,  2.43it/s]
   225/320    0.499G       0.595       0.883       0.778           1: 100% 4/4 [00:01<00:00,  2.52it/s]
   226/320    0.499G        0.59       0.901       0.778           1: 100% 4/4 [00:01<00:00,  2.51it/s]
   227/320    0.499G       0.581       0.933        0.75           1: 100% 4/4 [00:01<00:00,  2.55it/s]
   228/320    0.499G       0.624       0.958       0.722           1: 100% 4/4 [00:01<00:00,  2.42it/s]
   229/320    0.499G       0.579       0.932       0.778           1: 100% 4/4 [00:02<00:00,  1.41it/s]
   230/320    0.499G       0.574       0.945       0.722           1: 100% 4/4 [00:01<00:00,  2.27it/s]
   231/320    0.499G       0.603       0.976       0.694           1: 100% 4/4 [00:01<00:00,  2.56it/s]
   232/320    0.499G       0.578       0.978       0.694           1: 100% 4/4 [00:01<00:00,  2.32it/s]
   233/320    0.499G       0.566        1.04       0.639           1: 100% 4/4 [00:01<00:00,  2.45it/s]
   234/320    0.499G       0.551        1.14       0.639           1: 100% 4/4 [00:01<00:00,  2.56it/s]
   235/320    0.499G       0.572        1.17       0.639           1: 100% 4/4 [00:02<00:00,  1.94it/s]
   236/320    0.499G       0.564        1.09       0.667           1: 100% 4/4 [00:02<00:00,  1.57it/s]
   237/320    0.499G       0.616        0.96        0.75           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   238/320    0.499G       0.568       0.939       0.778           1: 100% 4/4 [00:01<00:00,  2.33it/s]
   239/320    0.499G       0.565       0.992       0.722           1: 100% 4/4 [00:01<00:00,  2.43it/s]
   240/320    0.499G       0.555        1.07       0.667           1: 100% 4/4 [00:01<00:00,  2.32it/s]
   241/320    0.499G       0.552        1.06       0.639           1: 100% 4/4 [00:01<00:00,  2.50it/s]
   242/320    0.499G       0.546        1.01       0.694           1: 100% 4/4 [00:02<00:00,  1.62it/s]
   243/320    0.499G       0.584       0.982       0.667           1: 100% 4/4 [00:02<00:00,  1.83it/s]
   244/320    0.499G       0.577        1.02       0.694           1: 100% 4/4 [00:01<00:00,  2.42it/s]
   245/320    0.499G       0.616        1.05       0.722           1: 100% 4/4 [00:01<00:00,  2.31it/s]
   246/320    0.499G       0.544        1.02       0.667           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   247/320    0.499G       0.536           1       0.667           1: 100% 4/4 [00:01<00:00,  2.44it/s]
   248/320    0.499G       0.584        1.08       0.639           1: 100% 4/4 [00:01<00:00,  2.42it/s]
   249/320    0.499G       0.517        1.19       0.639           1: 100% 4/4 [00:02<00:00,  1.46it/s]
   250/320    0.499G       0.566         1.2       0.639           1: 100% 4/4 [00:01<00:00,  2.20it/s]
   251/320    0.499G       0.553        1.18       0.639           1: 100% 4/4 [00:01<00:00,  2.55it/s]
   252/320    0.499G       0.557        1.13       0.639           1: 100% 4/4 [00:01<00:00,  2.53it/s]
   253/320    0.499G       0.564        1.13       0.639           1: 100% 4/4 [00:01<00:00,  2.49it/s]
   254/320    0.499G       0.557        1.17       0.639           1: 100% 4/4 [00:01<00:00,  2.38it/s]
   255/320    0.499G       0.576        1.13       0.639           1: 100% 4/4 [00:01<00:00,  2.33it/s]
   256/320    0.499G       0.574        1.05       0.694           1: 100% 4/4 [00:02<00:00,  1.45it/s]
   257/320    0.499G       0.532        1.04       0.639           1: 100% 4/4 [00:01<00:00,  2.47it/s]
   258/320    0.499G       0.574        1.02       0.722           1: 100% 4/4 [00:01<00:00,  2.48it/s]
   259/320    0.499G       0.525       0.991       0.722           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   260/320    0.499G       0.572       0.989        0.75           1: 100% 4/4 [00:01<00:00,  2.54it/s]
   261/320    0.499G       0.553       0.965        0.75           1: 100% 4/4 [00:01<00:00,  2.33it/s]
   262/320    0.499G       0.563       0.967       0.806           1: 100% 4/4 [00:01<00:00,  2.01it/s]
   263/320    0.499G        0.53       0.985       0.778           1: 100% 4/4 [00:02<00:00,  1.41it/s]
   264/320    0.499G       0.548        1.01        0.75           1: 100% 4/4 [00:01<00:00,  2.47it/s]
   265/320    0.499G       0.542        1.04       0.694           1: 100% 4/4 [00:01<00:00,  2.33it/s]
   266/320    0.499G       0.579        1.02       0.722           1: 100% 4/4 [00:01<00:00,  2.39it/s]
   267/320    0.499G       0.555       0.981       0.722           1: 100% 4/4 [00:01<00:00,  2.45it/s]
   268/320    0.499G       0.545       0.973        0.75           1: 100% 4/4 [00:01<00:00,  2.30it/s]
   269/320    0.499G       0.524       0.974        0.75           1: 100% 4/4 [00:02<00:00,  1.40it/s]
   270/320    0.499G       0.545       0.957        0.75           1: 100% 4/4 [00:03<00:00,  1.30it/s]
   271/320    0.499G        0.61       0.948       0.806           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   272/320    0.499G       0.542       0.969       0.722           1: 100% 4/4 [00:01<00:00,  2.38it/s]
   273/320    0.499G        0.53       0.972       0.722           1: 100% 4/4 [00:01<00:00,  2.38it/s]
   274/320    0.499G       0.532       0.994       0.694           1: 100% 4/4 [00:01<00:00,  2.55it/s]
   275/320    0.499G       0.542        1.02       0.694           1: 100% 4/4 [00:01<00:00,  2.52it/s]
   276/320    0.499G       0.518        1.04       0.694           1: 100% 4/4 [00:01<00:00,  2.07it/s]
   277/320    0.499G       0.545        1.02       0.722           1: 100% 4/4 [00:02<00:00,  1.44it/s]
   278/320    0.499G       0.562       0.998       0.694           1: 100% 4/4 [00:01<00:00,  2.31it/s]
   279/320    0.499G       0.509        1.03       0.694           1: 100% 4/4 [00:01<00:00,  2.35it/s]
   280/320    0.499G        0.54        1.07       0.667           1: 100% 4/4 [00:01<00:00,  2.51it/s]
   281/320    0.499G       0.534        1.12       0.611           1: 100% 4/4 [00:01<00:00,  2.33it/s]
   282/320    0.499G       0.512        1.16       0.639           1: 100% 4/4 [00:01<00:00,  2.38it/s]
   283/320    0.499G         0.5        1.19       0.611           1: 100% 4/4 [00:02<00:00,  1.71it/s]
   284/320    0.499G       0.511        1.24       0.611           1: 100% 4/4 [00:02<00:00,  1.70it/s]
   285/320    0.499G       0.539         1.2       0.583           1: 100% 4/4 [00:01<00:00,  2.46it/s]
   286/320    0.499G       0.534        1.15       0.611           1: 100% 4/4 [00:01<00:00,  2.39it/s]
   287/320    0.499G       0.546        1.14       0.639           1: 100% 4/4 [00:01<00:00,  2.44it/s]
   288/320    0.499G       0.556        1.14       0.639           1: 100% 4/4 [00:01<00:00,  2.32it/s]
   289/320    0.499G       0.509        1.12       0.639           1: 100% 4/4 [00:02<00:00,  1.79it/s]
   290/320    0.499G       0.544        1.09       0.639           1: 100% 4/4 [00:03<00:00,  1.31it/s]
   291/320    0.499G       0.555        1.04       0.667           1: 100% 4/4 [00:02<00:00,  1.70it/s]
   292/320    0.499G       0.507        1.01       0.667           1: 100% 4/4 [00:01<00:00,  2.45it/s]
   293/320    0.499G       0.553        1.01       0.639           1: 100% 4/4 [00:01<00:00,  2.46it/s]
   294/320    0.499G        0.54        1.01       0.667           1: 100% 4/4 [00:01<00:00,  2.41it/s]
   295/320    0.499G       0.523        1.01       0.639           1: 100% 4/4 [00:01<00:00,  2.58it/s]
   296/320    0.499G       0.509        1.02       0.667           1: 100% 4/4 [00:01<00:00,  2.35it/s]
   297/320    0.499G       0.508        1.03       0.694           1: 100% 4/4 [00:02<00:00,  1.47it/s]
   298/320    0.499G       0.548        1.05       0.667           1: 100% 4/4 [00:02<00:00,  1.99it/s]
   299/320    0.499G       0.491        1.08       0.639           1: 100% 4/4 [00:01<00:00,  2.82it/s]
   300/320    0.499G       0.499        1.11       0.639           1: 100% 4/4 [00:01<00:00,  2.54it/s]
   301/320    0.499G       0.507        1.15       0.639           1: 100% 4/4 [00:01<00:00,  2.44it/s]
   302/320    0.499G       0.509        1.16       0.639           1: 100% 4/4 [00:01<00:00,  2.44it/s]
   303/320    0.499G       0.497        1.19       0.639           1: 100% 4/4 [00:01<00:00,  2.25it/s]
   304/320    0.499G       0.529        1.17       0.667           1: 100% 4/4 [00:02<00:00,  1.37it/s]
   305/320    0.499G       0.519        1.14       0.667           1: 100% 4/4 [00:01<00:00,  2.47it/s]
   306/320    0.499G        0.54         1.1       0.667           1: 100% 4/4 [00:01<00:00,  2.53it/s]
   307/320    0.499G        0.54        1.06       0.667           1: 100% 4/4 [00:01<00:00,  2.48it/s]
   308/320    0.499G       0.495        1.02       0.667           1: 100% 4/4 [00:01<00:00,  2.55it/s]
   309/320    0.499G       0.517           1       0.694           1: 100% 4/4 [00:01<00:00,  2.36it/s]
   310/320    0.499G       0.529       0.982       0.694           1: 100% 4/4 [00:02<00:00,  1.84it/s]
   311/320    0.499G       0.469       0.976       0.722           1: 100% 4/4 [00:02<00:00,  1.60it/s]
   312/320    0.499G         0.5       0.979       0.722           1: 100% 4/4 [00:01<00:00,  2.53it/s]
   313/320    0.499G       0.515        0.97       0.722           1: 100% 4/4 [00:01<00:00,  2.37it/s]
   314/320    0.499G       0.536       0.971       0.722           1: 100% 4/4 [00:01<00:00,  2.47it/s]
   315/320    0.499G       0.504        0.98       0.722           1: 100% 4/4 [00:01<00:00,  2.34it/s]
   316/320    0.499G       0.521       0.986       0.694           1: 100% 4/4 [00:01<00:00,  2.54it/s]
   317/320    0.499G       0.538       0.994       0.694           1: 100% 4/4 [00:02<00:00,  1.71it/s]
   318/320    0.499G       0.524           1       0.694           1: 100% 4/4 [00:02<00:00,  1.79it/s]
   319/320    0.499G       0.547           1       0.694           1: 100% 4/4 [00:01<00:00,  2.40it/s]
   320/320    0.499G       0.509        1.01       0.694           1: 100% 4/4 [00:01<00:00,  2.35it/s]

Training complete (0.184 hours)
Results saved to runs/train-cls/exp2
Predict:         python classify/predict.py --weights runs/train-cls/exp2/weights/best.pt --source im.jpg
Validate:        python classify/val.py --weights runs/train-cls/exp2/weights/best.pt --data Banco-Imagem-1
Export:          python export.py --weights runs/train-cls/exp2/weights/best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train-cls/exp2/weights/best.pt')
Visualize:       https://netron.app


!python classify/val.py --weights runs/train-cls/exp/weights/best.pt --data $DATASET_NAME

classify/val: data=Banco-Imagem-1, weights=['runs/train-cls/exp/weights/best.pt'], batch_size=128, imgsz=224, device=, workers=8, verbose=True, project=runs/val-cls, name=exp, exist_ok=False, half=False, dnn=False
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

Fusing layers... 
Model summary: 117 layers, 1214869 parameters, 0 gradients, 2.9 GFLOPs
testing: 100% 1/1 [00:00<00:00,  1.01it/s]
                   Class      Images    top1_acc    top5_acc
                     all          36       0.806           1
                  avioes           7       0.714           1
                  barcos           6       0.833           1
                  carros          11       0.818           1
            helicopteros           8       0.875           1
                   motos           4        0.75           1
Speed: 0.1ms pre-process, 12.8ms inference, 1.0ms post-process per image at shape (1, 3, 224, 224)
Results saved to runs/val-cls/exp2



```
</details>
 
### Evidências do treinamento

Nessa seção você deve colocar qualquer evidência do treinamento, como por exemplo gráficos de perda, performance, matriz de confusão etc.

Exemplo de adição de imagem:

![Descrição](https://i.imgur.com/GB9Tihf.jpg)

#Download imagem para exemplificação
import requests
#carro
image_url = "https://i.imgur.com/GB9Tihf.jpg"
response = requests.get(image_url)
response.raise_for_status()
with open('carro.jpg', 'wb') as handler:
    handler.write(response.content)

# !python classify/predict.py --weights ./weigths/yolov5x-cls.pt --source carro.jpg

classify/predict: weights=['./weigths/yolov5x-cls.pt'], source=carro.jpg, data=data/coco128.yaml, imgsz=[224, 224], device=, view_img=False, save_txt=False, nosave=False, augment=False, visualize=False, update=False, project=runs/predict-cls, name=exp, exist_ok=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

Fusing layers... 
Model summary: 264 layers, 48072600 parameters, 0 gradients, 129.9 GFLOPs
image 1/1 /content/yolov5/carro.jpg: 224x224 sports car 0.95, race car 0.02, convertible 0.01, car wheel 0.00, grille 0.00, 12.9ms
Speed: 0.4ms pre-process, 12.9ms inference, 6.9ms NMS per image at shape (1, 3, 224, 224)
Results saved to runs/predict-cls/exp13



## Roboflow

Banco-Imagem > 2023-10-24 9:29pm

https://universe.roboflow.com/eniokilder/banco-imagem

Provided by a Roboflow user
License: CC BY 4.0


## HuggingFace

Link para o HuggingFace:  Não há
