# Projeto
Banco-Imagem

### Nome do aluno
Enio Kilder Oliveira da Silva

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|
|--|--|--|
Classificação de Objetos |YOLOv5|PyTorch|

## Performance

O modelo treinado possui performance de **%**.

### Output do bloco de treinamento

<details>
  <summary>Expandir Conteúdo!</summary>
  
  ```text
%%time
%cd ../yolov5
!python classify/train.py --model yolov5n-cls.pt --data $DATASET_NAME --epochs 128 --batch 16 --img 320 --pretrained weights/yolov5n-cls.pt

/content/yolov5
2023-10-28 01:49:35.242300: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2023-10-28 01:49:35.242363: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2023-10-28 01:49:35.242406: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
classify/train: model=yolov5n-cls.pt, data=Banco-Imagem-1, epochs=128, batch_size=16, imgsz=320, nosave=False, cache=None, device=, workers=8, project=runs/train-cls, name=exp, exist_ok=False, pretrained=weights/yolov5n-cls.pt, optimizer=Adam, lr0=0.001, decay=5e-05, label_smoothing=0.1, cutoff=None, dropout=None, verbose=False, seed=0, local_rank=-1
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

TensorBoard: Start with 'tensorboard --logdir runs/train-cls', view at http://localhost:6006/
albumentations: RandomResizedCrop(p=1.0, height=320, width=320, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1), HorizontalFlip(p=0.5), ColorJitter(p=0.5, brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[0, 0]), Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), ToTensorV2(always_apply=True, p=1.0, transpose_mask=False)
Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-cls.pt to yolov5n-cls.pt...
100% 4.87M/4.87M [00:00<00:00, 48.4MB/s]

Model summary: 149 layers, 1218405 parameters, 1218405 gradients, 3.0 GFLOPs
optimizer: Adam(lr=0.001) with parameter groups 32 weight(decay=0.0), 33 weight(decay=5e-05), 33 bias
Image sizes 320 train, 320 test
Using 1 dataloader workers
Logging results to runs/train-cls/exp
Starting yolov5n-cls.pt training on Banco-Imagem-1 dataset with 5 classes for 128 epochs...

     Epoch   GPU_mem  train_loss   test_loss    top1_acc    top5_acc
     1/128    0.508G        1.55        1.51       0.194           1: 100% 16/16 [00:06<00:00,  2.59it/s]
     2/128    0.508G        1.39        1.86       0.222           1: 100% 16/16 [00:02<00:00,  6.81it/s]
     3/128    0.508G         1.4        2.07       0.194           1: 100% 16/16 [00:02<00:00,  7.04it/s]
     4/128    0.508G        1.35        1.75       0.222           1: 100% 16/16 [00:02<00:00,  6.38it/s]
     5/128    0.508G        1.34        2.17       0.222           1: 100% 16/16 [00:02<00:00,  5.51it/s]
     6/128    0.508G        1.26        1.76        0.25           1: 100% 16/16 [00:04<00:00,  3.51it/s]
     7/128    0.508G        1.32         1.3       0.306           1: 100% 16/16 [00:02<00:00,  6.76it/s]
     8/128    0.508G        1.27        1.57       0.333           1: 100% 16/16 [00:02<00:00,  6.99it/s]
     9/128    0.508G        1.38         1.5       0.306           1: 100% 16/16 [00:02<00:00,  6.51it/s]
    10/128    0.508G         1.3        1.39       0.278           1: 100% 16/16 [00:02<00:00,  5.73it/s]
    11/128    0.508G         1.3        1.55       0.361           1: 100% 16/16 [00:03<00:00,  4.95it/s]
    12/128    0.508G        1.28        1.45       0.306           1: 100% 16/16 [00:02<00:00,  6.98it/s]
    13/128    0.508G        1.28        1.33       0.528           1: 100% 16/16 [00:02<00:00,  6.34it/s]
    14/128    0.508G        1.24        1.19       0.417           1: 100% 16/16 [00:02<00:00,  6.90it/s]
    15/128    0.508G        1.27        1.81       0.222           1: 100% 16/16 [00:03<00:00,  4.79it/s]
    16/128    0.508G        1.25        1.52       0.361           1: 100% 16/16 [00:02<00:00,  6.45it/s]
    17/128    0.508G        1.28         1.2       0.361           1: 100% 16/16 [00:02<00:00,  6.15it/s]
    18/128    0.508G        1.25        1.33       0.528           1: 100% 16/16 [00:02<00:00,  6.79it/s]
    19/128    0.508G        1.18        1.17         0.5           1: 100% 16/16 [00:02<00:00,  6.67it/s]
    20/128    0.508G        1.23        1.33       0.306           1: 100% 16/16 [00:04<00:00,  3.52it/s]
    21/128    0.508G        1.21        1.39       0.417           1: 100% 16/16 [00:02<00:00,  6.89it/s]
    22/128    0.508G        1.18        1.36       0.528           1: 100% 16/16 [00:02<00:00,  6.43it/s]
    23/128    0.508G        1.14        1.38         0.5           1: 100% 16/16 [00:02<00:00,  6.70it/s]
    24/128    0.508G        1.17         1.3       0.556           1: 100% 16/16 [00:03<00:00,  4.59it/s]
    25/128    0.508G         1.2        1.13       0.583           1: 100% 16/16 [00:02<00:00,  6.26it/s]
    26/128    0.508G        1.11        1.12       0.528           1: 100% 16/16 [00:02<00:00,  6.69it/s]
    27/128    0.508G        1.12        1.06       0.583           1: 100% 16/16 [00:02<00:00,  6.37it/s]
    28/128    0.508G        1.12        1.45       0.417           1: 100% 16/16 [00:02<00:00,  6.95it/s]
    29/128    0.508G        1.19        1.11         0.5           1: 100% 16/16 [00:03<00:00,  4.33it/s]
    30/128    0.508G        1.14         1.2       0.583           1: 100% 16/16 [00:02<00:00,  6.86it/s]
    31/128    0.508G         1.1        1.34         0.5           1: 100% 16/16 [00:02<00:00,  5.83it/s]
    32/128    0.508G        1.17        2.32       0.278           1: 100% 16/16 [00:02<00:00,  6.40it/s]
    33/128    0.508G        1.11        1.02       0.667           1: 100% 16/16 [00:02<00:00,  5.47it/s]
    34/128    0.508G        1.16        1.37         0.5           1: 100% 16/16 [00:03<00:00,  5.17it/s]
    35/128    0.508G         1.1        1.12       0.472           1: 100% 16/16 [00:02<00:00,  6.79it/s]
    36/128    0.508G        1.08         1.2       0.556           1: 100% 16/16 [00:03<00:00,  4.22it/s]
    37/128    0.508G        1.11        1.08       0.556           1: 100% 16/16 [00:02<00:00,  6.21it/s]
    38/128    0.508G        1.13        1.26       0.528           1: 100% 16/16 [00:03<00:00,  4.65it/s]
    39/128    0.508G        1.12        1.11       0.667           1: 100% 16/16 [00:02<00:00,  6.73it/s]
    40/128    0.508G        1.11        1.19       0.639           1: 100% 16/16 [00:02<00:00,  6.53it/s]
    41/128    0.508G        1.07       0.947       0.556           1: 100% 16/16 [00:02<00:00,  6.87it/s]
    42/128    0.508G        1.07        1.18       0.611           1: 100% 16/16 [00:03<00:00,  5.17it/s]
    43/128    0.508G        1.14        1.44       0.528           1: 100% 16/16 [00:02<00:00,  5.41it/s]
    44/128    0.508G        1.05        1.01       0.667           1: 100% 16/16 [00:02<00:00,  6.64it/s]
    45/128    0.508G        1.08        1.14       0.639           1: 100% 16/16 [00:02<00:00,  6.77it/s]
    46/128    0.508G        1.07        1.33       0.528           1: 100% 16/16 [00:02<00:00,  6.31it/s]
    47/128    0.508G        1.03           1       0.639           1: 100% 16/16 [00:03<00:00,  4.78it/s]
    48/128    0.508G        1.04        1.71       0.611           1: 100% 16/16 [00:02<00:00,  5.78it/s]
    49/128    0.508G        1.04        1.64       0.528           1: 100% 16/16 [00:02<00:00,  6.66it/s]
    50/128    0.508G        1.02           1        0.75           1: 100% 16/16 [00:02<00:00,  6.63it/s]
    51/128    0.508G        1.02        1.11       0.667           1: 100% 16/16 [00:02<00:00,  6.63it/s]
    52/128    0.508G        1.06        1.59       0.611           1: 100% 16/16 [00:03<00:00,  4.26it/s]
    53/128    0.508G       0.973        1.07       0.667           1: 100% 16/16 [00:02<00:00,  6.46it/s]
    54/128    0.508G       0.925        1.34       0.556           1: 100% 16/16 [00:02<00:00,  6.46it/s]
    55/128    0.508G         1.1       0.927       0.667           1: 100% 16/16 [00:03<00:00,  4.46it/s]
    56/128    0.508G           1        1.97       0.583           1: 100% 16/16 [00:05<00:00,  3.06it/s]
    57/128    0.508G       0.993        1.34       0.611           1: 100% 16/16 [00:02<00:00,  6.75it/s]
    58/128    0.508G       0.954        1.17       0.639           1: 100% 16/16 [00:02<00:00,  6.50it/s]
    59/128    0.508G        1.03        1.54         0.5           1: 100% 16/16 [00:02<00:00,  6.59it/s]
    60/128    0.508G        1.01        1.12       0.611           1: 100% 16/16 [00:03<00:00,  5.32it/s]
    61/128    0.508G           1        1.13       0.583           1: 100% 16/16 [00:03<00:00,  5.28it/s]
    62/128    0.508G       0.943       0.986       0.639           1: 100% 16/16 [00:02<00:00,  6.75it/s]
    63/128    0.508G       0.909        1.12       0.639           1: 100% 16/16 [00:02<00:00,  6.97it/s]
    64/128    0.508G       0.888       0.867        0.75           1: 100% 16/16 [00:02<00:00,  6.32it/s]
    65/128    0.508G       0.958       0.975       0.667           1: 100% 16/16 [00:03<00:00,  4.41it/s]
    66/128    0.508G       0.939       0.947       0.639           1: 100% 16/16 [00:02<00:00,  6.54it/s]
    67/128    0.508G        1.02        1.11       0.694           1: 100% 16/16 [00:03<00:00,  5.04it/s]
    68/128    0.508G       0.998       0.971       0.667           1: 100% 16/16 [00:02<00:00,  5.55it/s]
    69/128    0.508G       0.968        0.98       0.694           1: 100% 16/16 [00:03<00:00,  4.52it/s]
    70/128    0.508G       0.965        1.11       0.722           1: 100% 16/16 [00:02<00:00,  6.55it/s]
    71/128    0.508G       0.965        1.47       0.583           1: 100% 16/16 [00:02<00:00,  6.84it/s]
    72/128    0.508G       0.953         1.2       0.611           1: 100% 16/16 [00:02<00:00,  6.54it/s]
    73/128    0.508G       0.863       0.772       0.722           1: 100% 16/16 [00:02<00:00,  6.90it/s]
    74/128    0.508G       0.946       0.884       0.667           1: 100% 16/16 [00:03<00:00,  4.25it/s]
    75/128    0.508G       0.911       0.942       0.694           1: 100% 16/16 [00:02<00:00,  6.78it/s]
    76/128    0.508G       0.964        1.16       0.694           1: 100% 16/16 [00:02<00:00,  6.80it/s]
    77/128    0.508G       0.917         1.2       0.694           1: 100% 16/16 [00:02<00:00,  6.44it/s]
    78/128    0.508G       0.941       0.955       0.639           1: 100% 16/16 [00:02<00:00,  6.22it/s]
    79/128    0.508G       0.885        1.02       0.722           1: 100% 16/16 [00:03<00:00,  4.58it/s]
    80/128    0.508G       0.864       0.802       0.694           1: 100% 16/16 [00:02<00:00,  6.33it/s]
    81/128    0.508G       0.908        1.11       0.833           1: 100% 16/16 [00:02<00:00,  6.52it/s]
    82/128    0.508G       0.915       0.843       0.778           1: 100% 16/16 [00:02<00:00,  6.82it/s]
    83/128    0.508G       0.899        1.14       0.722           1: 100% 16/16 [00:03<00:00,  4.96it/s]
    84/128    0.508G       0.826        0.81        0.75           1: 100% 16/16 [00:02<00:00,  5.77it/s]
    85/128    0.508G       0.831       0.883       0.694           1: 100% 16/16 [00:02<00:00,  6.61it/s]
    86/128    0.508G       0.804        0.95       0.694           1: 100% 16/16 [00:02<00:00,  6.42it/s]
    87/128    0.508G       0.805       0.916       0.694           1: 100% 16/16 [00:02<00:00,  6.60it/s]
    88/128    0.508G       0.824       0.936       0.667           1: 100% 16/16 [00:03<00:00,  4.40it/s]
    89/128    0.508G       0.854       0.854       0.639           1: 100% 16/16 [00:02<00:00,  6.48it/s]
    90/128    0.508G        0.79        1.14       0.694           1: 100% 16/16 [00:02<00:00,  6.72it/s]
    91/128    0.508G        0.83       0.848        0.75           1: 100% 16/16 [00:02<00:00,  6.59it/s]
    92/128    0.508G       0.805        1.32       0.639           1: 100% 16/16 [00:02<00:00,  6.47it/s]
    93/128    0.508G       0.813        1.22        0.75           1: 100% 16/16 [00:03<00:00,  4.23it/s]
    94/128    0.508G       0.796        0.91       0.722           1: 100% 16/16 [00:02<00:00,  6.68it/s]
    95/128    0.508G       0.823       0.778        0.75           1: 100% 16/16 [00:02<00:00,  6.70it/s]
    96/128    0.508G       0.827       0.898       0.806           1: 100% 16/16 [00:02<00:00,  6.50it/s]
    97/128    0.508G       0.777       0.833       0.778           1: 100% 16/16 [00:02<00:00,  5.78it/s]
    98/128    0.508G        0.79       0.735       0.806           1: 100% 16/16 [00:03<00:00,  4.78it/s]
    99/128    0.508G       0.824       0.797       0.778           1: 100% 16/16 [00:02<00:00,  6.19it/s]
   100/128    0.508G       0.802       0.893       0.806           1: 100% 16/16 [00:02<00:00,  5.94it/s]
   101/128    0.508G       0.778        1.11       0.778           1: 100% 16/16 [00:02<00:00,  6.61it/s]
   102/128    0.508G       0.795        1.15       0.722           1: 100% 16/16 [00:03<00:00,  4.30it/s]
   103/128    0.508G       0.777        1.54       0.667           1: 100% 16/16 [00:02<00:00,  6.39it/s]
   104/128    0.508G       0.764       0.916       0.722           1: 100% 16/16 [00:02<00:00,  6.66it/s]
   105/128    0.508G       0.737        1.04       0.778           1: 100% 16/16 [00:02<00:00,  6.57it/s]
   106/128    0.508G       0.689       0.792        0.75           1: 100% 16/16 [00:02<00:00,  6.55it/s]
   107/128    0.508G       0.769       0.945        0.75           1: 100% 16/16 [00:03<00:00,  4.40it/s]
   108/128    0.508G        0.78        1.21        0.75           1: 100% 16/16 [00:02<00:00,  6.61it/s]
   109/128    0.508G       0.768       0.958        0.75           1: 100% 16/16 [00:02<00:00,  6.37it/s]
   110/128    0.508G       0.802       0.953        0.75           1: 100% 16/16 [00:02<00:00,  6.41it/s]
   111/128    0.508G       0.765        0.71        0.75           1: 100% 16/16 [00:02<00:00,  5.42it/s]
   112/128    0.508G       0.709        1.07       0.722           1: 100% 16/16 [00:03<00:00,  5.15it/s]
   113/128    0.508G       0.683         1.1       0.694           1: 100% 16/16 [00:02<00:00,  6.57it/s]
   114/128    0.508G       0.685       0.892       0.778           1: 100% 16/16 [00:02<00:00,  6.41it/s]
   115/128    0.508G       0.678        0.78       0.722           1: 100% 16/16 [00:02<00:00,  6.25it/s]
   116/128    0.508G       0.714        1.19       0.722           1: 100% 16/16 [00:03<00:00,  4.29it/s]
   117/128    0.508G       0.718       0.777       0.694           1: 100% 16/16 [00:02<00:00,  6.04it/s]
   118/128    0.508G       0.744       0.855       0.778           1: 100% 16/16 [00:02<00:00,  6.72it/s]
   119/128    0.508G       0.732       0.708        0.75           1: 100% 16/16 [00:02<00:00,  6.66it/s]
   120/128    0.508G         0.7        0.88       0.778           1: 100% 16/16 [00:02<00:00,  5.85it/s]
   121/128    0.508G       0.687       0.852       0.778           1: 100% 16/16 [00:03<00:00,  4.65it/s]
   122/128    0.508G       0.671        1.01       0.778           1: 100% 16/16 [00:02<00:00,  6.46it/s]
   123/128    0.508G       0.695       0.708        0.75           1: 100% 16/16 [00:02<00:00,  6.40it/s]
   124/128    0.508G       0.685       0.725       0.778           1: 100% 16/16 [00:02<00:00,  6.69it/s]
   125/128    0.508G       0.681       0.991        0.75           1: 100% 16/16 [00:03<00:00,  4.79it/s]
   126/128    0.508G       0.674        0.72        0.75           1: 100% 16/16 [00:03<00:00,  4.96it/s]
   127/128    0.508G       0.674       0.733        0.75           1: 100% 16/16 [00:02<00:00,  6.52it/s]
   128/128    0.508G       0.687       0.682        0.75           1: 100% 16/16 [00:02<00:00,  6.48it/s]

Training complete (0.105 hours)
Results saved to runs/train-cls/exp
Predict:         python classify/predict.py --weights runs/train-cls/exp/weights/best.pt --source im.jpg
Validate:        python classify/val.py --weights runs/train-cls/exp/weights/best.pt --data Banco-Imagem-1
Export:          python export.py --weights runs/train-cls/exp/weights/best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train-cls/exp/weights/best.pt')
Visualize:       https://netron.app

CPU times: user 4.67 s, sys: 452 ms, total: 5.12 s
Wall time: 6min 43s


!python classify/val.py --weights runs/train-cls/exp/weights/best.pt --data $DATASET_NAME

classify/val: data=Banco-Imagem-1, weights=['runs/train-cls/exp/weights/best.pt'], batch_size=128, imgsz=224, device=, workers=8, verbose=True, project=runs/val-cls, name=exp, exist_ok=False, half=False, dnn=False
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

Fusing layers... 
Model summary: 117 layers, 1214869 parameters, 0 gradients, 2.9 GFLOPs
testing: 100% 1/1 [00:00<00:00,  1.05it/s]
                   Class      Images    top1_acc    top5_acc
                     all          36       0.639           1
                  avioes           7       0.571           1
                  barcos           6       0.667           1
                  carros          11       0.545           1
            helicopteros           8       0.875           1
                   motos           4         0.5           1
Speed: 0.1ms pre-process, 14.8ms inference, 0.6ms post-process per image at shape (1, 3, 224, 224)
Results saved to runs/val-cls/exp



```
</details>
 
### Evidências do treinamento

#### Inferindo com o modelo personalizado
```
#Pega a localização de uma imagem do conjunto de testes ou validações 
if os.path.exists(os.path.join(dataset.location, "test")):
  split_path = os.path.join(dataset.location, "test")
else:
  os.path.join(dataset.location, "valid")
example_class = os.listdir(split_path)[4]
example_image_name = os.listdir(os.path.join(split_path, example_class))[4]
example_image_path = os.path.join(split_path, example_class, example_image_name)
os.environ["TEST_IMAGE_PATH"] = example_image_path

print(f"Inferindo sobre um exemplo da classe '{example_class}'")

#Infer
!python classify/predict.py --weights runs/train-cls/exp/weights/best.pt --source $TEST_IMAGE_PATH

Inferindo sobre um exemplo da classe 'carros'
classify/predict: weights=['runs/train-cls/exp/weights/best.pt'], source=/content/yolov5/Banco-Imagem-1/test/carros/00012_jpg.rf.9f0d32646e83139878c5788b040038f7.jpg, data=data/coco128.yaml, imgsz=[224, 224], device=, view_img=False, save_txt=False, nosave=False, augment=False, visualize=False, update=False, project=runs/predict-cls, name=exp, exist_ok=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

Fusing layers... 
Model summary: 117 layers, 1214869 parameters, 0 gradients, 2.9 GFLOPs
image 1/1 /content/yolov5/Banco-Imagem-1/test/carros/00012_jpg.rf.9f0d32646e83139878c5788b040038f7.jpg: 224x224 carros ```**0.91**```, avioes 0.08, motos 0.01, helicopteros 0.00, barcos 0.00, 2.7ms
Speed: 0.3ms pre-process, 2.7ms inference, 5.1ms NMS per image at shape (1, 3, 224, 224)
Results saved to runs/predict-cls/exp14
```



```
#### Modelo treinado com 80% ou mais de acurácia/precisão
=========================================================
```
![Descrição](https://i.imgur.com/GB9Tihf.jpg)

```
#carro
import requests
image_url = "https://i.imgur.com/GB9Tihf.jpg"
response = requests.get(image_url)
response.raise_for_status()
with open('carro.jpg', 'wb') as handler:
    handler.write(response.content)

!python classify/predict.py --weights ./weigths/yolov5x-cls.pt --source carro.jpg


classify/predict: weights=['./weigths/yolov5x-cls.pt'], source=carro.jpg, data=data/coco128.yaml, imgsz=[224, 224], device=, view_img=False, save_txt=False, nosave=False, augment=False, visualize=False, update=False, project=runs/predict-cls, name=exp, exist_ok=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

Fusing layers... 
Model summary: 264 layers, 48072600 parameters, 0 gradients, 129.9 GFLOPs
image 1/1 /content/yolov5/carro.jpg: 224x224 sports car 0.95, race car 0.02, convertible 0.01, car wheel 0.00, grille 0.00, 12.9ms
Speed: 0.4ms pre-process, 12.9ms inference, 6.9ms NMS per image at shape (1, 3, 224, 224)
Results saved to runs/predict-cls/exp13

### Modelo treinado com ao menos 50% de acurácia/precisão
=========================================================
```
![Descrição](https://i.imgur.com/ASwjAT5.jpg)
```

#Moto
import requests
image_url = "https://i.imgur.com/ASwjAT5.jpg"
response = requests.get(image_url)
response.raise_for_status()
with open('moto.jpg', 'wb') as handler:
    handler.write(response.content)
    
!python classify/predict.py --weights ./weigths/yolov5m-cls.pt --source moto.jpg

classify/predict: weights=['./weigths/yolov5m-cls.pt'], source=moto.jpg, data=data/coco128.yaml, imgsz=[224, 224], device=, view_img=False, save_txt=False, nosave=False, augment=False, visualize=False, update=False, project=runs/predict-cls, name=exp, exist_ok=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 v7.0-230-g53efd07 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

Fusing layers... 
Model summary: 166 layers, 12947192 parameters, 0 gradients, 31.7 GFLOPs
image 1/1 /content/yolov5/moto.jpg: 224x224 moped 0.64, scooter 0.17, disc brake 0.06, crash helmet 0.05, snowmobile 0.01, 5.4ms
Speed: 0.4ms pre-process, 5.4ms inference, 6.9ms NMS per image at shape (1, 3, 224, 224)
Results saved to runs/predict-cls/exp16

```

## Roboflow

Banco-Imagem > 2023-10-24 9:29pm

https://universe.roboflow.com/eniokilder/banco-imagem

Provided by a Roboflow user
License: CC BY 4.0


## HuggingFace

Link para o HuggingFace:  Não há
