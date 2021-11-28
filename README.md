# MinneApple: A Benchmark Dataset for Apple Detection and Segmentation
Este codigo se trabajo bajo el proyecto original del articulo [paper](https://arxiv.org/abs/1909.06441) MinneApple: A Benchmark Dataset for Apple Detection and Segmentation. Aqui se encuentran trabajados modelos para leer el dataset original  y entrenar en base a este a modelos Faster RCNN y Mask RCNN con un enfoque en el reconocimiento de manzanas.

## Resultados Originales
Los resultados a continuacion son del 1 Noviembre de 2019 que se dejan con el objetivo de hacer una comparativa con los resultados obtenidos.

### Deteccion de Manzanas.
| Metodo | Backbone | AP @ IoU=.50:.05:.95 | AP @ IoU=.50  |  AP @ IoU=.74 | AP_small | AP_medium | AP_large|
|---|---|---|---|---|---|---|---|
| Tiled Faster RCNN  |  ResNet50  |  0.341 | 0.639 | 0.339  | 0.197  | 0.519  | 0.208  |
| Faster RCNN  |  ResNet50 |  0.438 |  0.775 | 0.455  | 0.297  | 0.578  | 0.871  |
| Mask RCNN  | ResNet50  | 0.433  |  0.763 | 0.449  | 0.295  | 0.571  |  0.809 |

#### Comentarios
El proyecto del articulo original se encuentra en una competencia para obtener mejores resultados para el reconocimiento, conteo y estimacion de cosecha de las manzanas con el data set original, ante esto, el dataset que entregan desde la pagina no cuenta con la misma cantidad de fotos que el proyecto original desde el cual se obtuvieron los resultados de mas arriba. Como el objetivo del proyecto trabajado era academico y no un proyecto completo, se modifico el dataset diminutivo que entregan del proyecto original en 70/30 para el entrenamiento y testo, debido a que no se entrega un mask completo para hacer el testeo.
El 30% del dataset con mask se uso para el testeo con ese objetivo.

## Instalacion
* Levantar un ambiente de trabajo en Python 3
* Instalar Jupyter Notebok y levantar un venv para trabajar (altamente recomendado)
* Instalar Pytorch (1.0.1 o superior)
* Instalar TorchVision
* Instalar PYCOCOTOOLS (localmente y en el venv)
* Instalar el siguiente packete (necesario):
```
# Install necessary packages
pip install Pillow opencv-python sklearn numpy
```


## Cargar datos
El archivo ```data/apple_dataset.py``` es un archivo original del proyecto base que tiene una clase personalizada del dataset que permite cargar imagenes y masks sin complicaciones, extrae bounding boxes y masks de segmentacion. Se recomienda revisar las direcciones de carpetas en el archivo ya que pueden causar problemas con la maquina.

## Dataset
Para entrenar los modelos se necesita descargar el dataset, de preferencia seria el original pero debido a que este no se encuentra abierto al publico de manera rapida se recomienda la version modificada para acceso al publico que es mas pequena y no cuenta con todos los masks desde aqui [here](https://drive.google.com/drive/folders/1IWRPyXBwMcblBzQaxPB6ZaRWYTMJ6O8z?usp=sharing) donde tambien estan incluidos algunos modelos entrenados para este proyecto para faster rcnn y mask rcnn.

## Entrenamiento
* A continuacion se dejan los comandos para ejecutar los modelos para entrenar desde la consola.
```
# Entrenar Faster RCNN
python train_rcnn.py --data_path /directorio/al/dataset --model frcnn --epochs 50 --output-dir /directorio/donde-quieras/guardar/los-modelos-resultados
# Entrenar Mask RCNN
python train_rcnn.py --data_path /directorio/al/dataset --model mrcnn --epochs 50 --output-dir /directorio/donde-quieras/guardar/los-modelos-resultados

```
La cantidad de epochs originales que se dicen se utilizaron en el proyecto base es de 50 por lo que se recomienda utilizar ese valor, toma un tiempo aproximado entre 2 a 3 horas dependiendo de la maquina donde se ejecute el proyecto.

## Resultados Obtenidos

| Metodo | Backbone | AP @ IoU=.50:.05:.95 | AP @ IoU=.50  |  AP @ IoU=.74 | AP_small | AP_medium | AP_large|
|---|---|---|---|---|---|---|---|
| Faster RCNN  |  ResNet50 |  0.329 |  0.68 | 0.285  | 0.255  | 0.527  | -1  |
| Mask RCNN  | ResNet50  | 0.343 |  0.716 | 0.293  | 0.278 | 0.529  |  -1 |
