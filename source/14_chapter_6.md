# Análisis de los resultados obtenidos {#resultados}
A continuación se analizará el funcionamiento de los sistemas
descritos en el capítulo anterior. En todos los casos, el dataset
original ha sido dividido en 3 subgrupos, obteniendo así:

- **Dataset de entrenamiento**
- **Dataset de validación**
- **Dataset de  test**

Los resultados que se presentarán durante este capítulo son los
correspondientes a los datasets de validación y, en algunos casos, los
de test, puesto que la evaluación del clasificador con el conjunto de
datos de entrenamiento no es relevante para conocer la capacidad de
**generalización** de los sistemas. El dataset de validación nos ha
permitido evaluar el rendimiento de diferentes **arquitecturas e
hiperparámetros** mientras que el dataset de test nos ha permitido
dar, al final del proceso, una evaluación del sistema final con datos
nuevos y comprobar el funcionamiento del **Sistema de Predicción e
Interpretación**. Las métricas proporcionadas serán siempre las del
**mejor epoch**, es decir, el que ha obtenido el mínimo valor de
pérdidas con el dataset de validación.

## Evaluación del Sistema 1: Gran Clasificador
Como ya se anunciaba en el capítulo \ref{sistema}, debido al gran
desbalanceo existente este las clases, este sistema **no nos ha
permitido distinguir correctamente entre las 3 clases**. Tras
evaluarse varios valores distintos para el **learning rate**, **batch
size** e incluso añadirse pesos a las clases que compensaran el
desbalanceo existente, el proceso de descenso de gradiente ha quedado
constantemente *atrapado* en mínimos locales en los que el
clasificador predice la misma clase para todas las
instancias. Concluimos, por lo tanto, que **un solo clasificador no
tiene capacidad suficiente para obtener patrones de un conjunto de
datos tan desbalanceado** y habrá que buscar soluciones alternativas.

## Evaluación del Sistema 2: Clasificador Multietapa
El segundo sistema consta de 2 etapas: la clasificación
**Sano/Enfermo** y la clasificación **RD/DMAE**. A continuación, se
analizarán ambas etapas.


### Etapa 1: Clasificador Sano/Enfermo
Las 39118 imágenes de las que se componía nuestro cojunto de imágenes
inicial han sido divididas en 2 grupos: **Retinas Sanas y Retinas
Enfermas**.

Puesto que ha sido utilizada la técnica de **Transfer Learning**, se
han realizado varias ejecuciones descongelando, progresivamente cada
uno de los bloques convolucionales de la arquitectura utilizada,
**VGG16**. Esta arquitectura posee 5 bloques convolucionales con
**capas de Pooling** en cada uno de estos bloques.

Inicialmente se han realizado varios entrenamientos del bloque **Fully
Connected** para testear los posibles **batch size** y **learning
rate**. En la tabla \ref{batch} podemos ver los resultados obtenidos
para distintos **batch sizes**.  Las evaluaciones de la tabla son las
del dataset de validación para el mejor epoch ^[el que tiene menores
pérdidas con el dataset de validación]. El **learning rate** utilizado
ha sido de 0.0001.

------------------------------------
 batch size   accuracy     loss
-----------  ----------  ---------
     16       0.7062      0.5973

     32       0.6782      0.6151

     64       0.7072      0.6073
-------------------------------------

Table: Resultados del entrenamiento para distintos batch size. Modelos
evaluados con el dataset de validación \label{batch}

A partir de la tabla \ref{batch} se puede intuir que el tamaño del
batch size no juega un papel de gran importancia en el proceso de
entrenamiento por lo que se ha decidido usar, a partir de este momento
un tamaño de **64**. Este tamaño nos permite entrenar la red de forma
más rápida y nos asegura que en cada batch exista suficiente cantidad
de imágenes de las dos clases. Usar tamaños superiores habría
ocasionado problemas de memoria en la GPU utilizada.

De la misma forma, como vemos en la tabla \ref{lr}, también se han
realizado varios entrenamientos del clasificador que nos han permitido
comprobar cuál es el **learning rate** adecuado para nuestro
problema. Para ello se ha utilizado un **batch size de 64**.

---------------------------------------------------------------------------
 learning rate    accuracy      loss
---------------- -----------  ---------
     0.001         0.287         11.4

     0.0001        0.6782       0.6151

     0.00005       0.7199       0.6037
---------------------------------------------------------------------------

Table: Resultados del entrenamiento para distintos learning
rate. Evaluados con el dataset de validación \label{lr}

Como se ha podido comprobar, utilizar un **learning rate** demasiado
alto provoca que el descenso de gradiente quede *atrapado* en mínimos
locales o no sea capaz de converger, dando lugar a unos valores de
pérdidas demasiado altos. El **learning rate de 0.0005** es el que nos
da mejores resultados y por lo tanto ha sido utilizado en los
posteriores entrenamientos. Este valor tendrá que disminuirse
ligeramente cuando entrenemos varias capas convolucionales a la vez
para asegurarnos un descenso de gradiente lento que pueda converger en
un mínimo absoluto.

Una vez decididos los hiperparámetros se ha comenzado a *descongelar*
los diversos bloques de las capas convolucionales de la red,
obteniendo los resultados de la tabla \ref{training}

---------------------------------------------------------------------------
train blocks       LR      accuracy      loss    sensitivity    specifity
----------------  ----     --------    --------  -----------    ---------
      FC          5e-5      0.7149      0.6549     0.1713         0.9338

 Bloque 5         5e-5      0.7532      0.5294     0.3805         0.9012

 Bloques 4,5      5e-6      0.7126      0.6854        0              1

 Bloques 3,4,5    5e-6      0.7880      0.4748     0.4793         0.9131

 Bloques 2,3,4,5  5e-6      0.7961      0.4704     0.5867         0.8624

 Todos            5e-6      FALTA
---------------------------------------------------------------------------

Table: Resultados del entrenamiento para distintos bloques
convolucionales entrenados. Modelos evaluados con el dataset de
validación \label{training}


Como podemos ver en la tabla \ref{training}, **los mejores resultados
se han obtenido al entrenar todos los bloques convolucionales menos el
primero**, obteniendo una *accuracy* de casi el 80%. Será este, por lo
tanto, el clasificador que se usará en el Sistema de Predicción e
Interpretación en esta segunda etapa del Clasificador Multietapa. Cabe
destacar que, a partir del entrenamiento del bloque 4, hemos tenido
que disminuir el learning rate para evitar caer en mínimos locales y
asegurar la convergencia.

Debido a la gran carga computacional que ha supuesto entrenar este
modelo (cada entrenamiento ha durado una media de 72 horas),
únicamente se ha evaluado la arquitectura **VGG16**.

En la figura \ref{valloss} podemos ver la progresión de las pérdidas
durante el entrenamiento del clasificador final elegido para esta
etapa. Como vemos, **las mínimas pérdidas se obtienen alrededor del
epoch 45**. A partir de ese momento, la red empieza a sufrir de
**overfitting** y las pérdidas con el dataset de validación aumentarán
mientras que las del dataset de entrenamiento continuarán
descendiendo. Esto es un claro indicador de que la red está comenzando
a **memorizar** el dataset de entrenamiento en vez de detectar
patrones. Por lo tanto, nos quedaremos con el estado de la red en ese
epoch 45.

![Pérdidas para el dataset de validación del entrenamiento de los
bloques 2,3,4 y 5 (y FC) del clasificador Sano/Enfermo. Se ha aplicado
un filtro de
suavizado. \label{valloss}](source/figures/loss1.png){width=100%}


En las figuras \ref{hnh1}, \ref{hnh2} y \ref{hnh3} podemos ver
ejemplos de la respuesta proporcionada por esta etapa del sistema.


![Salida de la primera etapa del Sistema Multietapa para una imagen de
una retina enferma de RD
\label{hnh1}](source/figures/hnh1.png){width=80%}

![Salida de la primera etapa del Sistema Multietapa para una imagen de
una retina enferma de DMAE
\label{hnh2}](source/figures/hnh2.png){width=80%}

![Salida de la primera etapa del Sistema Multietapa para una imagen de
una retina sana
\label{hnh3}](source/figures/hnh3.png){width=80%}


\newpage

### Etapa 2: Clasificador RD/DMAE
La tabla \ref{training2} muestra los resultados del entrenamiento de
la arquitectura VGG16 para la segunda etapa del **Sistema
Multietapa**. La función de este clasificador era diferenciar, de
entre las imágenes de retinas detectadas como enfermas en la etapa 1,
cuáles son de Retinopatía Diabética y cuáles de Degeneración Macular
Asociada a la Edad.


---------------------------------------------------------------------------
train blocks       LR      accuracy      loss
----------------  ----     --------    --------
     FC           1e-5      0.9231      0.1910

 Bloque 5         1e-5      0.9359      0.1483

 Bloques 4,5      1e-5      0.8958      0.2093

 Bloques 3,4,5    1e-5      0.9615      0.1443

 Bloques 2,3,4,5  1e-5      0.9103      0.1773

 Todos            1e-5      0.9487      0.1691
---------------------------------------------------------------------------

Table: Resultados del entrenamiento de la segunda etapa del Sistema
Multietapa. Evaluados con el dataset de validación \label{training2}

Además, como vemos en la tabla \ref{training3}, también se han
entrenado otras arquitecturas. En este caso se han entrenado todos los
bloques convolucionales de las mismas.

---------------------------------------------------------------------------
Arquitectura       LR      accuracy      loss
----------------  ----     --------    --------
 InceptionV3      1e-5      0.9167       0.263
 Resnet50         1e-5      0.9744      0.0816
 Xception         1e-5      FALTA
---------------------------------------------------------------------------

Table: Resultados del entrenamiento de la segunda etapa del Sistema
Multietapa. Evaluados con el dataset de validación \label{training3}

Como vemos, los resultados en esta segunda etapa son mucho más
satisfactorios que los de la primera, obteniéndose la máxima
**accuracy** (97.4%) con la arquitectura Resnet50, que será la
utilizada para el **Sistema de Predicción e Interpretación**.

En las figuras \ref{dr1}, \ref{dr2} y \ref{dr3} podemos ver ejemplos
de la respuesta proporcionada por esta etapa del sistema.


![Salida de la segunda etapa del Sistema Multietapa para una imagen de
una retina enferma de RD
\label{dr1}](source/figures/dr1.png){width=80%}

![Salida de la segunda etapa del Sistema Multietapa para una imagen de
una retina enferma de DMAE
\label{dr2}](source/figures/dr2.png){width=80%}


\newpage
## Evaluación del Sistema 3: Ensemble de Clasificadores

En este caso, se han probado 3 arquitecturas distintas: VGG16, ResNet
e InceptionV3. Como se ha explicado anteriormente, cada una de estas tres
arquitecturas ha sido entrenada con un subconjunto distinto de los
datos. De esta forma obtenemos clasificadores no correlados que, al
ser combinados, permiten obtener un rendimiento superior al de cada
uno de ellos de forma individual.

Para la arquitectura VGG16, al utilizarse la técnica del **Transfer
Learning**, se han evaluado diferentes versiones, congelando cada vez
distinto número de capas como vemos en la tabla \ref{training4}.

---------------------------------------------------------------------------
train blocks       LR      accuracy      loss
----------------  ----     --------    --------
 FC               5e-5      0.6695      0.6676
 Bloque 5         5e-5      0.7458      0.6068
 Bloques 4,5      5e-6      0.7119      0.5878
 Bloques 3,4,5    5e-6      0.7119      0.5776
 Bloques 2,3,4,5  5e-6      0.7797      0.5456
 Todos            5e-6      0.7458      0.5973
---------------------------------------------------------------------------

Table: Resultados del entrenamiento para distintos learning
rage. Evaluados con el dataset de validación \label{training4}

Como se puede comprobar, el mejor resultado lo obtenemos de nuevo
dejando congelado el primer bloque convolucional y entrenando el resto
de bloques.

En la tabla \ref{training5} se muestran los resultados del
entrenamiento para otras arquitecturas.


---------------------------------------------------------------------------
Arquitectura       LR      accuracy      loss
----------------  ----     --------    --------
 InceptionV3      1e-5
 Resnet50         1e-5
 Xception         1e-5
---------------------------------------------------------------------------

Table: Resultados del entrenamiento de la segunda etapa del Sistema
Multietapa. Evaluados con el dataset de validación \label{training5}

## Sistema de Predicción e Interpretación

Los modelos de

HABLAR DEL PROBLEMA DE LAS IMAGENES DE KAGGLE
