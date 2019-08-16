# Diseño de sistema para la clasificación automática de RD y DMAE {#sistema}
La gran cantidad de conjuntos de imágenes utilizados y el extremo
desbalanceo de las clases han sido los dos factores que más han
condicionado el diseño de los sistemas. Esto ha provocado que se hayan
realizado 3 aproximaciones distintas al problema, todas ellas basadas
en Deep Learning, y únicamente obteniendo resultados útiles de 2 de
ellas. Durante este capítulo se analizarán estas tres
aproximaciones. Este análisis no se limitará a una simple descripción
del clasificador utilizado, sino que analizará los principales
aspectos de cualquier proyecto de este tipo: los **datos** usados, su
limpieza y procesado, el proceso de selección de **hiperparámetros** o
incluso las características y limitaciones impuestas por los
**recursos hardware y software** utilizados.

## Exploración de los datos
Una de las principales contribuciones de este trabajo ha sido
precisamente la **extensa cantidad de conjuntos distintos de
imágenes** utilizados en la creación de los modelos. Para entrenar
nuestro modelo se han seleccionado imágenes de prácticamente todos los
datasets utilizados por los modelos del capítulo \ref{arte}. En total
han sido 13 los conjuntos de imágenes utilizados, haciendo un total de
39118 imágenes^[Sin embargo, algunos de los clasificadores que se han
realizado no han utilizado el conjunto completo de imágenes para el
entrenamiento.]. Destaca el dataset **Kaggle** que contiene el 66% del
total de imágenes utilizadas. Este dataset proviene de una competición
[^https://www.kaggle.com/c/diabetic-retinopathy-detection/] realizada
en 2015 que supuso importantes avances en la detección de Retinopatía
Diabética en imágenes de fondo de ojo.

Como se ha visto, la cantidad total de imágenes es muy elevada, siendo
muy superior al tamaño medio de los datasets de los modelos analizados
en el capítulo \ref{arte}. Como se verá a continuación, algunos de los
modelos que se han creado han utilizado grupos más reducidos de
imágenes seleccionadas aleatoriamente del conjunto de datos
original. En la tabla \ref{datasets0} podemos ver la cantidad de
imágenes de cada clase en los datasets utilizados.

---------------------------------------------------------------------------
Dataset                             SANA               RD             DMAE
---------------               -----------     ------------   --------------
GRAND-CHALLENGE                     311                 0               89

ARIA                                 61                59               23

DIARET DB0                           20               110                0

E-OPTHA                             268               195                0

HEI-MED                               0               169                0

HRF                                  15                15                0

KAGGLE                            25810              9316                0

MESSIDOR                            540               660                0

ONHSD                                 0                99                0

ROC                                   0                50                0

DIAGNOS                              23                 0               22

STARE                                37                89               47

FOM                                 533               457              101

---------------------------------------------------------------------------

Table: Cantidad de imágenes de cada clase en cada uno de los conjuntos
de imágenes utilizados \label{datasets0}


Haber utilizado todos estos datasets ha supuesto una dificultad
añadida al proceso, pues se ha tenido que hacer un costoso trabajo
previo de **selección, limpieza y preparación de los datos**. Para
ello, se han realizado una serie de scripts que han recorrido cada una
de las carpetas y han separado las imágenes de cada tipo.


Como se puede observar en los datos de la tabla \ref{datasets1}, el
gran problema del conjunto de imágenes utilizado, que ha condicionado
en gran medida la forma de trabajar con él, es el gran **desbalanceo**
existente entre las clases. La clase predominante, las imágenes de
retinas sanas, contiene **más del 70%** del total de imágenes. Por el
contrario, la clase minoritaria, las imágenes de retinas con DMAE,
únicamente contiene 281 imágenes, **menos del 1%**. Este gran
desbalanceo nos obligará, como se analizará durante este capítulo a
aplicar diversas técnicas que permitan compensarlo como la asignación
de pesos a los objetivos o el submuestreo de las clases.


---------------------------------------------------------------------------
    Clase                Total de imágenes        % del dataset completo
---------------        ---------------------     -------------------------
    Todas                     39118                        100

    Sanas                     27618                       70.60

    RD                        11219                       28.68

    DMAE                        281                        0.72

---------------------------------------------------------------------------

Table: Cantidad de imágenes, de cada clase, en el conjunto de datos
utilizado \label{datasets1}

Otra dificultad derivada del uso de 13 datasets distintos es que
nuestro clasificador tendrá que tratar imágenes con características
muy distintas, como vemos en la tabla \ref{datasets2}. Las condiciones
en las que han sido tomadas, procesadas y almacenadas las imágenes
varían en gran medida entre los distintos datasets. Sin embargo, si se
pretende crear un clasificador robusto que sea capaz de trabajar en
diversos tipos de condiciones, utilizar esta elevada cantidad de
conjuntos de imágenes será de gran ayuda.

---------------------------------------------------------------------------------------------------------
        Dataset                                 Origen                      Tamaño       Formato
-------------------------------      -----------------------------        ---------     ---------
 GRAND-CHALLENGE                            [@ichallenge]                   Varios          JPG

 ARIA                                    [@zheng2012automated]             576x768          TIFF
                                       [@farnell2008enhancement]

 DIARET DB0                            [@kauppi2006diaretdb0]             1500x1152         PNG

 E-OPTHA                               [@decenciere2013teleophta]           Varios          JPG

 HEI-MED                                [@giancardo2012exudate]           2196×1958         JPG

 HRF                                   [@odstrcilik2013retinal]           3504×2336         JPG

 KAGGLE                                 [@cuadros2009eyepacs]               Varias          JPG

 MESSIDOR                             [@decenciere2014feedback]           2240×1488         TIFF

 ONHSD                                   [@lowell2004optic]                760×570          BMP

 ROC                                  [@niemeijer2009retinopathy]           Varias          JPG

 AMD DIAGNOS                                 Privada                        Varias          JPG

 STARE                                  [@hoover1998locating]              605x700          TIFF

 FOM                                        Privada                         Varias          JPG

-------------------------------------------------------------------------------------------------------

Table: Características de las imágenes de cada uno de los conjuntos
utilizados \label{datasets2}


Como puede verse en la tabla \ref{datasets2} algunas de las bases de
datos utilizadas no son bases de datos públicas sino que han sido
facilitadas por diversas instituciones al **Computer Vision and
Behaviour Analysis Lab (CVBLab)**^[http://www.cvblab.webs.upv.es] de
la Universidad Politécnica de Valencia en el contexto del proyecto
**Acrima**
^[http://www.cvblab.webs.upv.es/project/acrima_en/]. Gracias a estas
bases de datos privadas, se ha podido contar con un conjunto de
imágenes de DMAE suficientemente grande como para aplicar técnicas de
Deep Learning.


## Recursos utilizados
Respecto al **software** utilizado, la librería Open Source
**Keras**^[https://keras.io/] ha sido la elegida para la programación
de los clasificadores. Keras es una librería de Deep Learning de alto
nivel en Python. Keras permite realizar, de forma rápida y simple,
diversos tipos de redes neuronales. Además, es capaz de trabajar sobre
varios frameworks de más bajo nivel:
Theano ^[https://github.com/Theano/Theano],
CNTK ^[https://github.com/Microsoft/cntk] o
**Tensorflow**^[https://github.com/tensorflow/tensorflow]. Será éste
último, el framewok sobre el que crearemos nuestros modelos con Keras
(figura \ref{keras}). Otra importante característica de Keras a tener
en cuenta es que permite la ejecución tanto en CPU como en GPU, sin
necesidad de modificar para ello el código.


![Logos de las dos librerías utilizadas: Keras y
Tensorflow. \label{keras}](source/figures/keras.jpeg){width=100%}


También se ha hecho uso de las Jupyter Notebooks
^[https://jupyter.org/] (figura \ref{jupyter}), entornos de trabajo
que permiten la creación de documentos que combinen fragmentos de
código con texto, imágenes e incluso elementos interacivos. Las
Jupyter Notebooks han ayudado a mostrar de forma simple y ordenada al
usuario los resultados de las predicciones realizadas a partir de las
imágenes de fondo de ojo proporcionadas.


![Ejemplos de Jupyter Notebooks. Fuente: https://jupyter.org/
\label{jupyter}](source/figures/jupyter.png){width=100%}

En relación con el **hardware**, como todo proyecto de Deep Learning,
la tarjeta gráfica utilizada durante el entrenamiento ha jugado un
papel fundamental. En proyectos con grandes conjuntos de imágenes,
como es el caso, no disponer de tarjeta gráfica (o disponer de una
tarjeta sin la suficiente capacidad de procesamiento) puede hacer
inviable el entrenamiento de los modelos. En este caso, la tarjeta
gráfica utilizada ha sido la **NVIDIA TITAN Xp**. Esta tarjeta cuenta
con una **arquitectura Pascal** con una frecuencia de reloj de 1481
MHz, y 12 GB de memoria. La potencia de cómputo de la NVIDIA TITAN Xp
es de 12.15 TFLOPS. Además, parte del procesamiento en local se ha
realizado en un **MacBook Air** con un procesador **Intel Core i7 de
2.2 GHz** y 8 GB de memoria RAM.


A continuación se presentarán las características de los sistemas de
clasificación realizados. Estos sistemas tenían como propósito la
detección de imágenes de retinas sanas, enfermas de Retinopatía
Diabética o enfermas de Retinopatía Diabética asociada a la edad.  Sin
embargo, en ningún momento ha sido objetivo de este trabajo la
detección de los diferentes niveles de gravedad de ambas patologías,
principalmente debido a la falta de suficientes conjuntos de imágenes
que proporcionen esta información. Los 3 sistemas presentados a
continuación son totalmente independientes.

## Diseño del sistema 1: Gran clasificador
El primer sistema realizado se trataba de una CNN basada en la
arquitectura VGG16 que trataba de distinguir entre los 3 tipos de
imágenes (RD, DMAE, Sanas). Para entrenar esta red se utilizaron las
39118 imágenes que habían sido recopiladas previamente. Se realizaron
diversas ejecuciones, alterando diversos parámetros como el números de
capas Fully Conencted y la cantidad neuronas en cada una, el **batch
size** o el **learning rate** (factor de aprendizaje).


El gran desbalanceo existente entre las clases, como se presentaba en
la tabla \ref{datasets1} supuso que este diseño, como se verá en el
siguiente capítulo no fuera capaz de detectar correctamente las 3
clases que componen nuestro problema y, por lo tanto, fue desechado.


## Diseño del sistema 2: Clasificador en 2 etapas
Los resultados del primer sistema pusieron de manifiesto la necesidad
de aplicar técnicas que trataran el problema del desbalanceo. Por
ello, el segundo sistema realizado constaba de dos clasificadores
binarios en cascada:

   - El primer clasificador diferenciaba entre retinas sanas y retinas
     enfermas (Sin distinguir entre Retinopatía Diabética o
     Degeneración Macular).

   - El segundo clasificador diferenciaba, de entre las imágenes de
     retinas detectadas como enfermas en el paso anterior, si el
     paciente estaba afectado de RD o de DMAE.


Gracias a este sistema basado en dos etapas se conseguía un
desbalanceo entre clases, en cada una de las etapas inferior al del
conjunto original de imágenes. En la figura \ref{design2} podemos ver
un diagrama de este sistema. El primer clasificador hacía uso del
conjunto completo de imágenes pero, como se ha comentado, únicamente
distinguía entre dos posibles casos, retinas sanas y retinas
enfermas. En la table \ref{c1s2} vemos la distribución de las imágenes
del clasificador en los conjuntos de entrenamiento, validación y test.

![Arquitectura del sistema clasificador en dos etapas Elaboración
propia \label{design2}](source/figures/design2.png){width=100%}

---------------------------------------------------------------------------
Conjunto         Clase            Total          Total(%)
-------    -----------     ------------   --------------
train             sana            17610            70.34

train          enferma             7425            29.66

valid             sana             4463            71.31

valid          enferma             1796            28.69

test              sana             5544            70.86

test           enferma             2280            29.14

---------------------------------------------------------------------------

Table: Distribución de las imágenes del clasificador Sanas/Enfermas
del sistema 2. \label{c1s2}

Para tratar el desbalanceo existente en esta primera etapa, se ha
hecho uso de una técnica basada en aplicar a las instancias de la
clase minoritaria (en este caso, la clase **enferma**) durante el
entrenamiento, un peso que compense el desbalenceo.


Para el segundo clasificador la aproximación ha sido distinta. Si se
hubiera usado el conjunto de imágenes completo el desbalanceo hubiera
demasiado grande, imposible de abordar incluso por la técnica
utilizada anteriormente. Por lo tanto, esta segunda etapa consta de
varios clasificadores, siendo cada uno de ellos entrenado por un
subconjunto de X imágenes tomadas aleatoriamente del conjunto de datos
original. La predicción final que tendremos a la salida de esta segunda etapa será


Arquitectura
Data Augmentation
Balanceo de clases
Hiper parametros


## Diseño del sistema 3: Ensemble de clasificadores simples
## Diseño del software de predicción y análsis



### Tratamiento del desbalanceo de clases


\newpage




---------------------------------------------------------------------------
    Grupo                Total de imágenes        % del dataset completo
---------------        ---------------------     -------------------------

 Entrenamiento                7361                        64

   Validación                 1841                        16

    Test                      2300                        20

---------------------------------------------------------------------------


Table: This is the table caption. Suspendisse blandit dolor sed tellus
venenatis, venenatis fringilla turpis pretium. \label{datasets2}


\newpage

---------------------------------------------------------------------------
Conjunto         Clase            Total          Total(%)
-------    -----------     ------------   --------------
train               dr             7195            97.74

train              amd              166             2.26

valid               dr             1786            97.01

valid              amd               55             2.99

test                dr             2240            97.39

test                amd              60             2.61

---------------------------------------------------------------------------

Table: This is the table caption. Suspendisse blandit dolor sed tellus
venenatis, venenatis fringilla turpis pretium. \label{datasets3}


### Data Augmentation
Gracias al uso del **Data Augmentation**...




### Arquitecturas y entrenamiento
Para el entrenamiento de ambos clasificadores se ha hecho uso de la
técnica de **Transfer Learning** explicada anteriormente.

## Mejorando la interpretabilidad
