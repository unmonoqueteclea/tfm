# Estado del arte en detección de RD y DMAE {#arte}
<!-- Comienzo: Sabado, 9 de Junio
     Previsión:
     Papers útiles:  Practicamente todos (P5 es resumen)
     Fin previsto versión 1: Domingo
     Fin versión 1: 16 Junio
-->

Durante este capítulo, se analizarán los principales modelos hasta la
fecha de detección de Retinopatía Diabética y Degeneración Macular
Asociada a la Edad a partir de imágenes de fondo de ojo. En la
actualidad, prácticamente la totalidad de los modelos del estado del
arte en este campo, son modelos de Deep Learning. Sin embargo,
comenzaremos haciendo un pequeño análisis de los modelos que
precedieron a los actuales, basados en Machine Learning.


## Aproximaciones basadas en Machine Learning
<!-- Papers utiles: (P37) (P38) (P39) (P40) -->

Las modelos basados en Machine Learning para la detección de
patologías en imágenes de fondo de ojo requerían de una gran cantidad
de características elegidas de forma manual por los
investigadores. Para la obtención de las mismas, era necesario
conocimiento experto en la materia. Además, este tipo de
características tienden a no generalizar bien dando lugar a modelos
que, cuando se usaban con datos de distintos datasets, tenían una
eficacia mucho menos que la esperada.


### Detección de RD mediante Machine Learning
Este tipo técnicas se basaban en la búsqueda en las imágenes de cada
una de las lesiones que caracterizan la Retinopatía Diabética. Las
lesiones que caracterizan la RD, como se ha analizado anteriormente,
son: exudados, microaneurismas y hemorragias. A partir de estas
características, obtenidas principalmente mediante técnicas de
procesamiento digital de imágenes, y gracias al uso de clasificadores
basados en Machine Learning es posible detectar la enfermedad, e
incluso, estimar su gravedad.

Muchas de estas técnicas, comenzaban por la obtención de imágenes
binarias que representaran los **vasos sanguíneos** presentes en la
imagen de la retina. La longitud, tamaño o posición de los mismos son
de gran ayuda para el diagnóstica de la RD. Mediante la aplicación de
una serie de técnicas al canal verde de las imágenes de fondo de ojo,
es posible aislar estos vasos del resto de la imagen
[@acharya2009computer]. Otras técnicas, se basan también en la
detección y seguimiento de las líneas centrales de los vasos
sanguíneos [@tolias1998fuzzy] [@englmeier2004early]
[@vlachos2010multi]. También existen técnicas más avanzadas para ello
bassadas en el uso de **filtros adaptados** de dos dimensiones
[@katz1989detection] [@hoover1998locating] [@mookiah2013evolutionary]
[@gang2002detection]. A partir de estos pre-procesamientos existen
sistemas capaces de detectar anchuras anormales en estos vasos
sanguíneos [@hayashi2001development].

La presencia de **hemorragias** en las imágenes de fondo de ojo es mayor
en los estadios más graves de la enfermedad. Su detección se realiza
habitualmente junto con la detección de los vasos sanguíneos.

La presencia de **exudados** es el síntoma más característico de
RD. Para la detección de estos, es común comenzar por la eliminación
de las imágenes de los vasos sanguíneos o el disco óptico. Una vez
eliminados este elementos es posible detectar los exudados mediante
una secuencia de algoritmos de procesamiento de imágen
[@acharya2009computer]. Técnicas más avanzadas, basadas en
clasificadores estadísticos basados en los niveles de brillo y uso de
ventanas espaciales como estrategia de verificación han obtenido
resultados cercanos all 100% de exactitud en la detección de imágenes
con exudados y 70% de exactitud en la detección de imágenes de retinas
sanas [@wang2000effective]. Técnicas de detección de exudados basadas
en redes neuronales [@hunter2000quantification] o en el algoritmo PCA
[@li2000fundus] también han conseguido resultados similares a los
comentados anteriormente. Esta última técnica, también permitía
obtener la localización del disco óptico y la fóvea con unas
exactitudes, respectivamente de 99% y 100%.

A partir de la presencia de **microaneurismas** es también posibe la
detección de la RD, llegando a conseguirse una sesibilidad del 85% y
una especifidad del 90% [@jelinek2006automated]. La forma de
detectarlos es similar a las anteriores y requiere eliminar el disco
óptico y los vasos sanguíneos de la imagen antes de aplicar una serie
de técnicas de procesamiento de imágenes. También es posible el uso de
técnicas basadas en morfología matemática [@walter2007automatic]
[@hatanaka2008improvement]. Algunos de estos métodos realizan
transformaciones del espacio de color de las imágenes a HSV (Hue,
Saturation, Value), espacio donde les es más fácil realizar el
procesamiento. El uso de las transformada wavelet también demostró ser
eficaz en la detección de microaneurismas [@quellec2008optimal].

Otras investigaciones, se basaron en las características morfológicas
de la imagen [@sinthanayothin2003automated].

De igual forma que muchas de las técnicas explicadas hasta ahora basan
su predicción en la detección de alguna de las lesiones típicas de la
retinopatía diabética, también existen sistemas más completos que son
capaces de detectar, de forma simultánea, los tres tipos de lesiones y
realizar predicciones en base a la presencia de cada tipo de lesión
mediante clasificadores como árboles de decisión o redes neuronales
[@reza2011decision] [@ege2000screening] [@sinthanayothin2002automated]
[@sinthanayothin2003automated]. Técnicas de aprendizaje no supervisado
como el FCM también han demostrado ser eficaces en esta tarea
[@osareh2002classification]

Hasta ahora, todas las investigaciones explicadas trataban de predecir
una variable binaria, la presencia o no de retinopatía diabética. Sin
embargo, otras investigaciones han tratado de crear métodos capaces de
predecir también el tipo de RD (Proliferativa o No
Proliferativa). Estas técnicas han conseguido sensibilidad y
especifidad de más del 95% [@mookiah2013evolutionary] mediante el uso
de redes neuronales. Otros trabajos han intentado distinguir, con
éxito, hasta 5 grados de RD [@acharya2012integrated]
[@acharya2008application] [@acharya2009computer]

## Detección de DMAE mediante Machine Learning

## Detección de RD mediante Deep Learning

## Detección de DMAE mediante Deep Learning
