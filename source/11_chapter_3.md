# Machine Learning y aplicaciones médicas {#ml}
<!-- Comienzo: Lunes, 27 Mayo
     Previsión: Lunes, 3 de Junio
     Papers útiles: (P2) (P0) (P25-P36)
     Fin versión 1: Sabado, 9 de Junio
-->

Tradicionalmente, el trabajo de los ingenieros de software ha
consistido en dar a las computadoras una serie de reglas explícitas de
cómo tiene que procesar la información y cómo tiene que tomar
decisiones. Sin embargo, la complejidad del campo de la medicina es
tal que sería prácticamente imposible capturar toda la información
relevante mediante una serie de reglas definidas de forma explícita
[@schwartz1986artificial].

El Machine Learning ha permitido crear sistemas que aprendan de los
ejemplos sin necesidad de que se programen reglas específicas, lo que
ha supuesto una auténtica revolución en prácticamente cualquier sector
profesional imaginable entre los que, por supuesto, se encuentra
también la medicina. Estos sistemas buscan, de forma automática, la
mejor forma de predecir una variable en función de una serie de
variables de entrada del sistema. De esta forma se crea un sistema
que, idealmente, será capaz de obtener la salida correcta para
variables de entrada nunca vistas. Esto se conoce como **aprendizaje
supervisado** aunque es importante mencionar, que no es la única forma
de Machine Learning o Aprendizaje Automático.

El principal inconveniente del Machine Learning con respecto al
aprendizaje humano es, a la vez, su principal ventaja: la necesidad de
grandes cantidades de datos para su correcto funcionamiento.. Si se
alimentan con una cantidad suficiente de datos, los algoritmos de
Machine Learning podrán encontrar patrones que para los humanos sería
prácticamente imposible. Un modelo de Machine Learning podrá analizar
en segundos más pacientes de los que verá un médico en toda su
vida. Además, la cantidad de predictores distintos que manejará sería
totalmente inviable para un humano.


## IA, Big Data,  Machine Learning y Deep Learning
Inteligencia Artificial, Big Data, Machine Learning, Deep Learning;
actualmente existe mucha confusión en el uso de estos términos. Aunque
comparten características, no tienen el mismo significado. En este
capítulo se detallarán las similitudes y diferencias entre todos
ellos para evitar el lenguaje inexacto usado habitualmente en este
campo.

Comenzaremos por el Big Data, pues es el término más vago y
confuso. Cuando hablamos de Big Data nos referimos al análisis de
grandes cantidades de datos que no podrían ser analizados con técnicas
convencionales. Sin embargo, las líneas que marcan las fronteras del
Big Data están difusas, y a menudo es un término utilizado más por
medios de comunicación y gurús que por profesionales técnicos y
académicos.

Por otro lado, los campos de la Inteligencia Artificial, el Machine
Learning y el Deep Learning, sí que están más claramente definidos
aunque, el hecho de que cada uno de ellos sea un subcampo del
anterior, a menudo da lugar a confusión. Llamamos Inteligencia
Artificial a un conjunto de técnicas que tratan de que los ordenadores
imiten, de alguna forma, el comportamiento humano.

El Machine Learning es un subcampo dentro de la IA, que consiste en un
conjunto de técnicas y herramientas que permiten a los ordenadores
obtener patrones de grandes conjuntos de datos. Gracias a esos
patrones seremos capaces de entender mejor los datos o incluso hacer
predicciones. La forma más común de machine learning es el conocido
como **Aprendizaje Supervisado**. Durante el entrenamiento de los
modelos de Aprendizaje Supervisado, se proporcionan al algoritmo una
serie de datos históricos. Entre ellos se encuentra la **variable
objetivo**, es decir, la que posteriormente querremos predecir en los
nuevos datos. Por ejemplo, en un modelo de detección de cáncer a
partir de imágenes médicas, nuestra variable objetivo será
precisamente la que indique si una imágen pertenece a un paciente
enfermo o un paciente sano. Esta variable, por lo tanto, tendrá dos
posibles valores siendo este un problema de **clasificación**. En los
problemas de clasificación se tratan de predecir variables
discretas. Si, por ejemplo, realizáramos un modelo para predecir el
precio de una vivienda en función de sus características, nos
encontraríamos ante un problema de regresión, pues la probabilidad es
un valor contínuo.

Es común en los algoritmos para aprendizaje supervisado el uso de una
función de pérdidas. Esta función mide el error entre las predicciones
del modelo y los datos reales. De forma iterativa, los algoritmos
tratarán de ajustar una serie de parámetros (o pesos) intentando
minimizar esta función.

Precisamente estos últimos algoritmos, las redes neuronales, son las
que dan lugar al Deep Learning. Cuando añadimos más capas intermedias
a las redes neuronales somos capaces de detectar patrones mucho menos
evidentes además de tratar problemas complejos sin necesidad de un
pre-procesamiento manual previo que los simplifique. Los algoritmos de
Deep Learning son actualmente el estado del arte en tareas como
reconocimeiento de imágenes [@krizhevsky2012imagenet], reconocimiento
del habla [@deng2013new], procesamiento del lenguaje natural
[@collobert2011natura], análisis de información de aceleradores de
partículas [@baldi2014searching] o reconstrucción de los circuitos
cerebrales [@helmstaedter2013connectomic], entre muchas otras.



## Redes neuronales, descenso de gradiente y backpropagation
Una red neuronal consiste en un conjunto de nodos, conocidis como
**neuronas**, conectados entre si para transmitirse señales.

Estas neuronas están dispuestas en una serie de capas, en las que cada
neurona de una capa está conectado a todas las neuronas de las capas
anteriores. Cada neurona combina sus entradas con un conjunto de
coeficientes o pesos. El nombre de **peso** se debe a que la función
de estos, al multiplicarse por los valores de las entradas es definir
la importancia de cada una. Los resultados de todos estos productos se
suman y se pasa el resultado a lo que se conoce como **función de
activación**, que añade un comportamiento no-lineal al
proceso. Actualmente, la función de activación más conocida es la
**ReLU** (Rectified Linear Unit) cuya fórmula es simplemente
$f(z)=max(z,0)$.

Durante el entrenamiento, los pesos cambian de valor, intentand
minimizar la función de pérdidas explicada en el apartado
anterior. Para ajustar el vector de pesos se suele calcular el vector
de gradiente que indica, para cada peso cómo se modificaría el error
si ese peso se aumentara ligeramente. El vector de pesos es entonces
ajustado en el sentido opuesto al vector de gradiente. Esto es lo que
conocemos como **descenso de gradiente**. En la práctica, este proceso
no usa todos los datos cada vez sino que se utiliza el **Descenso de
Gradiente Estocástico** (SGD por sus iniciales en inglés). Gracias al
SGD podemos actualizar los pesos de nuestra red neuronal tomando cada
vez un pequeño conjunto de datos (conocido como **batch**).

El origen de las redes neuronales es el **Perceptrón** desarrollado en
los años 40 que eran redes simples de una sola capa de entrada y una
capa de salida. Sin embargo, fue en los años 80 cuando estas
comenzaron a desarrollar su verdadero potencial gracias al algoritmo
de backpropagation, que permitió que se añadieran nuevas capas
intermedias a las redes neuronales conocidas como **capas
ocultas**. La técnica de backpropagation no es más que una aplicación
de la regla de la cadena de las derivadas. Gracias a ella podemos
propagar el error a lo largo de las capas, para calcular en cada una
el vector de gradiente y actualizar con él los pesos.


Estas nuevas capas intermedias añadidas a las redes neuronales
permitían encontrar patrones más complejos, y dieron lugar a lo que
conocemos como **Deep Learning**. Estrictamente hablando, nos
referimos a Deep Learning cuando tenemos una red con más de una capa
oculta. El Deep Learning permite crear modelos computacionales
compuestos de múltiples capas de procesamiento que son capaces de
aprender representaciones de los datos con múltiples capas de
abstracción [@lecun2015dee].


En las redes profundas, cada capa de neuronas se entrena en un
conjunto de características distintos, en base a la salida de la capa
anterior. A medida que avanzamos a través de la red, las
características que las neuronas son capaces de detectar son más
complejas, ya que agregan y recombinan características de capas
anteriores. Esta propiedad, conocidad como jerarquía de
características hace posible que este tipo de redes sean capaces de
tratar datasets de muy alta dimensionalidad. Las redes neuronales
profundas realizan una **extracción automática de características**,
sin la necesidad de la intervención de un humano.

### Redes neuronales convolucionales
La capacidad de las redes neuronales de encontrar patrones complejos
en datasets con una gran cantidad de dimensiones las convierten en
candidatas perfectas para tareas como la clasificación de imágenes o
el reconocimiento de voz. Sin embargo, y como hemos visto a lo largo
de este capítulo, estos clasificadores necesitan un trabajo manual
previo de extracción de características.

La aparición de las conocidas como **Redes Convolucionales (CNN)**
permitió eliminar este paso y delegarlo en el propio algoritmo de
backpropagation. De esta forma, es posible usar como entradas de
nuestro modelo los *datos en bruto* (píxeles de las imágenes, muestras
de las pistas de audio, etc). Un momento clave para las redes
convolucionales fue en 2012, en el ImageNet Large Scale Visual
Recognition Challenge (ILSVRC) cuando una solución novedosa basada en
CNNs [@krizhevsky2012imagenet] obtuvo, de forma holgada, la primera
posición.

Una de las principales ventajas de las redes neuronales
convolucionales con respecto a otras aproximaciones al problema es que
poseen un cierto grado de invarianza a la distorsión y al
desplazamiento. Esto permite que podamos usar este tipo de redes sin
apenas pre-procesamiento.

La arquitectura de las redes convolucionales está basada en la
organización de la corteza visual del cerebro humano. En él existen
neuronas individuales que responden a estímulos en una región
delimitada del campo visual.

Las redes convolucionales constan de capas convolucionales y de
reducción (o pooling) alternadas.

- En las capas de convolución de aplican una serie de filtros (cuyos
  pesos son parámetros modificados durante el entrenamiento por el
  backpropagation). En ellas se producen también las transformaciones
  no lineales (ReLU). Cada uno de los filtros se desplazará sobre toda
  la imagen calculándose en cada posición el producto vectorial entre
  la región de la imagen y los valores del filtro. Estos filtros hacen
  de **detectores de características**.
- En las capas de reducción o pooling se disminuye la cantidad de
  parámetros. Para ello se obtiene el promedio o el máximo de una
  serie de regiones.

Sobre todas estas capas tenemos las **Fully Connected Layer**, redes
tradicionales que a partir de los parámetros extraídos por las capas
convolucionales y de pooling, realizan las clasificaciones o
regresiones finales.

El funcionamiento del algoritmo de backpropagation en las redes
convolucionales es tan simple como en las convencionales, por lo que
no supone ninguna dificultad añadida para el entrenamiento.

Las redes convolucionales explotan la propiedad de que sus principales
características no son más que composiciones de otras características
más simples. En una imágen, por ejemplo, mediante la composición de
varias líneas simples, damos lugar a motivos que, de nuevo mediante
composición dará lugar a las formas de los objetos. La detección de
cada uno de estos niveles de abstracción corresponderá a unas capas
concretas de nuestra red convolucional.

### Arquitecturas de redes neuronales

## Evaluación de sistemas de Deep Learning
<!--
    Comienzo: 189 Mayo
    Fin V1:
    Papers útiles: (P2 pag 112)
    Webs:
-->

Un paso tan importante como el modelado, en un proyecto de análisis de
datos, es la evaluación de los resultados. Es de vital importancia
establecer medidas que nos permitan saber cómo se está comportando
nuestro modelo.

En la literatura, existen una extensa cantidad de medidas, aunque en
este caso nos centraremos en algunas de las más comunes en problemas
de este tipo.


El problema que se está analizando en este trabajo es un **problema
binario**, es decir se trata de predecir una clase con sólo dos
posibles valores, verdadero o falso. Más concretamente, únicamente
trataremos de detectar si la persona tiene o no la enfermedad. Cuando,
en un problema de este tipo, comparamos la predicción realizada por un
modelo, con el *ground truth* (es decir, la clase que realmente
correspondería a esa instancia), pueden darse 4 posibles casos:

- **Verdadero Positivo (o True Positive, TP)**: El sistema predice que
  el paciente **SÍ** tiene la enfermedad y acierta.
- **Verdadero Negativo (o True Negative, TN)**: El sistema predice que
  el paciente **NO** tiene la enfermedad y acierta.
- **Falso Negativo (o False Negative, FN)**: EL sistema predice,
  erróneamente, que el paciente **NO** tiene la enfermedad cuando en
  realidad sí que la tiene.
- **Falso Positivo (o False Positive, FP)**: El sistema predice,
  erróneamente, que el paciente **SÍ** tiene la enfermedad cuando en
  realidad no la tiene.


A partir de la cantidad de predicciones de cada uno de estos posibles
4 tipos se pueden definir una serie de medidas muy comunes en
problemas de este tipo.

La **sensitividad**...

## Transfer Learning
La mayoría de métodos de Machine Learning asumen que los datos de
entrenamiento y los de test vienen de la misma distribución y espacio
funcional. [@pan2009survey]. Por ello, cuando esta distribución
cambia, debemos volver a entrenear nuestros modelos desde 0,
obteniendo datos totalmente nuevos. El Transfer Learning, sin
embargo, nos permite tener distribuciones distintas en entrenamiento y
test, mediante la transferencia de conocimiento entre modelos.

El Transfer Learning es una técnica de Machine Learning que permite
utilizar un modelo desarollado para una tarea específica como punto de
partida para otra tarea distinta (aunque relacionada). Además de
permitirnos obtener clasificadores de forma mucho más rápida
aprovechando el conocimiento previo, el Transfer Learning hace posible
el uso del Deep Learning con conjuntos de datos pequeños con los que
sería imposible entrenar una red desde 0. El Transfer Learning es
considerado por muchos investigadores como un paso más en dirección
hacia la AGI ^[Artificial General Intelligence: Aquella inteligencia
artificial que puede realizar con éxito cualquier tarea intelectual de
cualquier ser humano].

Aunque el Deep Learning es usado en diversidad de tareas como
Procesamiento del Lenguaje Natural o tratamiento de audio, en este
capítulo se analizará su uso en Visión por Computador, que es el caso
adecuado a nuestro problema.

### Transfer Learning con imágenes
En la práctica, muy poca gente entrena redes convolucionales desde 0.
Existen 2 principales motivos:

- En determinados ámbitos, no siempre existen datasets con una gran
  cantidad de imágenes, suficiente para entrenar una red desde 0.
- Aún existiendo dicho dataset, el tiempo necesario para su completo
  entrenamiento puede ser de días, semanas o incluso meses dependiendo
  del equipo usado.

Existen tres principales estrategias a la hora de realizar Transfer
Learning:

- **Red convolucional como extractor de características**: Como se ha
  analizado anteriormente una red convolucional puede ser vista como
  una herramienta para extraer características de las imágenes, que
  posteriormente serán usadas por capas totalmente conectadas (o por
  cualquier otro tipo de clasificador) para realizar la
  clasificación. Conociendo esto, podemos utilizar la red
  convolucional entrenada para un conjunto de imágenes, en otro
  conjunto de imágenes distinto, siendo el clasificador final el único
  que tendrá que reentrenarse.
- **Fine-tunning de la red convolucional** Como se ha analizado
  anteriormente, las capas iniciales de las redes convolucionales se
  encargan de detectar características más generales y patrones
  simples, que van siendo más complicados a medida que avanzamos hacia
  capas posteriores. Por lo tanto, es común que estas primeras capas
  tengan siempre contenidos similares incluso en modelos entrenados
  con diferentes conjuntos de imágenes. Por lo tanto, podrán ser
  reaprovechadas, con lo que únicamente tendremos que reentrenar las
  últimas capas y el clasificador final.
- **Modelos pre-entrenados**: Este tercer caso supone el
  reentrenamiento total de la red, sin embargo, partiendo de unos
  pesos que han sido previamente entrenados en otro conjunto de
  imágenes. De esta forma conseguimos que el número de iteraciones
  necesarias hasta llegar al nivel de exactitud requerido sea menor.

Los criterios para decidir qué estrategia de Transfer Learning usar en
cada caso dependen principalmente de las diferencias de contenido y
tamaño entre las imágenes de nuestro dataset y las del dataset
original (con el que se entrenó el modelo que vamos a reutilizar)

Es común usar las siguientes reglas como guía en función de 4 posibles
escenarios ^[http://cs231n.github.io/transfer-learning/]:

- **El nuevo dataset es pequeño pero similar al original**: Al ser un
  dataset muy pequeño, modificar las capas convolucionales de nuestro
  modelo original puede dar lugar a overfitting. Por lo tanto, y
  puesto que las imágenes de ambos datasets son similares, la
  estrategia adecuada será utilizar la red convolucional como
  extractor de características y entrenar únicamente el clasificador
  final.
- **El nuevo dataset es grande y similar al original**: En este caso,
  como tenemos más imágenes podremos realizar fine-tunning de la red
  sin miedo a caer en overfitting.
- **El nuevo dataset es pequeño y muy diferente al original**: De
  nuevo, al tener un dataset pequeño, descartaremos entrenar la red
  convolucional. En este caso, lo que haremos es entrenar solo un
  clasificador. Y además, al ser las imágenes distintas a las del
  dataset original, no podremos aprovechar las últimas capas de la red
  convolucional que serán eliminadas.
- **El nuevo dataset es grande y muy diferente al original**: En este
  caso entrenaremos la red convolucional al completo. Sin embargo,
  será de gran utilidad comenzar nuestro entrenamiento a partir de un
  modelo pre-entrenado.


## Aplicaciones médicas del Machine Learning
¿Cómo sería un sistema sanitario en el que cada decisión relacionada
con una enfermedad, en lugar de ser tomada por una sola persona, fuera
tomada por un conjunto de los principales expertos del mundo de esa
enfermedad? Esa es la pregunta que se hacen multitud de investigadores
[@rajkomar2019machine]. Estos concluyen que las tratamientos recetados
serán los más efectivos, y no, los más conocidos por la persona que
los prescribe. Además, se evitaría el error humano. Por desgracia, un
sistema de este tipo sería inviable, debido principalmente a la falta
de expertos, que no darían abasto para diagnosticar a millones de
pacientes cada día. Sin embargo, el Machine Learning nos promete un
sistema similar a este, pero realmente viable y escalable. Con la
capacidad de aplicar todas las lecciones recogidas de la experiencia
colectiva en cada una de las decisiones, sin que esto genere una gran
carga de trabajo para unos pocos expertos.

Hace ya 50 años se ponía de manifiesto la necesidad de "aumentar, o
incluso remplazar las funciones intelectuales de los médicos"
[@schwartz1970medicine]. Además, la implementación de los Historiales
Clínicos Electrónicos en diversidad de sistemas de salud, proporciona
una ingente cantidad de datos que podrían ser de gran utilidad para la
creación de modelos de Machine Learning de todo tipo. ^[Aunque no hay
que olvidar las limitaciones derivadas de la privacidad y la
protección de datos]

El uso de herramientas estadísticas en medicina no es ninguna
novedad. Desde antes de la irrupción de las técnicas más novedosas de
Machine Learning y Deep Learning, la estadística descriptiva tenía un
papel fundamental, estando prácticamente siempre presente en los
artículos de las revistas de medicina. Son necesarias técnicas
estadísticas que nos permitieran estudiar la eficacia de los fármacos
o los factores de riesgo de determinadas enfermedades.

La rama de la **epidemilogía**, cuyos orígenes se sitúan hacia el
siglo IV a.C., trata de recopilar y tratar los datos de los pacientes
y sus patologías, para estudiar la frecuencia y distribución de los
diversos fenómenos relacionados con la salud. La epidemilogía trata de
encontrar patrones en las enfermedades centrándose principalmente en
tres aspectos: tiempo, lugar y persona. Gracias a ella somos capaces
de definir los problemas de salud más importantes de una comunidad,
además de sus factores de riesgo. Con esta información podremos
desarrollar programas de prevención o control, e incluso predecir
tendencias de una enfermedad.

Sin embargo, la revolución del Machine Learning y el Deep Learning de
los últimos años se empieza a hacer notar, aunque de forma más lenta
que en otros campos, en la medicina. Por primera vez, este tipo de
técnicas salen del ámbito de la investigación y son utilizadas para el
**diagnóstico**. Tradicionalmente los programas utilizados en
diagnóstico eran **sistemas expertos**. Este tipo de programas
simplemente se limitaban a pedir una serie de datos sobre el paciente,
y obtenían conclusiones a partir de una serie de reglas que
previamente habían tenido que ser definidas por especialistas. Sin
embargo, con sistemas basados en Machine Learning, **estas reglas son
inferidas a partir de datos históricos**. Una de las principales
características del Machine Learning que le hace destacar sobre otros
métodos tradicionales, es su capacidad de manejar enormes cantidades
de predictores y encontrar complicados patrones entre ellos.

Además, debido a la gran cantidad de información no estructurada
existente (imágenes, señales, textos, etc) en medicina, como era de
esperar, el **Deep Learning** juego un papel esencial, permitiendo que
los datos "hablen por sí mismos". Sin embargo, en todo momento tenemos
que tener presente que nuestras evaluaciones pueden ser demasiado
optimistas o que el sobreajuste puede hacer que nuestros modelos dejen
de funcionar al ponerlos en producción. Tener una Inteligencia
Artificial explicable de la que no solo obtengamos predicciones, sino
el por qué de las mismas, es algo que hará más fácil la entrada de
estos algoritmos en el día a día de los médicos.

### Pronóstico
El Machine Learning nos puede ayudar, mediante la búsqueda de
patrones, en la predicción de la evolución de un enfermo. En varios
sistemas de salud existen ya implantados sistemas que, mediante
Machine Learning son capaces de identificar a los pacientes que están
en riesgo de tener que ser transferidos a las unidades de cuidados
intensivos. [@escobar2016piloting]. Además, diversos estudios sugieren
que se pueden crear eficaces modelos de pronóstico médico a partir de
la información en bruto de historiales [@rajkomar2018scalabl] e
imágenes médicas [@de2018clinically].

La estandarización de los historiales médicos electrónicos sería de
gran ayuda para la implantación de estos sistemas permitiendo, además,
la agregación de datos. Formatos como el **Fast Healthcare
Interoperability Resources (FHIR)** [@mandel2016smart] en los últimos
años con este propósito.

### Diagnóstico
Segeun concluye la Academia Nacional de Ciencias de EEUU,
prácticamente todos los pacientes serán diagnosticados de forma
errónea al menos una vez en su vida [@ball2015improving].  Diversos
estudios han encontrado problema sistemáticos en los servicios de
salud de todo el mundo. Hay evidencias de que, en los sistemas en los
que los servicios de diagnóstico y tratamiento los realiza una misma
organización, obteniendo mayores ingresos la compañía mediante la
prescripción de drogas y la solicitud de nuevas pruebas médicas, la
tendencia a hacerlo aumenta. [@currie2014addressing].

Los datos históricos pueden ser de gran ayuda para la identificación
de posibles patologías durante las visitas clínicas. Los modelos
podrían sugerir nuevas pruebas a los médicos en base a los datos
recogidos en tiempo real [@slack1966computer].

### Tratamiento
La aproximación más directa al problema del tratamiento mediante
Machine Learning sería la creación de modelos entrenados con datos
históricos que aprendieran los medicamentos recetados por los médicos
en cada situación. Sin embargo, esta aproximación tiene un claro
problema, el modelo aprendería los hábitos de prescripción de los
médicos, que no tienen por qué ser los ideales. Por lo tanto, en este
campo aún más, es de vital importancia generar datasets fiables y
analizados en profundiad por expertos para entrenar los modelos
[@rajkomar2019machine].



### Retos clave
Uno de los principales retos en la creación de modelos de Machine
Learning para medicina es la **falta de datos de calidad**. Este tipo de
modelos, sobretodo los de Deep Learning, funcionan mejor cuanto mayor
es la cantidad de datos de los que disponen. Sin embargo, en el campo
de la medicina no existen tanta disponibilidad de los mismos como sí
que existe en otros ámbitos. Una de las principales causas de esa
escasez ese la inviolable privacidad de los datos, que a menudo impide
la creación de grandes datasets. [@rajkomar2019machine]

Otro de los retos es el **sesgo** existente en los datos. Toda actividad
humana está influenciada por un sesgo, ya sea consciente o
inconsciente. La máxima **Entrada basura / Salida basura** de la
analítica de datos está presente también en este campo. De nada
servirá contar con potentes modelos capaces de aprender complicados
patrones, si luego esos patrones los encontrará en datos erróneos o
sesgados.

La **interpretabilidad** de los modelos es también clave. Los médicos
deben conocer el grado de veracidad y las limitaciones de estas
técnicas, para poder incorporarlas como una herramienta más. La
sobre-confianza en estos sistemas puede conllevar una disminución de
la alerta de los médicos que puede tener consecuencias letales. Que
los modelos proporcionen, junto con sus predicciones, un grado de
confiabilidad es un buen principio, pero no basta. De hecho, en
ocasiones estis intervalos de fiabilidad pueden ser interpretados de
forma incorrecta [@jiang2018trust]. Es necesario crear modelos que
sean capaces de explicar el por qué de sus predicciones. De hecho,
esto era uno de los requisitos que indicó la Unión Europea en su
**Guia ética para una Inteligencia Artificial fiable**
[^https://ec.europa.eu/digital-single-market/en/news/ethics-guidelines-trustworthy-ai]. Esto
pude suponer un problema en técnicas de Deep Learning, que siempre han
sido tachadas de ser **cajas negras**. Sin embargo, en los últimos
años se han realizado diferentes estudios que demuestran que los
modelos de Deep Learning pueden ser interpretables con las
herramientas adecuadas. [@cruz2013deep] [@zhang2018visual]
[@lipton2016mythos].

## Correlación no implica causalidad
Aunque sea un mantra repetido hasta la saciedad en la literatura, esta
advertencia merece una apartado propio en un trabajo de estas
características, pues es algo a tener en cuenta y que implica tener
mucha cautela al obtener conclusiones mediante este tipo de métodos.

En muchas ocasiones creemos, de forma errónea, que existe una relación
de causa y efecto entre dos variables que están correlacionadas.


La correlación entre dos variables, puede ser debida a una tercera
**variable oculta** que no tenemos por qué conocer o simplemente puede
ser lo que conocemos como **correlación espúrea**, es decir, mera
casualidad (que no causalidad).

Sin embargo, la falacia **Cum hoc ergo propter hoc** (en latín, "Con
esto, por tanto a causa de esto") sigue siendo estando muy presente en
los medios de comunicación y en las **pseudociencias**.

Si alguna vez el lector divisa a un sujeto disfrazado de pirata, no lo
tome por loco. Ese sujeto podría ser un seguidor de **Bobby Henderson**,
creador de la iglesia pastafari que, cansado de argumentos de los
creacionistas basados en esta falacia, realizó un estudio en el que se
podía apreciar una clara correlación entre la temperatura global y el
descenso del número de piratas. Es común, desde entonces, que los
seguidores de Henderson se disfracen de piratas para recordarlo.


<!-- TODO: Imagen de la grafica de piratas y temperatura
    https://jdcdn-wabisabiinvestme.netdna-ssl.com/wp-content/uploads/2016/06/1-1-768x483.jpg-->

Otro ejemplo, muy popular, es la singular correlación entre el número
de ahogados en piscinas en Estados Unidos y el número de apariciones
en películas de Nicholas Cage.

<!-- TODO: Imagen y explicación de
http://www.tylervigen.com/spurious-correlations-->

Sin embargo, lejos de quedar en una mera anécdota como las anteriores,
es extremadamente preocupante que existan familias en todo el mundo
que estén decidiendo no vacunar a sus hijos debido a una aparente
correlación en un estudio de 2010 entre el número de casos de autismo
y las vacunaciones.

Por lo tanto, es necesaria una gran cautela antes de obtener
conclusiones de los sistemas de Machine Learning. Además, no estaría
de más, aunque no serán objeto de análisis en este trabajo, tener
presentes el Sesgo del Superviviente y la Paradoja de Simpson).
