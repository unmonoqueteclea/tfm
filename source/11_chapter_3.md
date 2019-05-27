# Deep learning y aplicaciones médicas

## IA, Machine Learning y Deep Learning

## Redes neuronales

### Arquitecturas de redes neuronales

## Medidas de evaluación de sistemas de Deep Learning
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

## Deep Learning para aplicaciones médicas

<!---------------------------------------->

## Introduction

This is the introduction. Nam mollis congue tortor, sit amet convallis tortor mollis eget. Fusce viverra ut magna eu sagittis. Vestibulum at ultrices sapien, at elementum urna. Nam a blandit leo, non lobortis quam. Aliquam feugiat turpis vitae tincidunt ultricies. Mauris ullamcorper pellentesque nisl, vel molestie lorem viverra at.

## Method

Suspendisse iaculis in lacus ut dignissim. Cras dignissim dictum eleifend. Suspendisse potenti. Suspendisse et nisi suscipit, vestibulum est at, maximus sapien. Sed ut diam tortor.

### Subsection 1 with example code block

This is the first part of the methodology. Cras porta dui a dolor tincidunt placerat. Cras scelerisque sem et malesuada vestibulum. Vivamus faucibus ligula ac sodales consectetur. Aliquam vel tristique nisl. Aliquam erat volutpat. Pellentesque iaculis enim sit amet posuere facilisis. Integer egestas quam sit amet nunc maximus, id bibendum ex blandit.

For syntax highlighting in code blocks, add three "`" characters before and after a code block:

```python
mood = 'happy'
if mood == 'happy':
    print("I am a happy robot")
```

Alternatively, you can also use LaTeX to create a code block as shown in the Java example below:
\lstinputlisting[style=javaCodeStyle, caption=Main.java]{source/code/HelloWorld.java}

If you use `javaCodeStyle` as defined in the `preamble.tex`, it is best to keep the maximum line length in the source code at 80 characters.

### Subsection 2

This is the second part of the methodology. Proin tincidunt odio non sem mollis tristique. Fusce pharetra accumsan volutpat. In nec mauris vel orci rutrum dapibus nec ac nibh. Praesent malesuada sagittis nulla, eget commodo mauris ultricies eget. Suspendisse iaculis finibus ligula.

<!--
Comments can be added like this.
-->

## Results

These are the results. Ut accumsan tempus aliquam. Sed massa ex, egestas non libero id, imperdiet scelerisque augue. Duis rutrum ultrices arcu et ultricies. Proin vel elit eu magna mattis vehicula. Sed ex erat, fringilla vel feugiat ut, fringilla non diam.

## Discussion

This is the discussion. Duis ultrices tempor sem vitae convallis. Pellentesque lobortis risus ac nisi varius bibendum. Phasellus volutpat aliquam varius. Mauris vitae neque quis libero volutpat finibus. Nunc diam metus, imperdiet vitae leo sed, varius posuere orci.

## Conclusion

This is the conclusion to the chapter. Praesent bibendum urna orci, a venenatis tellus venenatis at. Etiam ornare, est sed lacinia elementum, lectus diam tempor leo, sit amet elementum ex elit id ex. Ut ac viverra turpis. Quisque in nisl auctor, ornare dui ac, consequat tellus.
