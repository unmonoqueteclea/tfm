# Conclusiones {#conclusiones}

En plena era de los datos y la automatización, la medicina no puede
quedar atrás. Las enfermedades analizadas son solo 2 ejemplos de cómo
el Machine Learning puede ayudar a los especialistas a detectar
posibles enfermedades en estadios muy tempranos, lo que nos permitirá
tratarlas antes de que puedan afectar a la vida diaria del paciente.

Durante este trabajo hemos podido comprobar cómo las dos enfermedades
que más casos de ceguera producen en todo el mundo podrían ser
detectadas de forma muy temprana, pudiendo así ser tratadas antes de
que avancen. El uso de clasificadores de Machine Learning, entrenados
con datos históricos fiables, nos permite crear sistemas robustos que
aúnen todo el conocimiento de muchos especialistas, y puedan llegar a
todos estos sitios donde no es fácil encontrar este tipo de
especialistas.

Los resultados obtenidos, lejos de los de algunos de los modelos
estudiados durante el análisis del estado del arte, son más un motivo
de optimismo que de decepción. Como ya se ha explicado a lo largo del
trabajo, la mayoría de estos procedían de datasets con una cantidad
muy limitada de imágenes. El sobreajuste, con cantidades tan pequeñas
de imágenes es prácticamente inevitable. Esos modelos, cuando se
utilizaran en *el mundo real* con imágenes procedentes de otras
cámaras con distintos tamaños, profundidades de color, artefactos, etc
tendrán serios problemas para generalizar.



## Trabajo futuro
Lejos de buscar el *número bonito*, o el *gran titular*, durante este
trabajo se ha puesto énfasis en crear un sistema verdaderamente
útil. Para ello, es necesario conseguir un sistema **robusto** e
**interpretable**. La robustez nos la proporcionará haber usado más de
39000 imágenes procedentes de 13 conjuntos distintos de datos.  La
interpretabilidad nos la proporcionarán la interfaz y los mapas de
activación descritos en apartados anteriores. Aunque sea común oír
aquello de *"Tortura los datos y te confesarán lo que quieras oír"*,
en este caso esa frase no describe la forma de trabajar que ha sido
utilizada. Una vez creado un sistema inicial verdaderamente útil,
conseguir esos *números bonitos* requerirá principalmente de tres
cosas: nuevos datos, mayor pre-procesamiento, y mayor capacidad de
computación para entrenar modelos más complejos.
