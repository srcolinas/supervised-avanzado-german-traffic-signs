# Principal Components Analysis + Suport Vector Machines
Para esta solución se utilizaron **500** vectores de características (features) usando PCA y se hizo una 
clasificación lineal utilizando SVM. Se usaron **todos** los datos

## Modelo
1. PCA(n = 500)
2. LinearSVC

## Resultado

82% < Tasa de aciertos < 83%

## Descripción de los archivos

| Archivo | Descripción|
| - | - |
| model.py | Contiene el modelo, la definición del gráfico de tensorflow y su entrenamiento |
| explore.ipynb | Exploración inicial de los datos |

Los archivos asumen que se tiene una carpeta llamada 'german-traffic-signs' la cual contiene 
las carpatas 'training-set' y 'test-set', las cuales corresponden a los datos de entrenamiento 
y prueba respectivamente (como son entregados por el paquete [dataget](https://github.com/cgarciae/dataget))

model.py crea carpetas donde guarda algunos datos del modelo y el enternamiento, pero estos no 
serán guardados en el repositorio.

### Más detalles del reto 
Para más detalles visite la página principal del reto en colomb-ia:
* [reto](https://github.com/colomb-ia/supervised-avanzado-german-traffic-signs)
* [ranking](https://github.com/colomb-ia/supervised-avanzado-german-traffic-signs/blob/master/ranking.md)

    


