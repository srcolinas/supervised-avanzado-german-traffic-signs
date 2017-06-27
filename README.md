# RED NEURONAL CONVOLUCIONAL DE PROFUNDIDAD MEDIA
Para esta solución se utilizó una red neuronal convolucional y **todos** los datos de entrenamiento 
disponibles

## Modelo
La red cuenta con las siguientes capas:
1. Convolución (40 filtros, kernel de 7x7)
2. Pooling (2x2, stride 1)
3. Convolución (20 filtros, kernel de 5x5)
4. Convolución (10 filtros, kernel de 3x3)
5. Convolución (5 filtros, kernel de 3x3)
6. Pooling (2x2, stride 1)
7. Capa normal (fully connected) con 768 unidades y dropout
8. Capa normal (fully connected) con 43 unidades (número de clases)
    
El dropout se hizo con una probabilidad de conservación de 0.3

## Resultado

 95% < Tasa de aciertos < 96%

## Descripción de los archivos

| Archivo | Descripción|
| - | - |
| model.py | Contiene el modelo, la definición del gráfico de tensorflow y su entrenamiento |
| explore.ipynb | Exploración inicial de los datos |
| restore_and_predict.ipynb | Permite explorar más a fondo el resultado del entrenamiento | 

Los archivos asumen que se tiene una carpeta llamada 'german-traffic-signs' la cual contiene 
las carpatas 'training-set' y 'test-set', las cuales corresponden a los datos de entrenamiento 
y prueba respectivamente (como son entregados por el paquete [dataget](https://github.com/cgarciae/dataget))

model.py crea carpetas donde guarda algunos datos del modelo y el enternamiento, pero estos no 
serán guardados en el repositorio.

### Más detalles del reto 
Para más detalles visite la página principal del reto en colomb-ia:
* [reto](https://github.com/colomb-ia/supervised-avanzado-german-traffic-signs)
* [ranking](https://github.com/colomb-ia/supervised-avanzado-german-traffic-signs/blob/master/ranking.md)

    


