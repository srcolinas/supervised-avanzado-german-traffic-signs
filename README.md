# VARIAS CNN CON POCA PROFUNDIAD
Para esta solución se utilizó **ocho** CNNs y **todos** los datos de entrenamiento 
disponibles. Cada red se entrenó con datos de diferentes y el criterio de parada (early stopping) 
se hizo con datos diferentes para cada modelo.

## Modelo
Cada red es de la siguiente forma :
1. Convolución (20 filtros, kernel de 5x5)
2. Pooling (2x2, stride 2)
3. Capa normal (fully connected) con 1024 unidades y droput
4. Capa normal (fully connected) con 43 unidades (número de clases)
  
    
## Resultado

89% < Tasa de aciertos < 90%

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

    


