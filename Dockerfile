# Usa una imagen base de TensorFlow
FROM tensorflow/tensorflow:latest

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala Git para clonar el repositorio
RUN apt-get update && apt-get install -y git

# Clona el repositorio desde GitHub
RUN git clone https://github.com/pelaokano/docker_tensorflow.git .

# Comando para ejecutar la aplicación
CMD ["python", "prueba_tensorflow.py"]