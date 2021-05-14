# A Real-Time Fall Detection System: A video based solution

Este programa es capaz de detectar caídas a tiempo real haciendo uso de las imágenes de una cámara o varias cámaras.

## Puesta en marcha 🚀
Para ejecutar el programa primero necesitas instalar las siguientes bibliotecas:
- [Tensorflow](https://www.tensorflow.org/)
- [OpenCV-Python](https://pypi.org/project/opencv-python/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pypi.org/project/pandas/)
- [Sklearn](https://pypi.org/project/scikit-learn/)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [Telegram-send](https://pypi.org/project/telegram-send/)
- [argsparse](https://docs.python.org/3/library/argparse.html)

También es necesario que guardes el modelo de [estimación de pose PoseNet](https://www.tensorflow.org/lite/examples/pose_estimation/overview), para descargar, [pulsa aquí](https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite). 📁

### Alertas de Telegram 📣
Para que el programa funcione correctamente, es necesario configurar un bot de Telegram (también puedes desactivar la opción de recibir alertas con -t False).

Para configurar el bot de Telegram: 
1. Con tu cuenta de Telegram, envía el mensaje `/newbot` a [@botfather](https://telegram.me/botfather) y copia el `token`.
1. Con telegram-send ya instalado en tu entorno de trabajo, escribe en la consola `telegram-send --configure` y pulsa enter.
1. Pega el `token` y copia la `contraseña`
1. Envía la `contraseña` a tu bot para ponerlo en marcha.
1. ¡Listo! 🎉  

## Ya tienes lo necesario 💪
Pero si quieres, también puedes entrenar el modelo tú mismo/a, puedes ver los pasos en `/human_state_classifier/train.ipynb`, **¡de esta manera funcionará mucho mejor!**

### Es hora de poner parámetros 🔥🔥
1. **Debes añadir las cámaras 🎥 en el archivo** `cameras.conf`, puede se una ruta a un vídeo o una ruta a un vídeo en vivo:
   > ruta/enlace, habitación

1. ¿Te acuerdas del modelo que dije que guardaras? 📁, es hora de usarlo, **indica la `ruta` donde tienes el modelo de estimación de pose PoseNet**:
    > py fall_detector -p "`ruta`"
   
3. También puedes añadir otros parámetros, como, por ejemplo, para ver las cámaras y la situación que han detectado.
    > py fall_detector -p "`ruta`" -d True
   
4. Para ver el resto de parámetros usa `-h`
    > py fall_detector -h
   

### ¡Recuerda que las cámaras  no tienen por qué ser buenas para hacer funcionar esta maravilla!
<p align="center">
⭐
</p>