# PPE Safety Monitor

El monitor comprueba en el navegador si una persona lleva casco y mascarilla. La cámara permanece en el dispositivo del usuario: el navegador captura imágenes y el backend las envía a los workflows de Roboflow existentes.

## Ejecutar localmente

1. Copia `.env.example` a `.env` y completa la clave de Roboflow, el workspace y los dos IDs de workflow.
2. Instala las dependencias:

   ```powershell
   pip install -r requirements.txt
   ```

3. Inicia la interfaz web desde la carpeta del proyecto:

   ```powershell
   python web.py
   ```

4. Abre `http://127.0.0.1:5000` y selecciona **START CAMERA**. El navegador solicitará permiso para utilizar la cámara.

La ejecución de escritorio que ya existía se conserva y puede iniciarse por separado con `python main.py`.

## Desplegar en Render

1. Sube el proyecto a un repositorio GitHub, comprobando que `.env` no esté incluido.
2. En Render selecciona **New + → Blueprint** y conecta ese repositorio. Render detectará `render.yaml`.
3. Define los valores solicitados para `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `PERSON_WORKFLOW_ID`, `PPE_WORKFLOW_ID`, `APP_USERNAME` y `APP_PASSWORD`. Usa una clave de Roboflow vigente y credenciales de acceso únicas.
4. Abre la URL HTTPS de Render, inicia sesión con las credenciales configuradas y permite el uso de cámara.

Render no puede abrir una cámara conectada a tu computadora. La cámara se captura en la página HTTPS y los fotogramas se envían al servicio. El backend mantiene un único trabajador de Gunicorn porque el estado del check-in y el trabajador de inferencia son compartidos en memoria; no aumentes el número de trabajadores ni de instancias sin implementar estado compartido.

La ruta `/healthz` comprueba que el servidor web arrancó; no valida la conexión con Roboflow. El estado de inferencia y los errores de Roboflow se muestran en la interfaz y en los logs del servicio.

Para mantener desactivadas las llamadas de RapidAPI, `ROBOFLOW_ONLY` está establecido en `true` en `render.yaml`. Las dependencias de Render usan OpenCV headless; las dependencias locales conservan OpenCV normal para que siga funcionando la ventana de escritorio.
