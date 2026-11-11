---
name: audit-backend-production-readiness
description: Escáner universal pre-producción de seguridad (OWASP, cabeceras HTTP, sanitización de path traversal, secretos), eficiencia de concurrencia (bloqueo del event loop, fugas de recursos) y observabilidad (healthchecks, logging). Reutilizable en cualquier stack (Python, TypeScript/Node, Go, C#).
---

# Skill: Escáner Pre-Producción, Seguridad y Eficiencia (Backend Production Readiness)

Esta skill define el protocolo de inspección y endurecimiento pre-producción para aplicaciones de backend. Combina las auditorías de seguridad estática de herramientas de grado empresarial (SonarQube, Semgrep, Bandit, OSS Scanners) con las directivas de seguridad HTTP evaluadas por navegadores modernos (Lighthouse Best Practices) y los patrones de rendimiento en concurrencia asíncrona.

Es **100% agnóstica a la tecnología** y aplicable a cualquier stack (Python/FastAPI/Django, TypeScript/Node/Nest/Express, Go/Gin/Echo, C#/ASP.NET Core).

---

## 1. Escaneo Automatizado por Terminal (Análisis Estático y Seguridad)

Antes de validar cualquier backend para despliegue productivo, ejecuta las herramientas CLI nativas del ecosistema detectado sin requerir infraestructura pesada:

### 1.1 Python
```bash
# Análisis estático de vulnerabilidades Bandit + Bugbear + Concurrencia Async + Prints
ruff check . --select S,B,ASYNC,BLE,T20

# Escaneo de dependencias y vulnerabilidades conocidas en paquetes (CVEs)
pip audit || safety check
```

### 1.2 TypeScript / Node.js
```bash
# Linter de seguridad y buenas prácticas
npx eslint . --ext .ts,.js --rule 'no-console: error'

# Auditoría de dependencias en cadena de suministro
npm audit --production --audit-level=moderate
```

### 1.3 Go
```bash
# Escaneo de seguridad estático (gosec)
gosec -quiet ./...

# Detección de vulnerabilidades en dependencias Go
govulncheck ./...
```

### 1.4 Universal (Semgrep CLI)
```bash
# Reglas comunitarias esenciales de seguridad OWASP
semgrep --config=auto --error --severity=ERROR --severity=WARNING
```

---

## 2. Seguridad y Prevención de Vulnerabilidades (OWASP Top 10)

### 2.1 Cero Secretos o Rutas Locales Hardcodeadas
- **Prohibido**: Rutas absolutas vinculadas al entorno del desarrollador (`/home/...`, `/Users/...`, `C:\Users\...`).
- **Prohibido**: Claves API, tokens HMAC, credenciales de base de datos o contraseñas quemadas en el código fuente.
- **Configuración Externalizada**:
  - Todo parámetro sensible debe resolverse mediante variables de entorno (`os.getenv`, `process.env`, `os.Getenv`) con validación al inicio de la aplicación.
  - El host y puerto deben ser configurables mediante variables `HOST` (ej. `0.0.0.0`) y `PORT` (ej. `8000`).

### 2.2 Sanitización Estricta de Entradas y Prevención de Path Traversal
- **Validación de Parámetros de Ruta**:
  - Todo identificador o slug recibido en URLs debe validarse contra una expresión regular segura (ej. `^[a-zA-Z0-9_\-]+$`) para impedir secuencias `../` o inyecciones.
- **Nombres de Archivo Seguros**:
  - Los nombres de archivos recibidos en cargas multiparte deben sanitizarse usando `os.path.basename()` o reemplazos de caracteres no alfanuméricos.
- **Extracción Segura de Archivos Comprimidos (Anti Zip-Slip / Tar-Slip)**:
  - Al descomprimir archivos `.zip`, `.tar` o `.tar.gz`, es obligatorio verificar que la ruta resuelta de cada miembro sea relativa y descanse estrictamente dentro del directorio de destino permitido:
  ```python
  target_path = (dest_dir / member.name).resolve()
  try:
      target_path.relative_to(dest_dir.resolve())
  except ValueError as err:
      raise ValueError(f"Intento de Path Traversal bloqueado: {member.name}") from err
  ```
  - Si el archivo contiene enlaces simbólicos (`islnk()` o `issym()`), se debe verificar igualmente que el objetivo del enlace no apunte fuera del directorio de destino.
- **Directorio Raíz Permitido**:
  - Al servir archivos locales o estáticos mediante endpoints de descarga (`FileResponse`), validar que la ruta canónica `resolve()` esté contenida dentro de una lista blanca de directorios permitidos (`models/`, `data/`, `static/`, `docs/`).

### 2.3 Cabeceras de Seguridad HTTP (Lighthouse & OWASP Compliance)
Toda respuesta servida por el servidor o middleware de API debe inyectar cabeceras defensivas para mitigar ataques de clickjacking, MIME-sniffing y XSS:

| Cabecera HTTP | Valor Requerido | Propósito |
| :--- | :--- | :--- |
| `X-Content-Type-Options` | `nosniff` | Impide que el navegador interprete archivos con tipos MIME erróneos |
| `X-Frame-Options` | `SAMEORIGIN` o `DENY` | Previene ataques de Clickjacking embebidos en iframes |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Protege información sensible en la cabecera Referer |
| `Cross-Origin-Opener-Policy` (COOP) | `same-origin` | Aísla el contexto de navegación contra ataques cross-origin |
| `Content-Security-Policy` (CSP) | Directivas estrictas de fuentes (`'self'`, etc.) | Mitiga ejecución de scripts no autorizados |
| `Strict-Transport-Security` (HSTS) | `max-age=31536000; includeSubDomains` | Obliga a conexiones HTTPS seguras (cuando TLS esté habilitado) |

### 2.4 Política de CORS Restrictiva
- **Prohibición Estricta**: No usar jamás `allow_origins=["*"]` en combinación con `allow_credentials=True`. Esta configuración viola las especificaciones de los navegadores modernos y expone la aplicación a ataques CSRF.
- **Enfoque de Producción**:
  - Leer orígenes permitidos desde una variable de entorno (`CORS_ALLOWED_ORIGINS`).
  - Si no hay orígenes configurados o se usa comodín (`*`), deshabilitar `allow_credentials` (`False`).

---

## 3. Eficiencia en Producción y Concurrencia

### 3.1 Detección y Erradicación de Bloqueo del Event Loop (Asincronía Real)
En frameworks basados en bucles de eventos asíncronos (FastAPI, Express, Node.js, asyncio):
- **Problema Crítico**: Ejecutar operaciones intensivas de CPU (cálculo de tensores en PyTorch, compilación de Grad-CAM, hashing de archivos gigantes) o I/O síncrono bloqueante dentro de una función asíncrona (`async def` / `async () => {}`) congela el hilo principal, degradando la latencia de todos los demás clientes concurrentes.
- **Solución Obligatoria**:
  - Toda operación de cálculo pesado o llamada síncrona en rutas `async` debe delegarse al hilo de trabajo del pool de ejecutores:
    ```python
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, sync_heavy_task, *args)
    ```
  - En FastAPI, las rutas declaradas con `def` ordinario se ejecutan automáticamente en un hilo del threadpool, mientras que las declaradas con `async def` se ejecutan directamente en el event loop.

### 3.2 Prevención de Fugas de Recursos y Gestión de Memoria
- **Gestión de Descriptores y Streams**:
  - Toda lectura de archivos, conexión a sockets, clientes HTTP (`httpx.Client`, `requests.Session`) o handles de base de datos debe cerrarse explícitamente mediante gestores de contexto (`with`, `async with`, `try-finally`, `using`).
- **Protección contra Desbordamientos de Memoria (OOM)**:
  - Todo endpoint que procese flujos de subida (`UploadFile`) debe verificar un límite máximo de tamaño antes o durante la lectura por trozos (`chunks`), rechazando cargas excesivas con código `413 Payload Too Large`.
  - Los endpoints de consulta masiva deben imponer paginación obligatoria (`page`, `page_size`) con un límite máximo (ej. `page_size <= 100`).

---

## 4. Observabilidad y Healthchecks

1. **Erradicación de `print()` y `console.log()` en Producción**:
   - Prohibido el uso de sentencias de salida directa por consola para depuración.
   - Usar el sistema de logging estructurado del framework con niveles adecuados:
     - `DEBUG`: Detalles de diagnóstico de bajo nivel.
     - `INFO`: Hitos del ciclo de vida del servicio (modelo cargado, tarea en cola).
     - `WARNING`: Anomalías recuperables o peticiones rechazadas con contexto.
     - `ERROR`: Excepciones no controladas con traza de pila (stack trace).
2. **Endpoint de Diagnóstico `/health` o `/ready`**:
   - Debe ser ultraligero y responder en < 5ms.
   - Debe verificar el estado de los componentes vitales (modelo cargado, ruta de pesos disponible, arquitectura y clases en memoria) devolviendo HTTP 200 con JSON informativo.

---

## 5. Procedimiento de Auditoría Pre-Producción Paso a Paso

Al auditar cualquier proyecto para producción:

```bash
# 1. Escanear rutas locales hardcodeadas
grep -rnE "(/home/|/Users/|C:\\Users\\)" src/ scripts/ lib/ app/

# 2. Escanear sentencias print() de depuración
grep -rnE "(print\(|console\.log\()" src/ lib/ app/

# 3. Comprobar que safe_extract verifique relative_to contra zip-slip
grep -rnE "(extractall|extract)" src/

# 4. Verificar cabeceras de seguridad en la respuesta HTTP
curl -s -I http://127.0.0.1:8000/health | grep -iE "(content-security-policy|nosniff|sameorigin|coop|strict-transport)"

# 5. Ejecutar la suite completa de pruebas
pytest tests/ -v
```

