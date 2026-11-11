---
name: audit-backend-architecture-and-smells
description: Auditoría implacable de arquitectura backend, modularización, contratos de datos estrictos y erradicación de código simulado (anti-mocking/anti-placeholders) y fallos silenciados. Aplicable a cualquier stack (Python, TypeScript/Node, Go, C#).
---

# Skill: Auditoría de Arquitectura Backend, Modularización y "Anti-Mocking"

Esta skill define el estándar universal de ingeniería para garantizar arquitecturas de backend limpias, mantenibles, robustas y de producción real. Combina métricas estáticas de complejidad y tamaño (estándar CodeClimate/SonarQube) con una política de **cero tolerancia a código simulado**, errores silenciados y controladores monolíticos ("código espagueti").

Es **agnóstica a la tecnología** y aplicable a cualquier lenguaje y ecosistema backend (Python/FastAPI/Django, TypeScript/Express/Nest, Go/Echo/Gin, C#/ASP.NET Core).

---

## 1. Detección Implacable de "Código que Simula" (Anti-Placeholders & Fake Logic)

Fuera del directorio dedicado de pruebas automatizadas (`tests/`, `__tests__/`, `spec/`), el código de producción debe realizar **exclusivamente operaciones reales**:

1. **Prohibición de Respuestas Falsas o Hardcodeadas**:
   - Ningún endpoint o servicio puede devolver respuestas sintéticas fijas (ej. `return {"status": "ok", "processed": 100}`) sin haber ejecutado la lógica de negocio, consulta a base de datos, llamada I/O o procesamiento en segundo plano correspondiente.
2. **Prohibición de Retardos Artificiales**:
   - Queda terminantemente prohibido el uso de retardos artificiales para simular carga o latencia (ej. `time.sleep()`, `setTimeout`, `Thread.Sleep()`, `time.Sleep()`) en rutas de producción. Si una operación es asíncrona, debe ser orquestada mediante un sistema de colas, procesos en background o hilos de trabajo reales.
3. **Cero Comentarios de Deuda Técnica no Resuelta**:
   - Escaneo estricto contra palabras clave: `TODO`, `FIXME`, `HACK`, `MOCK`, `PLACEHOLDER`, `FOR NOW`, `TEMPORARY`, `STUB`.
   - Si una funcionalidad no está completada o lista para producción, no debe simularse; debe diseñarse la abstracción adecuada o rechazar la petición con un código de error semántico claro (ej. `501 Not Implemented`).
4. **Cero Métodos o Funciones Vacías (Stubs)**:
   - Prohibido `def action(): pass`, `{ /* do nothing */ }` o clases ficticias que no implementen el contrato declarado.

---

## 2. Erradicación de Errores Silenciados (Zero Silent Failures)

1. **Prohibición de Captura Genérica Ciega**:
   - Prohibido `except Exception: pass`, `except: pass`, `catch (e) {}` o bloques que capturen el error y devuelvan `null`/`None`/`{}` sin registrar (loggear) ni propagar la causa raíz.
2. **Registro de Fallos con Contexto**:
   - Todo bloque `catch` / `except` debe:
     - Capturar la excepción más específica posible (`FileNotFoundError`, `ValueError`, `ValidationError`, `DbConnectionError`, `(json.JSONDecodeError, OSError)`).
     - Registrar el stack trace o contexto descriptivo en los logs con nivel `error` o `warning` (usando el logger del módulo, nunca `print`).
     - Propagar la excepción o transformarla en una respuesta HTTP semántica apropiada (`400 Bad Request`, `404 Not Found`, `409 Conflict`, `422 Unprocessable Entity`, `500 Internal Server Error`).

---

## 3. Modularización y Separación de Responsabilidades (SoC)

### 3.1 La Regla de Capas Estricta (Controller / Service / Repository)
Los controladores o manejadores de rutas (routers) deben ser **delgados (Thin Controllers)**:
- **Responsabilidad del Router/Controller**:
  1. Validar el payload de entrada contra un esquema formal.
  2. Extraer parámetros de autenticación/sesión si aplica.
  3. Invocar al método del **Servicio** correspondiente.
  4. Devolver el DTO o modelo de respuesta tipado con el código HTTP correspondiente.
- **Prohibido en Rutas/Controllers**:
  - Manipulación directa de archivos en disco (`open()`, `fs.readFile()`, `os.remove()`).
  - Lógica de cálculo o inferencia pesada.
  - Consultas directas a base de datos o queries SQL en línea.
  - Orquestación de subprocesos y comandos de terminal.
  Toda esta lógica pertenece a la capa de **Servicios** (`services/`, `domain/`, `operations/`).

### 3.2 Patrón de Coordinador Delgado y Re-exportación para Compatibilidad
Al refactorizar archivos monolíticos de entrada (como `serve.py` o `server.ts`):
- El archivo de entrada debe ser un coordinador ligero (< 200 líneas) que inicializa middlewares, monta los routers temáticos y orquesta el ciclo de vida (startup/shutdown).
- **Re-exportación con `__all__`**: Debe re-exportar los tipos, constantes y constructores públicos que la suite de tests existente o consumidores externos importen, evitando romper la API pública histórica.
- **Inyección de Estado Compartido**: Proveer acceso desacoplado al estado global (`request.app.state` o Dependency Injection container) para permitir que los fixtures de pruebas sigan inyectando mocks de test sin alterar el código de producción.

---

## 4. Límites de Complejidad Ciclomática y Tamaño (Estándar CodeClimate)

1. **Límites de Archivo**:
   - Ningún archivo de lógica de negocio debe superar las **300-400 líneas**.
   - Si un archivo excede este umbral, debe dividirse en submódulos cohesivos (ej. separar rutas en submódulos temáticos: `catalog.py`, `jobs.py`, `inference.py`, `governance.py`).
2. **Límites de Función/Método**:
   - Ninguna función debe superar las **50 líneas de código**.
   - Ninguna función debe tener más de **3 niveles de anidamiento** (`if / for / try`).
3. **Refactorización Obligatoria**:
   - **Cláusulas de Guarda (Early Returns)**: Validar precondiciones al inicio y retornar inmediatamente para aplanar la indentación.
   - **Extracción de Funciones Auxiliares**: Extraer bloques repetitivos o algoritmos internos a funciones privadas puras con nombres semánticos.

---

## 5. Tipado Estricto y Contratos de Datos

1. **Prohibición de Diccionarios y Objetos Genéricos**:
   - Prohibido el uso de `dict`, `Dict[str, Any]`, `any`, `Object`, `map[string]interface{}` como firma de entrada o salida en endpoints y servicios de negocio.
2. **Esquemas Formales Obligatorios**:
   - Python: Modelos `Pydantic` (`BaseModel`) o `dataclasses` tipados.
   - TypeScript: Interfaces estrictas con validación en runtime (`Zod`, `class-validator`, `TypeBox`).
   - C#: DTOs (`record`, `class`) con DataAnnotations.
   - Go: `struct` fuertemente tipados con tags JSON.
3. **Consistencia de Respuestas de API**:
   - Toda respuesta de éxito o error debe seguir una estructura predecible con tipos explícitos para facilitar el consumo del cliente y la generación de OpenAPI / Swagger.

---

## 6. Procedimiento de Auditoría Paso a Paso

Al auditar cualquier repositorio, ejecutar esta secuencia:

### Paso 1: Escaneo de Mocks, Placeholders y TODOs
```bash
# Buscar TODOs, MOCKs, STUBs y retardos artificiales fuera de tests
grep -rnE "(TODO|FIXME|HACK|MOCK|PLACEHOLDER|sleep\()" src/ lib/ app/ --exclude-dir=tests --exclude-dir=__tests__
```

### Paso 2: Escaneo de Fallos Silenciados
```bash
# Python: buscar bloques except ciegos
python3 -c '
import ast, glob
for path in glob.glob("src/**/*.py", recursive=True) + glob.glob("scripts/**/*.py", recursive=True):
    tree = ast.parse(open(path).read(), path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                print(f"Silent failure in {path}:{node.lineno}")
'

# JavaScript / TypeScript: buscar catch vacíos
grep -rnE "catch\s*\([a-zA-Z0-9_]*\)\s*\{\s*\}" src/
```

### Paso 3: Análisis de Líneas y Archivos Monolíticos
```bash
# Identificar archivos que superan 400 líneas
wc -l $(find src/ scripts/ -name "*.py" -o -name "*.ts" -o -name "*.go" -o -name "*.cs") | sort -n
```

### Paso 4: Refactorización y Verificación de Contratos
1. Diseñar schemas Pydantic/DTOs para payloads y respuestas de endpoints.
2. Extraer lógica pesada hacia servicios de dominio dedicados (`dataset_service`, `inference_service`, `mlops_service`, `jobs`).
3. Convertir controladores en routers temáticos delgados (`api/catalog.py`, `api/datasets.py`, etc.).
4. Reducir el archivo de entrada a un coordinador liviano con re-exportación de símbolos históricos para no romper integraciones.
5. Ejecutar la suite completa de tests automatizados (`pytest tests/`, `npm test`, etc.) hasta alcanzar el **100% de tests aprobados**.
