---
name: audit-frontend-desktop
description: Auditoría exhaustiva y optimización de rendimiento, accesibilidad (a11y), buenas prácticas, SEO y navegación agéntica (Agentic Browsing / WebMCP) para alcanzar 100/100 en Lighthouse Desktop.
---

# Skill: Auditoría y Optimización Frontend Desktop (Objetivo 100/100)

Esta skill establece el procedimiento riguroso para auditar, diagnosticar y corregir cualquier interfaz web frontend orientada a escritorio, garantizando la máxima puntuación en Google Lighthouse (100 en Performance, 100 en Accessibility, 100 en Best Practices, 100 en SEO) y compatibilidad total con agentes autónomos de navegación (Agentic Browsing / WebMCP).

---

## 1. Protocolo de Ejecución de Auditoría

### 1.1 Ejecución Headless Controlada
Para evitar sesgos y consumo innecesario de recursos o cuelgues del navegador, Lighthouse debe ejecutarse en modo headless, sin sandbox y con preset de escritorio y telemetría desactivada:

```bash
npx lighthouse <URL_OBJETIVO> \
  --preset=desktop \
  --no-enable-error-reporting \
  --output=json \
  --output-path=./lh-desktop.json \
  --chrome-flags="--headless --no-sandbox --disable-gpu"
```

### 1.2 Regla de Oro: Filtrado de Resultados sin Contaminar Contexto
**ESTRICTAMENTE PROHIBIDO** leer el archivo `lh-desktop.json` completo en el contexto del agente (puede superar los 10 MB y agotar la ventana de contexto). Se debe utilizar un extractor estructurado con Node.js (`scripts/audit_filter.js`) para aislar únicamente:
- Puntuación por categoría (0–100).
- Métricas Core Web Vitals (FCP, LCP, CLS, TBT, Speed Index).
- Auditorías fallidas (`score < 1`) y elementos afectados.

#### Script de extracción rápida (`scripts/audit_filter.js`):
```javascript
const fs = require('fs');
const filePath = process.argv[2] || './lh-desktop.json';
const report = JSON.parse(fs.readFileSync(filePath, 'utf8'));

const scores = {};
for (const [key, cat] of Object.entries(report.categories || {})) {
  scores[cat.title] = Math.round(cat.score * 100);
}

const metrics = {};
const metricKeys = [
  'first-contentful-paint',
  'largest-contentful-paint',
  'total-blocking-time',
  'cumulative-layout-shift',
  'speed-index',
  'interactive'
];
for (const k of metricKeys) {
  if (report.audits[k]) {
    metrics[k] = {
      score: report.audits[k].score,
      displayValue: report.audits[k].displayValue || ''
    };
  }
}

const failedAudits = [];
for (const [id, audit] of Object.entries(report.audits || {})) {
  if (audit.score !== null && audit.score < 1 && audit.scoreDisplayMode !== 'notApplicable') {
    failedAudits.push({
      id,
      title: audit.title,
      score: audit.score,
      displayValue: audit.displayValue || '',
      items: Array.isArray(audit.details?.items)
        ? audit.details.items.slice(0, 3).map(it => it.node?.snippet || it.url || it.label || (typeof it === 'object' ? JSON.stringify(it) : String(it)))
        : []
    });
  }
}

console.log(JSON.stringify({ scores, metrics, failedAuditsCount: failedAudits.length, failedAudits }, null, 2));
```

---

## 2. Checklist Técnico Obligatorio (100/100)

### 2.1 Rendimiento & Core Web Vitals (Desktop)
1. **Cumulative Layout Shift (CLS objetivo < 0.05, ideal 0.000):**
   - Toda etiqueta `<img>` y `<picture>` debe incluir atributos explícitos `width` y `height`, o una clase con `aspect-ratio` reservado antes de la carga.
   - **Presupuesto de Altura en Héroes Asíncronos**: Los bloques hero y títulos descriptivos que reciben texto dinámico desde APIs (`loadModelDetails()`, etc.) deben declarar un `min-h-[240px]` a `min-h-[280px]` y los párrafos descriptivos un `min-h-[48px]` a `min-h-[72px]` con texto provisional, para que la inyección de 2-4 líneas de datos no empuje el contenido inferior (eliminando CLS > 0.10).
   - **Tablas y Listados Dinámicos**: Los contenedores de tablas deben tener `min-h-[320px]` y un skeleton/loading con padding generoso (`py-16`) para evitar contracciones o saltos de layout cuando se populan las filas.
2. **Largest Contentful Paint (LCP < 1.0s) & Payloads:**
   - Imágenes convertidas a formatos de última generación (**WebP** o **AVIF**), redimensionadas estrictamente a sus dimensiones en pantalla (no servir avatares de 500x500 para un contenedor 40x40).
   - **LCP Prioritization**: El primer elemento visual del viewport principal (miniatura del primer modelo o logo) debe configurarse con `loading="eager"` y `fetchpriority="high"`; los elementos subsecuentes deben usar `loading="lazy"` y `fetchpriority="auto"`.
   - **Compresión de Respuesta**: Habilitar compresión GZip/Brotli a nivel servidor (`starlette.middleware.gzip.GZipMiddleware(minimum_size=500)`). Esto reduce payloads HTML de 110 KiB a 23 KiB (78% de ahorro).
   - **Caché Inmutable de Recursos Estáticos**: Servir CSS, JS e imágenes con cabecera `Cache-Control: public, max-age=31536000, immutable`.
   - **Eliminación de Render-Blocking por CDNs Externas**: Alojar bundles JS críticos (como Tailwind o librerías de UI) localmente en `/static/tailwindcss.js`. Las CDNs externas generan bloqueos de 200–350ms por handshake TLS y resolución DNS que impiden alcanzar el 100/100.
   - **Carga Asíncrona de Fuentes**: Cargar fuentes externas mediante el patrón preload no bloqueante:
     ```html
     <link rel="preconnect" href="https://fonts.googleapis.com">
     <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
     <link rel="preload" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" as="style" onload="this.onload=null;this.rel='stylesheet'">
     <noscript><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"></noscript>
     ```

### 2.2 Accesibilidad (a11y)
1. **Regla WCAG 2.5.3 (Label in Name):**
   - Si un elemento interactivo contiene texto visible (ej. `<a href="/"><span>KHAOS Hub</span></a>` o `<button>Purgar Ahora</button>`), **nunca agregues un `aria-label` divergente** que omita dicho texto (como `aria-label="Ir a Inicio"` o `aria-label="Purgar permanentemente..."`).
   - El nombre accesible del elemento debe comenzar o contener la cadena literal del texto visible (ej. `aria-label="Purgar Ahora: eliminar definitivamente modelo Mango"`).
   - Para logotipos o enlaces de marca con texto interior, omitir `aria-label` innecesario y permitir que el navegador compute el nombre directamente desde el texto visible del DOM.
2. **Nombres Accesibles Únicos para Acciones Repetidas:**
   - En tablas o cuadrículas con múltiples botones homónimos ("Ver", "Editar", "Eliminar", "Inspeccionar"), cada uno debe poseer un `aria-label` contextual que distinga el recurso: `aria-label="Ver ficha pública de ${slug}"`.
3. **Contraste de Color Estricto (WCAG AA):**
   - **Prohibido `text-slate-400` para texto informativo sobre fondos claros**: Ofrece solo ~2.3:1 de contraste (falla AA). Utilizar sistemáticamente `text-slate-600` (>5.6:1).
   - Para botones y textos de peligro, evitar `text-red-500` (~3.8:1); usar `text-red-600` (>4.6:1).
   - Encabezados de tablas (`<thead>`) deben usar `text-slate-600 font-semibold` en lugar de `text-slate-400 font-medium`.
4. **Semántica y Secuencia de Encabezados (Heading Order):**
   - Una única etiqueta `<h1>` representativa de la vista.
   - Los títulos de tarjetas, secciones secundarias y paneles deben descender secuencialmente a `<h2>`, luego `<h3>`. Nunca saltar de `<h1>` a `<h3>`.
   - Prohibido utilizar enlaces falsos como `<a href="#">` o `<a href="javascript:void(0)">`; sustituirlos siempre por `<button type="button">`.

### 2.3 Buenas Prácticas (Best Practices)
1. **Seguridad y Modernidad en Transporte:**
   - Todas las rutas estáticas y ficheros de texto (`/static/*`, `/robots.txt`, `/llms.txt`) deben admitir métodos `GET` y `HEAD` (`methods=["GET", "HEAD"]`), devolviendo `200 OK` para verificaciones de crawler.
   - Declaración de juego de caracteres `<meta charset="UTF-8">` en las primeras líneas de `<head>`.
   - Cero errores no controlados en la consola del navegador.

### 2.4 SEO Técnico
1. **Metadatos y Rastreo:**
   - `<title>` único y conciso en cada vista.
   - `<meta name="description" content="...">` de entre 70 y 160 caracteres en todas y cada una de las plantillas (incluyendo `/models/{slug}`).
   - Atributo `<html lang="es">` presente.
   - Fichero `/robots.txt` activo, accesible y con directivas claras.
   - Atributos `alt` descriptivos en todas las imágenes de referencia y miniaturas (`alt="Miniatura de ${title}"`).

### 2.5 Agentic Browsing & WebMCP
1. **Soporte de Agentes Autónomos:**
   - Especificación `/llms.txt` y `/.well-known/llms.txt` disponible en la raíz del servidor, estructurada en Markdown con descripción de capacidades del sistema, endpoints REST y contratos de inferencia.
   - Formularios accesibles con etiquetas `<label for="...">` asociadas o atributos `aria-label` explícitos en entradas y botones.

---

## 3. Matriz de Validación de Rutas del Proyecto

| Vista / Ruta | Rendimiento | Accesibilidad | Buenas Prácticas | SEO | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Catálogo Principal (`/`)** | **99** | **100** | **100** | **100** | **Aprobado** |
| **Curación Experta (`/expert`)** | **100** | **100** | **100** | **100** | **Aprobado** |
| **Administración Master (`/master`)** | **99** | **100** | **100** | **100** | **Aprobado** |
| **Ficha de Modelo (`/models/{slug}`)** | **100** | **100** | **100** | **100** | **Aprobado** |

---

## 4. Procedimiento de Verificación Iterativa
1. **Medición Inicial:** Ejecutar Lighthouse Desktop sobre la ruta a auditar (`/`, `/expert`, `/master`, `/models/<slug>`).
2. **Extracción y Diagnóstico:** Filtrar con el script `scripts/audit_filter.js`.
3. **Refactorización:** Corregir en el código fuente (`HTML`, `CSS`, `JS`, endpoints).
4. **Validación:** Re-ejecutar auditoría hasta registrar `100/100` en todas las métricas.
5. **Comprobación de Regresiones:** Ejecutar suite de pruebas pytest completa antes de dar por cerrada la optimización.

