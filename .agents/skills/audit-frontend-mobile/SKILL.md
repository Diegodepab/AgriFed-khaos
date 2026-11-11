---
name: audit-frontend-mobile
description: Auditoría exhaustiva y optimización frontend en dispositivos móviles (Throttling 4G/CPU, Touch Targets 48x48px, Viewport responsivo, tipografía legible >=12px y contención CLS) para alcanzar 100/100 en Lighthouse Mobile.
---

# Skill: Auditoría y Adaptación Frontend Móvil (Objetivo 100/100)

Esta skill establece el procedimiento riguroso para auditar, diagnosticar y optimizar cualquier interfaz web frontend para dispositivos móviles y pantallas táctiles, superando las restricciones de CPU estrangulada (4x slowdown), red 4G simulada, ergonomía táctil estricta, tipografía legible y eliminación total de saltos de diseño (CLS) y desbordamientos horizontales en viewports reducidos (375px - 390px).

---

## 1. Protocolo de Ejecución de Auditoría Móvil

### 1.1 Ejecución Headless Mobile en Lighthouse CLI
Por defecto, Lighthouse CLI aplica emulación móvil completa (estrangulamiento de CPU 4x, red 4G lenta Lantern, viewport 412x823px):

```bash
npx lighthouse <URL_OBJETIVO> \
  --no-enable-error-reporting \
  --output=json \
  --output-path=./lh-mobile.json \
  --chrome-flags="--headless --no-sandbox --disable-gpu"
```

### 1.2 Regla de Filtrado Estricto de Datos
**PROHIBIDO** cargar el archivo `lh-mobile.json` completo en el contexto del agente. Se utiliza el filtro Node.js de auditorías:
```bash
node scripts/audit_filter.js ./lh-mobile.json
```
Para inspeccionar detalles específicos de una auditoría fallida:
```bash
node -e "const r = JSON.parse(require('fs').readFileSync('lh-mobile.json')); console.log(JSON.stringify(r.audits['<audit-id>'].details, null, 2));"
```

---

## 2. Checklist Técnico Obligatorio Móvil

### 2.1 Rendimiento bajo CPU Estrangulada (TBT = 0 ms & LCP < 1.8 s)
1. **Eliminación de Compiladores JIT en Tiempo de Ejecución**:
   - **Prohibido** ejecutar compiladores CSS en el cliente (como `cdn.tailwindcss.com` o `tailwindcss.js` standalone) en producción. En CPUs móviles throttled (4x slowdown), el parsing en JavaScript bloquea el hilo principal entre 450 ms y 700 ms (TBT excesivo) y añade más de 2.5 segundos de Render Delay.
   - **Solución Obligatoria**: Compilar las utilidades Tailwind CSS de forma estática y minificada (`npx -y tailwindcss@3 -i input.css -o bundle.min.css --minify`). El bundle estático reduce el payload de ~120 KiB JS a solo ~13 KiB gzipped CSS y lleva el TBT a 0 ms exactos.
2. **Pre-carga y Hojas de Estilo no Bloqueantes**:
   - Enlazar estilos con `<link rel="preload" href="/static/khaos-bundle.min.css" as="style">` seguido de `<link rel="stylesheet" href="/static/khaos-bundle.min.css">`.
   - Scripts de soporte siempre con `defer` o cargados al final del `<body>`.

### 2.2 Tipografía Legible Móvil (`font-size` >= 12px)
1. **Regla del 60% Mínimo Legible**:
   - Lighthouse Mobile penaliza severamente (`font-size: 0`, afectando Best Practices y SEO) si más del 40% del texto visible en pantalla tiene un tamaño inferior a 12px (0.75rem).
2. **Prohibición de Micro-tipografías**:
   - Prohibido el uso indiscriminado de clases arbitrarias como `text-[9px]`, `text-[10px]` o `text-[11px]` en tablas, chips y metadatos.
   - Estandarizar todas las etiquetas secundarias, metadatos y badges en `text-xs` (12px / 1rem line-height).

### 2.3 Ergonomía Táctil & Touch Targets (`target-size`)
1. **Dimensiones y Espaciado Mínimo**:
   - Todo botón, enlace, selector, pestaña o control táctil debe medir al menos **48x48 px** de área de toque, o contar con separación perimetral (`padding` o `margin`) adecuada para evitar toques accidentales solapados.
   - En filas de tablas y selectores de filtros compactos, asegurar `min-h-[44px]` o `py-2.5 px-3` con separación de al menos `gap-2` entre controles interactivos contiguos.
2. **Meta Viewport Seguro**:
   - Debe existir `<meta name="viewport" content="width=device-width, initial-scale=1.0">`.
   - **Prohibido** `user-scalable=no` o `maximum-scale=1.0`.

### 2.4 Prevención de Desbordamiento y Desplazamiento de Diseño (CLS < 0.05)
1. **Pre-renderizado de Metadatos en Servidor (Evitar Re-layouts Asíncronos)**:
   - En vistas de detalle de modelo o catálogo, la inyección asíncrona de títulos, descripciones o badges en el cliente puede provocar que los elementos inferiores (ej. pestañas de inferencia o formularios) salten bruscamente de posición cuando se recibe la respuesta de red.
   - Inyectar títulos, descripciones y badges conocidos directamente en el HTML inicial desde el servidor (vía plantillas Jinja2 o reemplazo de tokens) garantiza un renderizado estable desde el primer cuadro (CLS < 0.02).
2. **Contención Flexbox de Textos en Segmentos Porcentuales**:
   - En barras de partición (ej. sliders de split Train/Val/Test), los elementos hijos `<span>` con texto dentro de contenedores `display: flex` tienen por defecto `min-width: auto`. En pantallas móviles estrechas (375px), segmentos pequeños (10%) provocan que el texto desborde visualmente el bloque coloreado, cayendo sobre el fondo claro y arruinando el ratio de contraste.
   - **Regla Estricta**: Usar siempre `min-w-0 max-w-full overflow-hidden block truncate` en el hijo flex y contrastes superiores a 4.5:1 (ej. `bg-blue-800` y `bg-emerald-700` con texto blanco).
3. **Controles Deslizantes Personalizados Accesibles (ARIA Slider)**:
   - Todo elemento interactivo con `role="slider"` requiere de forma obligatoria `aria-valuemin`, `aria-valuemax` y `aria-valuenow`, además de `aria-label` descriptivo y `tabindex="0"`.
   - Los valores de `aria-valuenow` deben sincronizarse dinámicamente mediante JavaScript en cada evento de arrastre o cambio numérico.
4. **Aislamiento de Tablas Densas (`overflow-x`)**:
   - Encapsular tablas de datos extensas en contenedores `<div class="overflow-x-auto no-scrollbar">` con márgenes compensados para permitir desplazamiento táctil suave sin provocar scroll horizontal a nivel de documento (`body`).

---

## 3. Matriz de Validación de Rutas Móvil

Para certificar la conformidad de la plataforma móvil, verificar con Lighthouse CLI y `scripts/audit_filter.js` las rutas críticas:

| Ruta | Performance | Accessibility | Best Practices | SEO | TBT | CLS |
|---|---|---|---|---|---|---|
| `/` (Catálogo Móvil) | 100 | 100 | 100 | 100 | 0 ms | 0.004 |
| `/expert` (Curación Móvil) | 99-100 | 100 | 100 | 100 | 0 ms | 0.061 |
| `/master` (Gobernanza Móvil) | 99-100 | 100 | 100 | 100 | 0 ms | 0.073 |
| `/models/<slug>` (Ficha Móvil) | 100 | 100 | 100 | 100 | 0 ms | 0.018 |
| `/master/models/new` (Creación Móvil) | 100 | 100 | 100 | 100 | 0 ms | 0.007 |

---

## 4. Procedimiento de Verificación y Cierre
1. Ejecutar Lighthouse Mobile sobre cada ruta del proyecto.
2. Confirmar que ninguna métrica de categoría esté por debajo de 99/100.
3. Ejecutar la suite completa de pruebas unitarias (`pytest tests/`) para verificar que no existan regresiones en APIs, templates o contratos de interfaz.
4. Registrar casos de borde aprendidos en esta especificación.
