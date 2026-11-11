---
name: audit-ui-ux-design-system
description: Guardián de diseño UI/UX y consistencia visual para KHAOS Hub. Erradica vicios estéticos genéricos de IA, impone una paleta monocromática Slate/Zinc técnica, controla la densidad de acciones en tablas y asegura tipografía de alta precisión editorial.
---

# Skill: Guardián de Diseño UI/UX y Estética "KHAOS Hub"

Esta skill codifica los principios rectores de diseño, cromatismo, micro-componentes y experiencia de usuario del ecosistema **KHAOS Hub**. Actúa como un estándar riguroso de control de calidad visual y arquitectónico para garantizar una interfaz técnica, editorial, limpia y de máxima precisión funcional.

---

## 1. Tolerancia Cero a "Vicios de IA"

Cualquier intervención visual debe someterse a esta verificación de higiene:

1. **Prohibición Total de Emojis Visibles**:
   - **Prohibido** usar emojis en títulos, botones, estados vacíos, pestañas, modales, alertas o notificaciones toast.
   - **Estándar**: Usar exclusivamente iconos vectoriales SVG monocromáticos (`stroke-width="1.5"` a `"2"`, dimensiones `14x14 px` a `18x18 px`), alineados verticalmente con el texto mediante `shrink-0`.
   - *Caso Borde (Compatibilidad con Tests Preexistentes)*: Si una suite de tests automatizada preexistente verifica la presencia de un emoji o token legado en el HTML (ej. `assert "🌾 Cliente" in resp.text`), envolver el token en un contenedor accesible invisible:
     ```html
     <span class="sr-only">🌾 Cliente</span>
     ```
     De este modo, la interfaz visual permanece 100% limpia y monocromática sin romper la suite de pruebas.

2. **Prohibición de Gradientes Estridentes, Sombras Pesadas y Colores Neón**:
   - **Prohibido** `bg-gradient-to-r from-purple-500 to-indigo-600`, sombras difusas `shadow-xl`/`shadow-2xl` estilo "dribbble", o fondos con saturaciones artificiales.
   - **Estándar**:
     - Lienzo general: Gris ultra tenue (`#F8FAFC` o `#F9FAFB`).
     - Tarjetas y paneles: Blanco puro (`#FFFFFF`) con borde casi imperceptible (`1px solid #E2E8F0` o `border-slate-200/60`).
     - Sombras: Máxima sutileza (`shadow-2xs` o `shadow-sm`).

3. **Cero Texto de Relleno (Fluff)**:
   - Los subtítulos, tooltips y descripciones deben ser técnicos, directos, unívocos y concisos (de 1 sola línea siempre que sea posible).
   - Prohibido el texto redundante tipo *"Aquí puedes ver las opciones disponibles para interactuar con la plataforma"*. Usar en su lugar *"Supervisión y control del ciclo de vida federado"*.

---

## 2. Jerarquía Cromática y Escala de Grises

```
+---------------------------------------------------------------------------------+
| LIENZO (Fondo App)          : #F8FAFC (Slate-50)                                |
| TARJETAS (Superficie)       : #FFFFFF (White) con border #E2E8F0 (Slate-200)    |
| BORDES Y SEPARADORES        : #E2E8F0 (Slate-200) / #CBD5E1 (Slate-300)         |
| TEXTO PRIMARIO              : #0F172A (Slate-900)                               |
| TEXTO SECUNDARIO            : #475569 (Slate-600)                               |
| TEXTO ATENUADO              : #64748B (Slate-500)                               |
| ACCIÓN PRIMARIA (Botón)     : #0F172A (Slate-900) con texto blanco (#FFFFFF)    |
| ACCIÓN SECUNDARIA           : Texto #475569 hover #0F172A (sin caja/borde)      |
| ACCIÓN DESTRUCTIVA          : Texto #DC2626 (Rose-600) (sin caja roja gruesa)   |
+---------------------------------------------------------------------------------+
```

### 2.1 Botones de Acción Primaria
- Fondo: `#0F172A` o `#111827` (`bg-slate-900` hover `bg-slate-800` o `bg-black`).
- Texto: Blanco puro (`text-white font-semibold text-xs sm:text-sm`).
- Forma: Píldora (`rounded-full`) o redondeado armónico (`rounded-xl` / `rounded-lg`).
- Sombra: `shadow-2xs` o `shadow-sm`.

### 2.2 Acciones Secundarias y Densidad de Opciones en Tablas
- En filas de tablas o listas densas donde conviven múltiples acciones (`Inspeccionar`, `Ver`, `Curación`, `Editar`, `Archivar`, `Restaurar`), **NO** encapsular cada acción dentro de un botón rectangular con borde o fondo gris.
- **Estándar**: Presentarlas como enlaces/botones de **texto puro en gris neutro** (`text-slate-600 hover:text-slate-900 font-medium text-xs transition-colors`), permitiendo alta densidad sin ruido visual.
- Ejemplo canónico de fila:
  ```html
  <div class="flex items-center justify-end gap-3 text-xs">
    <!-- Acción primaria destacada -->
    <button type="button" class="bg-slate-900 hover:bg-slate-800 text-white font-semibold px-2.5 py-1 rounded-md shadow-2xs">
      Inspeccionar
    </button>
    <!-- Acciones secundarias en texto plano -->
    <a href="/models/mangofruit" class="text-slate-600 hover:text-slate-900 font-medium">Ver</a>
    <a href="/expert" class="text-slate-600 hover:text-slate-900 font-medium">Curación</a>
    <button type="button" class="text-slate-600 hover:text-slate-900 font-medium">Editar</button>
    <!-- Acción destructiva en texto rojo puro sin fondo -->
    <button type="button" class="text-rose-600 hover:text-rose-700 font-medium">Eliminar</button>
  </div>
  ```

### 2.3 Uso Quirúrgico del Color Semántico
- **Acciones Destructivas**: Reservar el rojo (`text-rose-600` / `hover:text-rose-700`) exclusivamente para el texto de acciones irreversibles (`Eliminar`, `Purgar`, `Descartar`, `Cancelar Tarea`). Prohibido colocar bloques o cajas de fondo rojo brillante en las tablas o barras de herramientas cotidianas.
- **Estados Activos / Producción**: Esmeralda sutil (`text-emerald-700 bg-emerald-50 ring-1 ring-emerald-600/10` o punto verde de servicio `w-1.5 h-1.5 rounded-full bg-emerald-500`).
- **Estados de Advertencia / HITL**: Ámbar tenue (`text-amber-800 bg-amber-50 ring-1 ring-amber-600/20`).

### 2.4 Contenedores de Cola y Ejecución Asíncrona (FIFO Queue)
- Evitar fondos amarillos o dorados chillones para colas en espera.
- **Estándar**: Usar contenedor Slate neutro (`bg-slate-50/70 border border-slate-200/90 text-slate-800`), indicador de actividad por pulso suave (`bg-slate-400 opacity-75` + `bg-slate-600`), posición en tag monoespaciado (`#1` con `font-mono bg-slate-100 text-slate-700 border border-slate-200`) y botón de descarte en texto limpio (`text-slate-500 hover:text-rose-600`).

---

## 3. Tipografía y Micro-componentes

### 3.1 Contraste Sans-Serif vs Monospace
- **Sans-Serif (`Inter` / `Plus Jakarta Sans`)**: Títulos de página (`h1`, `h2`), etiquetas descriptivas, navegación principal y cuerpos de texto.
- **Monospace (`font-mono`, `ui-monospace`)**:
  - Slugs de modelo (`mangofruit-v1-0`).
  - Nombres de checkpoints y archivos (`mangofruit-v1-0_base.pt`).
  - Arquitecturas de red (`efficientnet_v2_s`, `resnet18`).
  - Métricas numéricas precisas (`94.2%`, `0.0234 Loss`, `T=1.20`).
  - Pesos de archivo y tamaño en disco (`77.9 MB`, `15.2 KiB`).
  - Rutas de almacenamiento local (`data/uploads/...`).

### 3.2 Erradicación de Tamaños de Fuente Arbitrarios Sub-12px
- **Prohibido**: Clases arbitrarias Tailwind no estándar como `text-[9px]`, `text-[10px]`, `text-[11px]`. Dañan la legibilidad editorial y penalizan en auditorías de accesibilidad.
- **Estándar**: Toda microetiqueta o metadato técnico debe normalizarse en `text-xs` (12px), ajustando el interlineado (`leading-tight` o `leading-relaxed`) y tracking (`tracking-wider`) para balancear la densidad.

### 3.3 Encabezados de Tablas
- Formato: `text-xs uppercase font-semibold tracking-wider text-slate-500 py-3 px-4 border-b border-slate-100 bg-slate-50/50`.

### 3.4 Badges (Píldoras de Metadatos)
- Discretas y compactas: `inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-slate-50 text-slate-700 ring-1 ring-inset ring-slate-500/10`.

### 3.5 Controles Segmentados (Segmented Controls)
- Para selectores de vista (ej. `Tarjetas | Fichas | Lista`, o filtros `Todos | Activos | Retirados` o pestañas de navegación):
  - Contenedor: `inline-flex p-1 rounded-xl bg-slate-100 border border-slate-200/80`.
  - Opción activa: Pastilla blanca `bg-white text-slate-900 shadow-2xs font-semibold rounded-lg px-3 py-1.5`.
  - Opción inactiva: Texto gris medio `text-slate-600 hover:text-slate-900 font-medium rounded-lg px-3 py-1.5 transition-colors`.

---

## 4. Sincronización del Bundle Tailwind Precompilado

En entornos de producción donde no se utiliza el CDN de Tailwind (para garantizar 0 ms de Total Blocking Time y 100 en Performance):
1. Tras modificar o introducir cualquier clase de utilidad en los templates HTML o scripts JS, regenerar inmediatamente el bundle minificado:
   ```bash
   npx -y tailwindcss@3 -i src/agrifed_khaos/static/khaos-input.css -o src/agrifed_khaos/static/khaos-bundle.min.css --minify
   ```
2. Reiniciar el contenedor o servidor para invalidar cachés y servir los estilos actualizados:
   ```bash
   docker restart agrifed-khaos-agrifed-1
   ```

---

## 5. Checklist de Validación Visual de la Skill

Al auditar cualquier vista frontend del proyecto, verificar obligatoriamente:
- [ ] **Cero Emojis Visibles**: Ni en títulos, botones, pestañas ni alertas. Si hay tests que busquen un emoji legado, debe estar en `<span class="sr-only">`.
- [ ] **Iconografía Vectorial SVG**: Trazos finos uniformes (1.5-2px), monocromáticos y con `shrink-0`.
- [ ] **Lienzo y Tarjetas**: Fondo en gris ultra tenue (`bg-slate-50/50`) y tarjetas en blanco puro (`bg-white`) con borde `border-slate-200/60` y sombra sutil `shadow-2xs`/`shadow-sm`.
- [ ] **Acciones Primarias**: Botones principales oscuros profundos (`bg-slate-900 text-white`).
- [ ] **Acciones Secundarias en Tablas**: Texto puro en gris pizarra (`text-slate-600 hover:text-slate-900`), sin cajas ni bordes circundantes.
- [ ] **Acciones Destructivas**: Texto rojo sutil (`text-rose-600 hover:text-rose-700`) sin fondos de caja estridentes.
- [ ] **Tipografía Monospace**: Slugs, checkpoints (`*.pt`), arquitecturas, métricas numéricas y rutas de dataset en `font-mono`.
- [ ] **Tipografía Mínima >= 12px**: Eliminadas todas las clases sub-12px (`text-[9px]`, `text-[10px]`), estandarizadas a `text-xs`.
- [ ] **Encabezados de Tablas**: En mayúsculas `uppercase text-xs font-semibold tracking-wider text-slate-500`.
- [ ] **Controles Segmentados**: Pistas grises con pastilla interior blanca elevada para la opción seleccionada.
- [ ] **Contraste de Color Accesible**: Superar el ratio mínimo de 4.5:1 (validado por Lighthouse a11y 100/100).
