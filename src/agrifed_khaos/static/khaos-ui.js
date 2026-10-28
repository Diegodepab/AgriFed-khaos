/**
 * KHAOS Hub — Shared Frontend Runtime & Component Library (v2.0)
 * Provides unified Catalog Card & Thumbnail rendering, accessible Toasts,
 * formatting utilities, and keyboard/modal accessibility helpers.
 */
(function (global) {
  'use strict';

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatText(s) {
    if (!s) return '';
    return String(s).replace(/[_-]/g, ' ').replace(/\b\w/g, function (l) {
      return l.toUpperCase();
    });
  }

  function formatBytesMB(sizeBytes, fallbackText) {
    if (typeof sizeBytes === 'number' && sizeBytes > 0) {
      return (sizeBytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
    return fallbackText || '~42.7 MB';
  }

  /**
   * Accessible Toast Notification System
   */
  function ensureToastContainer() {
    var container = document.getElementById('khToastContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'khToastContainer';
      container.className = 'kh-toast-container';
      container.setAttribute('aria-live', 'polite');
      container.setAttribute('aria-atomic', 'false');
      document.body.appendChild(container);
    }
    return container;
  }

  function toast(message, type, durationMs) {
    var kind = type || 'info';
    var ttl = typeof durationMs === 'number' ? durationMs : 6000;
    var container = ensureToastContainer();

    var el = document.createElement('div');
    el.className = 'kh-toast kh-toast-' + kind;
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');

    var iconHtml = '';
    if (kind === 'success') {
      iconHtml = '<svg class="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (kind === 'error') {
      iconHtml = '<svg class="w-4 h-4 text-rose-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    } else if (kind === 'warning') {
      iconHtml = '<svg class="w-4 h-4 text-amber-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    } else {
      iconHtml = '<svg class="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }

    el.innerHTML =
      iconHtml +
      '<div class="flex-1 pr-1 font-medium leading-relaxed">' + escapeHtml(message) + '</div>' +
      '<button type="button" class="text-slate-400 hover:text-white p-1 rounded-md transition-colors cursor-pointer shrink-0" aria-label="Cerrar notificación"><svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>' +
      (ttl > 0 ? '<div class="kh-toast-progress" style="animation-duration: ' + ttl + 'ms;"></div>' : '');

    var closeBtn = el.querySelector('button');
    var progressBar = el.querySelector('.kh-toast-progress');
    var removed = false;
    var timerId = null;
    var startTime = Date.now();
    var remainingTime = ttl;

    function removeToast() {
      if (removed) return;
      removed = true;
      if (timerId) clearTimeout(timerId);
      el.classList.add('kh-toast-out');
      setTimeout(function() {
        if (el.parentNode) el.parentNode.removeChild(el);
      }, 220);
    }

    function startTimer() {
      if (ttl <= 0) return;
      startTime = Date.now();
      timerId = setTimeout(removeToast, remainingTime);
    }

    function pauseTimer() {
      if (ttl <= 0) return;
      if (timerId) clearTimeout(timerId);
      remainingTime -= (Date.now() - startTime);
      if (progressBar) {
        progressBar.style.animationPlayState = 'paused';
      }
    }

    function resumeTimer() {
      if (ttl <= 0 || remainingTime <= 0) return;
      if (progressBar) {
        progressBar.style.animationPlayState = 'running';
      }
      startTimer();
    }

    if (closeBtn) closeBtn.addEventListener('click', removeToast);
    el.addEventListener('mouseenter', pauseTimer);
    el.addEventListener('mouseleave', resumeTimer);

    container.appendChild(el);
    startTimer();
    return el;
  }

  function checkFlashToast() {
    try {
      var item = sessionStorage.getItem('khaos_flash_toast');
      if (item) {
        sessionStorage.removeItem('khaos_flash_toast');
        var d = JSON.parse(item);
        if (d && d.message) {
          toast(d.message, d.type || 'info', d.duration || 6500);
        }
      }
    } catch (e) { /* ignore */ }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkFlashToast);
  } else {
    setTimeout(checkFlashToast, 50);
  }

  /**
   * Unified Thumbnail / Technical Placeholder Component
   * Ensures identical card header height with or without an image, lazy loading,
   * graceful fallback on image error, and clear "Referencia ilustrativa" badge.
   */
  function renderThumbnailHeader(m, options) {
    var opts = options || {};
    var slug = m.slug || m.name || 'model';
    var safeSlug = encodeURIComponent(slug);
    var domain = m.domain || 'General';
    var arch = m.architecture || 'resnet18';
    var isOutOfService = opts.previewMode ? false : !m.active_checkpoint;
    var hasExplicitThumb = Boolean(m.thumbnail_url && String(m.thumbnail_url).trim());
    var thumbSrc = hasExplicitThumb
      ? String(m.thumbnail_url).trim()
      : (opts.previewMode ? '' : '/api/models/' + safeSlug + '/thumbnail');

    var cropTag = m.crop
      ? '<span class="px-2 py-0.5 rounded-md bg-emerald-950/75 text-emerald-200 border border-emerald-700/50 text-[10px] font-bold tracking-wide backdrop-blur-sm">' +
        escapeHtml(m.crop) +
        '</span>'
      : '';

    var statusOverlay = isOutOfService
      ? '<span class="px-2 py-0.5 rounded-md bg-amber-500/90 text-white text-[10px] font-bold tracking-wide shadow-sm">Fuera de servicio</span>'
      : (m.has_calibration
          ? '<span class="px-2 py-0.5 rounded-md bg-emerald-600/90 text-white text-[10px] font-bold tracking-wide shadow-sm" title="Umbrales de confianza calibrados">Calibrado</span>'
          : '');

    var imgHtml = '';
    if (thumbSrc) {
      imgHtml =
        '<img src="' + escapeHtml(thumbSrc) + '" alt="Vista ilustrativa de ' + escapeHtml(m.title || slug) + '" loading="lazy" ' +
        'class="w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform duration-500" ' +
        'onerror="this.style.display=\'none\'; if(this.nextElementSibling){this.nextElementSibling.classList.remove(\'hidden\');} if(this.parentElement.querySelector(\'.kh-illus-badge\')){this.parentElement.querySelector(\'.kh-illus-badge\').classList.add(\'hidden\');}" />' +
        '<span class="kh-illus-badge absolute bottom-2 right-2.5 z-10 px-1.5 py-0.5 rounded bg-slate-950/70 text-slate-200 text-[9px] font-medium backdrop-blur-sm pointer-events-none">Referencia ilustrativa</span>';
    }

    var placeholderHiddenClass = thumbSrc ? 'hidden' : '';
    var placeholderHtml =
      '<div class="' + placeholderHiddenClass + ' absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 via-slate-900 to-indigo-950 text-slate-400 select-none">' +
        '<div class="flex flex-col items-center gap-1 opacity-75">' +
          '<svg class="w-7 h-7 text-indigo-400/80" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">' +
            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />' +
          '</svg>' +
          '<span class="text-[10px] font-mono uppercase tracking-wider text-slate-300">' + escapeHtml(arch) + ' • ' + escapeHtml(domain) + '</span>' +
        '</div>' +
      '</div>';

    return (
      '<div class="h-32 w-full bg-slate-900 relative overflow-hidden flex items-center justify-center shrink-0">' +
        imgHtml +
        placeholderHtml +
        '<div class="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-slate-950/20 to-transparent pointer-events-none"></div>' +
        '<div class="absolute top-2.5 left-3 right-3 flex items-center justify-between gap-1.5 z-10">' +
          '<div class="flex items-center gap-1.5 flex-wrap">' +
            '<span class="px-2.5 py-0.5 rounded-md bg-slate-900/90 text-white text-[10px] font-bold uppercase tracking-wider backdrop-blur-sm">' +
              escapeHtml(domain) +
            '</span>' +
            cropTag +
          '</div>' +
          statusOverlay +
        '</div>' +
        '<div class="absolute bottom-2 left-3 flex items-center gap-1.5 z-10">' +
          '<span class="text-[10px] font-mono text-white/95 bg-slate-900/75 px-2 py-0.5 rounded backdrop-blur-sm border border-white/10">' +
            escapeHtml(arch) +
          '</span>' +
        '</div>' +
      '</div>'
    );
  }

  /**
   * Unified Catalog Card HTML Renderer
   * Used by both Public Catalog (client.html) and Model Creation Studio (model_create.html).
   */
  function renderCatalogCardHtml(m, options) {
    var opts = options || {};
    var isPreview = Boolean(opts.previewMode);
    var slug = m.slug || m.name || 'nuevo-modelo';
    var safeSlug = encodeURIComponent(slug);
    var title = m.title || 'Título del Modelo';
    var desc = m.description || 'Diagnóstico visual automatizado de muestras e imágenes.';
    var classesList = Array.isArray(m.classes) ? m.classes : [];
    var isOutOfService = isPreview ? false : !m.active_checkpoint;
    var sizeMB = formatBytesMB(m.size_bytes, m.size_label || '~42.7 MB');
    var versionCount = Array.isArray(m.versions) && m.versions.length ? m.versions.length : 1;

    var tagsHtml = '';
    if (classesList.length > 0) {
      tagsHtml = classesList.slice(0, 4).map(function (c) {
        return '<span class="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 text-[11px] font-medium border border-slate-200/80">' +
          escapeHtml(formatText(c)) +
          '</span>';
      }).join('');
      if (classesList.length > 4) {
        tagsHtml += '<span class="px-2 py-0.5 text-[11px] font-semibold text-slate-500">+' + (classesList.length - 4) + ' más</span>';
      }
    } else {
      tagsHtml = '<span class="text-xs text-slate-400 italic">Sin clases catalogadas</span>';
    }

    var metaPills = [];
    if (m.target_element) {
      metaPills.push(
        '<span class="px-2 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200/70 text-[10px] font-semibold">Muestra: ' +
        escapeHtml(m.target_element) +
        '</span>'
      );
    }
    if (m.purpose) {
      metaPills.push(
        '<span class="px-2 py-0.5 rounded bg-violet-50 text-violet-700 border border-violet-200/70 text-[10px] font-semibold">Propósito: ' +
        escapeHtml(m.purpose) +
        '</span>'
      );
    }

    var footerActions = '';
    if (isPreview) {
      footerActions =
        '<div class="flex items-center gap-1.5">' +
          '<span class="px-2.5 py-1.5 rounded-lg border border-slate-200 text-slate-600 text-xs font-semibold bg-slate-50">Validar</span>' +
          '<span class="px-3.5 py-1.5 rounded-lg bg-slate-900 text-white text-xs font-bold shadow-sm">Usar Modelo &rarr;</span>' +
        '</div>';
    } else if (isOutOfService) {
      footerActions =
        '<div class="flex items-center gap-1.5">' +
          '<span class="px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-bold">Temporalmente fuera de servicio</span>' +
        '</div>';
    } else {
      var escapedSlugJs = String(slug).replace(/'/g, "\\'");
      footerActions =
        '<div class="flex items-center gap-1.5">' +
          '<a href="/models/' + safeSlug + '/expert" class="px-2.5 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold transition" title="Validación Experta HITL">Validar</a>' +
          '<button type="button" onclick="openAnalysisStudioBySlug(\'' + escapedSlugJs + '\')" class="px-3.5 py-1.5 rounded-lg bg-slate-900 hover:bg-indigo-600 text-white text-xs font-bold transition shadow-sm">Usar Modelo &rarr;</button>' +
        '</div>';
    }

    var titleHtml = isPreview
      ? '<h3 class="text-base font-bold text-slate-900 leading-snug mb-1.5 line-clamp-1">' + escapeHtml(title) + '</h3>'
      : '<h3 class="text-base font-bold text-slate-900 group-hover:text-indigo-600 transition leading-snug mb-1.5 line-clamp-1">' +
          '<a href="/models/' + safeSlug + '" class="hover:underline">' + escapeHtml(title) + '</a>' +
        '</h3>';

    var detailLinkHtml = isPreview
      ? '<span class="text-[11px] font-semibold text-indigo-600">Ver detalles &rarr;</span>'
      : '<a href="/models/' + safeSlug + '" class="text-[11px] font-semibold text-indigo-600 hover:text-indigo-800">Ver detalles &rarr;</a>';

    return (
      '<article class="kh-card kh-card-interactive overflow-hidden flex flex-col justify-between group h-full ' + (isOutOfService ? 'opacity-80' : '') + '">' +
        '<div>' +
          renderThumbnailHeader(m, opts) +
          '<div class="p-5">' +
            titleHtml +
            '<p class="text-xs text-slate-600 leading-relaxed line-clamp-2 mb-3 min-h-[2.25rem]">' +
              escapeHtml(desc) +
            '</p>' +
            (metaPills.length > 0 ? '<div class="flex flex-wrap gap-1.5 mb-3">' + metaPills.join('') + '</div>' : '') +
            '<div class="pt-3 border-t border-slate-100">' +
              '<div class="flex items-center justify-between text-[11px] text-slate-400 font-semibold uppercase tracking-wider mb-2">' +
                '<span>Clases Detectables</span>' +
                '<span class="text-slate-600 kh-tabular-nums">' + (m.num_classes || classesList.length || 0) + ' clases</span>' +
              '</div>' +
              '<div class="flex flex-wrap gap-1.5 min-h-[1.75rem]">' +
                tagsHtml +
              '</div>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="px-5 py-3.5 bg-slate-50/70 border-t border-slate-200/80 flex items-center justify-between gap-2 mt-auto">' +
          '<div class="flex flex-col">' +
            '<span class="text-[11px] text-slate-500 mono kh-tabular-nums">' + escapeHtml(sizeMB) + ' &bull; v' + versionCount + '.0</span>' +
            detailLinkHtml +
          '</div>' +
          footerActions +
        '</div>' +
      '</article>'
    );
  }

  // =========================================================================
  // CONTEXTUAL HELP MODAL SYSTEM (Linear/Vercel Aesthetic)
  // =========================================================================

  var HELP_CONTENT = {
    client: {
      scope: 'Catálogo de Modelos • Diagnóstico',
      title: '¿Cómo funciona el Catálogo de Modelos?',
      desc: 'Explora modelos de visión artificial agrícola y realiza diagnósticos en tiempo real sobre tus muestras.',
      steps: [
        {
          num: '1',
          title: 'Encuentra el modelo para tu cultivo o muestra',
          text: 'Filtra por cultivo o dominio (ej. Olivo, Aguacate, Frutales), elemento analizado (hoja, fruto) o escribe en el buscador instantáneo.'
        },
        {
          num: '2',
          title: 'Sube una fotografía o captura con la cámara',
          text: 'Haz clic en "Analizar Fotografía" en el modelo elegido. Si accedes desde el campo con tu móvil, pulsa "Tomar con la Cámara" para capturar la muestra directamente.'
        },
        {
          num: '3',
          title: 'Diagnóstico instantáneo y calibración',
          text: 'El sistema calcula la distribución de probabilidades, identifica la patología o defecto principal y muestra su nivel de certeza calibrado.'
        },
        {
          num: '4',
          title: 'Vistas flexibles y fichas técnicas',
          text: 'Alterna entre vista en Tarjetas, Fichas Técnicas o Lista compacta según tus preferencias de trabajo.'
        }
      ],
      tip: 'Recomendación de captura: Para un diagnóstico óptimo, toma fotografías nítidas con iluminación natural difusa sobre los síntomas a evaluar.'
    },
    expert: {
      scope: 'Validación Agronómica',
      title: '¿Cómo funciona el Espacio de Curación?',
      desc: 'Revisión experta de predicciones dudosas y validación de muestras para perfeccionar la precisión de los modelos.',
      steps: [
        {
          num: '1',
          title: 'Cola priorizada por incertidumbre (Uncertainty Sampling)',
          text: 'Las fotografías de campo se ordenan priorizando aquellas donde el modelo tuvo menor confianza (< 65%), maximizando la ganancia de tu tiempo experto.'
        },
        {
          num: '2',
          title: 'Certificación y corrección de diagnósticos',
          text: 'Si la predicción es correcta, pulsa "Confirmar Predicción". Si requiere ajuste, selecciona la clase adecuada en el desplegable (o introduce una nueva) y pulsa "Corregir".'
        },
        {
          num: '3',
          title: 'Descarte justificado de muestras',
          text: 'Si una imagen está borrosa, desenfocada o no corresponde a una muestra evaluable, descártala con su motivo para no contaminar el dataset.'
        },
        {
          num: '4',
          title: 'Colaboración Federada On-Premise',
          text: 'Descarga los pesos vigentes del modelo, entrena con tus datos locales y envía actualizaciones firmadas con HMAC para la siguiente ronda FedAvgM.'
        }
      ],
      tip: 'Impacto directo: Las muestras que certificas nutren la base verificada que los administradores utilizan para reentrenar y perfeccionar el modelo en caliente.'
    },
    master: {
      scope: 'Administración MLOps • Orquestación Federada',
      title: '¿Cómo funciona la Administración y MLOps?',
      desc: 'Supervisión integral del catálogo, orquestación de reentrenamientos HITL, agregación federada y ciclo de vida.',
      steps: [
        {
          num: '1',
          title: 'Registro y despliegue de modelos',
          text: 'Despliega nuevos modelos mediante "Registrar Modelo" cargando archivos de pesos .pt o datasets de imágenes (carpetas por clases o formato YOLO) con validación automática.'
        },
        {
          num: '2',
          title: 'Inspector lateral contextual',
          text: 'Haz clic en "Inspeccionar" en cualquier fila del catálogo para abrir su panel lateral: versiones activas, estadísticas HITL y bandeja de actualizaciones federadas.'
        },
        {
          num: '3',
          title: 'Fine-Tuning con Curación Experta (HITL)',
          text: 'Lanza reentrenamientos en caliente utilizando las muestras validadas por los técnicos de campo sin interrumpir la disponibilidad del servicio.'
        },
        {
          num: '4',
          title: 'Agregación FedAvgM y Papelera de Seguridad',
          text: 'Ejecuta rondas de agregación federada con momento y métricas de convergencia. Los modelos eliminados van a cuarentena temporal y se pueden restaurar en 1 clic.'
        }
      ],
      tip: 'Procesos en segundo plano: Consulta el progreso detallado de cualquier tarea de entrenamiento o agregación desde la consola del Job Runner inferior.'
    },
    detail: {
      scope: 'Ficha Técnica de Modelo • API & Inferencia',
      title: '¿Cómo utilizar la Ficha Técnica del Modelo?',
      desc: 'Evaluación en vivo, taxonomía completa de patologías detectables y código de integración API.',
      steps: [
        {
          num: '1',
          title: 'Diagnóstico interactivo en vivo',
          text: 'Prueba el modelo directamente en el navegador subiendo una imagen para evaluar su respuesta y distribución probabilística completa.'
        },
        {
          num: '2',
          title: 'Taxonomía y umbrales de certeza',
          text: 'Consulta la lista exhaustiva de patologías soportadas y los umbrales mínimos calibrados para cada condición fitosanitaria.'
        },
        {
          num: '3',
          title: 'Integración vía API (cURL / Python)',
          text: 'Copia los snippets de código listos para integrar el modelo en sensores de campo, drones o aplicaciones agrícolas externas.'
        },
        {
          num: '4',
          title: 'Descarga de pesos PyTorch (.pt)',
          text: 'Descarga el checkpoint activo para auditorías locales o accede directamente al espacio de curación experta.'
        }
      ],
      tip: 'Parámetro save_for_training: Al invocar /api/predict puedes enviar "save_for_training=true" para remitir casos dudosos a la cola de curación humana.'
    }
  };

  function getHelpSection() {
    var p = window.location.pathname || '';
    if (p.startsWith('/master')) return 'master';
    if (p.startsWith('/expert') || p.includes('/expert')) return 'expert';
    if (p.startsWith('/models/')) return 'detail';
    return 'client';
  }

  function ensureHelpModalElement() {
    var modal = document.getElementById('khContextualHelpModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'khContextualHelpModal';
      modal.setAttribute('role', 'dialog');
      modal.setAttribute('aria-modal', 'true');
      modal.setAttribute('aria-labelledby', 'khHelpTitle');
      modal.className = 'hidden fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm overflow-y-auto';

      modal.innerHTML =
        '<div class="bg-white rounded-2xl border border-slate-200/60 shadow-2xl max-w-xl w-full p-6 sm:p-7 space-y-5 my-6 animate-in fade-in duration-200">' +
          '<div class="flex items-start justify-between pb-3 border-b border-slate-100 gap-4">' +
            '<div class="space-y-1">' +
              '<span id="khHelpScope" class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-[10px] font-semibold bg-slate-50 text-slate-600 ring-1 ring-inset ring-slate-500/10 uppercase tracking-wider"></span>' +
              '<h3 id="khHelpTitle" class="text-lg font-semibold text-slate-900 tracking-tight pt-1"></h3>' +
              '<p id="khHelpDesc" class="text-xs text-slate-500 leading-relaxed"></p>' +
            '</div>' +
            '<button type="button" data-modal-close onclick="KhaosUI.closeHelp()" aria-label="Cerrar ayuda" class="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-slate-100 transition-colors">' +
              '<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18M6 6l12 12"/></svg>' +
            '</button>' +
          '</div>' +
          '<div id="khHelpStepsList" class="space-y-3 text-xs leading-relaxed max-h-[55vh] overflow-y-auto pr-1"></div>' +
          '<div id="khHelpTipBox" class="p-3.5 rounded-xl bg-slate-50/80 ring-1 ring-inset ring-slate-200/60 text-xs text-slate-600 flex items-start gap-2.5">' +
            '<svg class="w-4 h-4 text-amber-500 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>' +
            '<p id="khHelpTipText" class="text-[11px] leading-relaxed"></p>' +
          '</div>' +
          '<div class="pt-2 flex justify-end">' +
            '<button type="button" onclick="KhaosUI.closeHelp()" class="px-4 py-2 rounded-lg text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 transition-all shadow-2xs">' +
              'Entendido' +
            '</button>' +
          '</div>' +
        '</div>';

      modal.addEventListener('click', function (e) {
        if (e.target === modal) KhaosUI.closeHelp();
      });

      document.body.appendChild(modal);
    }
    return modal;
  }

  function openHelp(forceSection) {
    var secKey = forceSection || getHelpSection();
    var data = HELP_CONTENT[secKey] || HELP_CONTENT.client;
    var modal = ensureHelpModalElement();

    var scopeEl = document.getElementById('khHelpScope');
    var titleEl = document.getElementById('khHelpTitle');
    var descEl = document.getElementById('khHelpDesc');
    var stepsList = document.getElementById('khHelpStepsList');
    var tipText = document.getElementById('khHelpTipText');

    if (scopeEl) scopeEl.textContent = data.scope;
    if (titleEl) titleEl.textContent = data.title;
    if (descEl) descEl.textContent = data.desc;
    if (tipText) tipText.textContent = data.tip;

    if (stepsList) {
      stepsList.innerHTML = '';
      data.steps.forEach(function (step) {
        var row = document.createElement('div');
        row.className = 'flex items-start gap-3 p-3 rounded-xl bg-slate-50/60 ring-1 ring-inset ring-slate-200/40';
        row.innerHTML =
          '<span class="w-5 h-5 rounded-md bg-slate-900 text-white font-semibold flex items-center justify-center shrink-0 text-[11px]">' +
            escapeHtml(step.num) +
          '</span>' +
          '<div class="space-y-0.5">' +
            '<strong class="text-slate-900 font-semibold block text-xs">' + escapeHtml(step.title) + '</strong>' +
            '<p class="text-slate-600 text-[11px] leading-relaxed">' + escapeHtml(step.text) + '</p>' +
          '</div>';
        stepsList.appendChild(row);
      });
    }

    modal.classList.remove('hidden');
  }

  function closeHelp() {
    var modal = document.getElementById('khContextualHelpModal');
    if (modal) modal.classList.add('hidden');
    // Also support any existing legacy helpModal
    var legacyModal = document.getElementById('helpModal');
    if (legacyModal) legacyModal.classList.add('hidden');
  }

  // Global helper aliases for templates
  global.openContextualHelp = openHelp;
  global.closeContextualHelp = closeHelp;
  global.toggleHelpModal = function (show) {
    if (show === false) closeHelp();
    else openHelp();
  };

  // Global Escape key handler for open modals
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape') {
      var dialogs = document.querySelectorAll('[role="dialog"]:not(.hidden)');
      for (var i = dialogs.length - 1; i >= 0; i--) {
        var dlg = dialogs[i];
        var closeBtn = dlg.querySelector('[data-modal-close]');
        if (closeBtn && typeof closeBtn.click === 'function') {
          closeBtn.click();
          break;
        }
      }
    }
  });

  global.KhaosUI = {
    escapeHtml: escapeHtml,
    formatText: formatText,
    formatBytesMB: formatBytesMB,
    toast: toast,
    checkFlashToast: checkFlashToast,
    renderThumbnailHeader: renderThumbnailHeader,
    renderCatalogCardHtml: renderCatalogCardHtml,
    openHelp: openHelp,
    closeHelp: closeHelp
  };
})(window);

