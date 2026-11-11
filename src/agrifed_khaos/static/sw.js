/**
 * KHAOS Hub Progressive Web App (PWA) Service Worker
 * Provides offline caching for application shell, static assets, and ONNX models.
 */

const CACHE_NAME = 'khaos-hub-pwa-v28';
const STATIC_ASSETS = [
  '/',
  '/expert',
  '/static/khaos-bundle.min.css',
  '/static/khaos-ui.css',
  '/static/khaos-ui.js',
  '/static/khaos-fl-pwa.js',
  '/static/vendor/ort.min.js',
  '/static/vendor/ort-wasm-simd-threaded.wasm',
  '/static/vendor/ort-wasm-simd-threaded.mjs',
  '/static/favicon.ico',
  '/static/logo.png',
  '/static/logo.webp',
  '/static/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS).catch((err) => {
        console.warn('[ServiceWorker] Some assets could not be pre-cached:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Do not intercept non-GET requests or mutation endpoints
  if (event.request.method !== 'GET') {
    return;
  }

  // Cache-first strategy for immutable ONNX backbone models and vendor wasm
  if (url.pathname.startsWith('/federate/backbone.onnx') || url.pathname.startsWith('/static/vendor/')) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        const cached = await cache.match(event.request);
        if (cached) {
          return cached;
        }
        try {
          const response = await fetch(event.request);
          if (response && response.status === 200) {
            cache.put(event.request, response.clone());
          }
          return response;
        } catch (err) {
          if (cached) return cached;
          throw err;
        }
      })
    );
    return;
  }

  // Network-first strategy for HTML pages and APIs, falling back to cache
  if (event.request.mode === 'navigate' || url.pathname === '/expert') {
    event.respondWith(
      fetch(event.request).catch(async () => {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(event.request);
        return cached || cache.match('/expert');
      })
    );
    return;
  }

  // Network-first strategy for application scripts and styles so updates deploy immediately
  if (url.pathname.endsWith('.js') || url.pathname.endsWith('.css')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response && response.status === 200) {
            const respClone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, respClone));
          }
          return response;
        })
        .catch(async () => {
          const cache = await caches.open(CACHE_NAME);
          const cached = await cache.match(event.request);
          if (cached) return cached;
          throw new Error('Offline and asset not cached');
        })
    );
    return;
  }

  // Cache-first strategy for other static assets (images, fonts, manifest)
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (response && response.status === 200) {
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, response.clone()));
          }
          return response;
        });
      })
    );
  }
});
