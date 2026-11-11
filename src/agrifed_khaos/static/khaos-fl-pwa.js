/**
 * AgriFed-KHAOS Edge Federated Learning Engine (PWA Cross-Device)
 * 
 * Provides 100% Zero Data Leakage federated learning on edge devices (Mobile / Desktop):
 * 1. Local image storage in IndexedDB (0 bytes sent to server).
 * 2. On-device feature extraction via onnxruntime-web (WebAssembly / WebGPU).
 * 3. Local classification head training with FedProx proximal regularization + RMS noise.
 * 4. Dual on-device inference (Global vs Local comparison + OOD energy scoring).
 * 5. Secure parameter submission (canonical projection, relative norm checks, HMAC).
 */

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KhaosEdgeFL = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const DB_NAME = 'khaos_edge_fl_db';
  const DB_VERSION = 2;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];

  // Configure onnxruntime-web WASM binary path and threading
  if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths = '/static/vendor/';
    // Only configure multi-threading if crossOriginIsolated is active (SharedArrayBuffer available)
    if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
      ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 2, 4);
    } else {
      ort.env.wasm.numThreads = 1;
    }
  }

  // =========================================================================
  // 1. INDEXEDDB PERSISTENCE LAYER
  // =========================================================================
  class DBManager {
    constructor() {
      this.db = null;
    }

    async getDB() {
      if (this.db) return this.db;
      return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains('local_samples')) {
            const sStore = db.createObjectStore('local_samples', { keyPath: 'id' });
            sStore.createIndex('model_slug', 'model_slug', { unique: false });
            sStore.createIndex('class_compound', ['model_slug', 'className'], { unique: false });
          }
          if (!db.objectStoreNames.contains('cached_embeddings')) {
            const eStore = db.createObjectStore('cached_embeddings', { keyPath: 'id' });
            eStore.createIndex('sample_and_base', ['sample_id', 'base_model_sha256'], { unique: false });
            eStore.createIndex('model_slug', 'model_slug', { unique: false });
          }
          if (!db.objectStoreNames.contains('model_states')) {
            db.createObjectStore('model_states', { keyPath: 'model_slug' });
          }
          if (!db.objectStoreNames.contains('local_model_versions')) {
            const vStore = db.createObjectStore('local_model_versions', { keyPath: 'version_id' });
            vStore.createIndex('model_slug', 'model_slug', { unique: false });
            vStore.createIndex('created_at', 'created_at', { unique: false });
          }
        };
        req.onsuccess = (e) => {
          this.db = e.target.result;
          resolve(this.db);
        };
        req.onerror = (e) => reject(new Error('IndexedDB error: ' + e.target.error));
      });
    }

    async saveSample(sample) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_samples', 'readwrite');
        tx.objectStore('local_samples').put(sample);
        tx.oncomplete = () => resolve(sample.id);
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async getSamplesForModel(modelSlug) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_samples', 'readonly');
        const idx = tx.objectStore('local_samples').index('model_slug');
        const req = idx.getAll(modelSlug);
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = (e) => reject(e.target.error);
      });
    }

    async getSample(sampleId) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_samples', 'readonly');
        const req = tx.objectStore('local_samples').get(sampleId);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = (e) => reject(e.target.error);
      });
    }

    async deleteSample(sampleId) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(['local_samples', 'cached_embeddings'], 'readwrite');
        tx.objectStore('local_samples').delete(sampleId);
        // Note: Clean cached embeddings with sample_id prefix
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async clearSamplesForClass(modelSlug, className) {
      const samples = await this.getSamplesForModel(modelSlug);
      const toDelete = samples.filter((s) => s.className === className);
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_samples', 'readwrite');
        const store = tx.objectStore('local_samples');
        toDelete.forEach((s) => store.delete(s.id));
        tx.oncomplete = () => resolve(toDelete.length);
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async updateSampleClass(sampleId, newClassName) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_samples', 'readwrite');
        const store = tx.objectStore('local_samples');
        const req = store.get(sampleId);
        req.onsuccess = () => {
          const item = req.result;
          if (!item) return resolve(false);
          item.className = newClassName;
          store.put(item);
          resolve(true);
        };
        tx.oncomplete = () => resolve(true);
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async clearAllSamplesForModel(modelSlug) {
      const db = await this.getDB();
      const samples = await this.getSamplesForModel(modelSlug);
      return new Promise((resolve, reject) => {
        const tx = db.transaction(['local_samples', 'cached_embeddings'], 'readwrite');
        const sStore = tx.objectStore('local_samples');
        samples.forEach((s) => sStore.delete(s.id));

        const eStore = tx.objectStore('cached_embeddings');
        const eIdx = eStore.index('model_slug');
        const eReq = eIdx.getAllKeys(modelSlug);
        eReq.onsuccess = () => {
          (eReq.result || []).forEach((k) => eStore.delete(k));
        };
        tx.oncomplete = () => resolve(samples.length);
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async saveEmbedding(item) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('cached_embeddings', 'readwrite');
        tx.objectStore('cached_embeddings').put(item);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async getCachedEmbedding(sampleId, baseSha) {
      const db = await this.getDB();
      const id = `${sampleId}::${baseSha}`;
      return new Promise((resolve, reject) => {
        const tx = db.transaction('cached_embeddings', 'readonly');
        const req = tx.objectStore('cached_embeddings').get(id);
        req.onsuccess = () => resolve(req.result ? req.result.embedding : null);
        req.onerror = (e) => reject(e.target.error);
      });
    }

    async saveModelState(state) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('model_states', 'readwrite');
        tx.objectStore('model_states').put(state);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async getModelState(modelSlug) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('model_states', 'readonly');
        const req = tx.objectStore('model_states').get(modelSlug);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = (e) => reject(e.target.error);
      });
    }

    async saveVersion(versionObj) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_model_versions', 'readwrite');
        tx.objectStore('local_model_versions').put(versionObj);
        tx.oncomplete = () => resolve(versionObj.version_id);
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async getVersionsForModel(modelSlug) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_model_versions', 'readonly');
        const idx = tx.objectStore('local_model_versions').index('model_slug');
        const req = idx.getAll(modelSlug);
        req.onsuccess = () => {
          const versions = req.result || [];
          versions.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
          resolve(versions);
        };
        req.onerror = (e) => reject(e.target.error);
      });
    }

    async getVersion(versionId) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_model_versions', 'readonly');
        const req = tx.objectStore('local_model_versions').get(versionId);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = (e) => reject(e.target.error);
      });
    }

    async deleteVersion(versionId) {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('local_model_versions', 'readwrite');
        tx.objectStore('local_model_versions').delete(versionId);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
      });
    }

    async purgeModelData(modelSlug) {
      const db = await this.getDB();
      const samples = await this.getSamplesForModel(modelSlug);
      const versions = await this.getVersionsForModel(modelSlug);
      return new Promise((resolve, reject) => {
        const tx = db.transaction(['local_samples', 'cached_embeddings', 'local_model_versions', 'model_states'], 'readwrite');
        const sStore = tx.objectStore('local_samples');
        const eStore = tx.objectStore('cached_embeddings');
        const vStore = tx.objectStore('local_model_versions');
        const mStore = tx.objectStore('model_states');

        samples.forEach((s) => sStore.delete(s.id));
        versions.forEach((v) => vStore.delete(v.version_id));
        mStore.delete(modelSlug);

        try {
          const eIdx = eStore.index('model_slug');
          const eReq = eIdx.getAllKeys(modelSlug);
          eReq.onsuccess = () => {
            (eReq.result || []).forEach((k) => eStore.delete(k));
          };
        } catch (_) {}

        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
      });
    }
  }

  const dbManager = new DBManager();

  // =========================================================================
  // 2. IMAGE PREPROCESSING & CANVAS NORMALIZATION (IMAGENET EXACT MATCH)
  // =========================================================================
  class Preprocessor {
    static async blobToImage(blob) {
      return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          URL.revokeObjectURL(url);
          resolve(img);
        };
        img.onerror = () => {
          URL.revokeObjectURL(url);
          reject(new Error('Failed to decode image'));
        };
        img.src = url;
      });
    }

    static async createThumbnail(imageSource, maxDim = 96) {
      const img = imageSource instanceof Blob ? await this.blobToImage(imageSource) : imageSource;
      let w = img.naturalWidth || img.width;
      let h = img.naturalHeight || img.height;
      if (w > h) {
        if (w > maxDim) {
          h = Math.round((h * maxDim) / w);
          w = maxDim;
        }
      } else {
        if (h > maxDim) {
          w = Math.round((w * maxDim) / h);
          h = maxDim;
        }
      }
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      return canvas.toDataURL('image/jpeg', 0.85);
    }

    static async processImageToTensor(imageSource, targetSize = 224, shortestEdge = 256) {
      const img = imageSource instanceof Blob ? await this.blobToImage(imageSource) : imageSource;
      const origW = img.naturalWidth || img.width;
      const origH = img.naturalHeight || img.height;

      // 1. Maintain aspect ratio so shortest edge equals 256
      let scale = shortestEdge / Math.min(origW, origH);
      let resizedW = Math.round(origW * scale);
      let resizedH = Math.round(origH * scale);

      // 2. Center crop 224x224
      let cropX = Math.max(0, Math.floor((resizedW - targetSize) / 2));
      let cropY = Math.max(0, Math.floor((resizedH - targetSize) / 2));

      let canvas;
      if (typeof OffscreenCanvas !== 'undefined') {
        canvas = new OffscreenCanvas(targetSize, targetSize);
      } else {
        canvas = document.createElement('canvas');
        canvas.width = targetSize;
        canvas.height = targetSize;
      }
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // Draw resized image with crop offset
      ctx.drawImage(img, -cropX, -cropY, resizedW, resizedH);
      const imgData = ctx.getImageData(0, 0, targetSize, targetSize).data;

      // 3. Planar NCHW Float32Array projection: [1, 3, 224, 224]
      const totalPixels = targetSize * targetSize;
      const float32Data = new Float32Array(3 * totalPixels);

      const mean = IMAGENET_MEAN;
      const std = IMAGENET_STD;

      for (let i = 0; i < totalPixels; i++) {
        const r = imgData[i * 4] / 255.0;
        const g = imgData[i * 4 + 1] / 255.0;
        const b = imgData[i * 4 + 2] / 255.0;

        float32Data[i] = (r - mean[0]) / std[0]; // Channel R
        float32Data[totalPixels + i] = (g - mean[1]) / std[1]; // Channel G
        float32Data[2 * totalPixels + i] = (b - mean[2]) / std[2]; // Channel B
      }

      return float32Data;
    }
  }

  // =========================================================================
  // 3. ONNX RUNTIME BACKBONE SESSION & FEATURE EXTRACTOR
  // =========================================================================
  class FeatureExtractor {
    constructor() {
      this.sessions = new Map(); // modelSlug -> ort.InferenceSession
    }

    async getSession(modelSlug, onnxUrl) {
      // Sessions are keyed by slug but validated against the backbone URL (which embeds the base
      // SHA-256). If the server model changed, the stale backbone must be released so embeddings
      // are never computed with an old backbone against a new head.
      const cached = this.sessions.get(modelSlug);
      if (cached && cached.url === onnxUrl) {
        return cached.session;
      }
      if (cached) {
        this.sessions.delete(modelSlug);
        try {
          await cached.session.release();
        } catch (_) {}
      }
      if (typeof ort === 'undefined') {
        throw new Error('onnxruntime-web (ort) is not loaded. Check network or vendor scripts.');
      }

      if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
        if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) {
          ort.env.wasm.numThreads = 1;
        }
      }

      const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      };

      try {
        const session = await ort.InferenceSession.create(onnxUrl, options);
        this.sessions.set(modelSlug, { url: onnxUrl, session });
        return session;
      } catch (err) {
        console.error('[FeatureExtractor] Failed to instantiate ORT session:', err);
        throw err;
      }
    }

    async extractEmbedding(modelSlug, onnxUrl, tensorNCHW) {
      const session = await this.getSession(modelSlug, onnxUrl);
      const tensor = new ort.Tensor('float32', tensorNCHW, [1, 3, 224, 224]);
      let output = null;
      try {
        output = await session.run({ input: tensor });
        const embTensor = output.embedding;
        if (!embTensor) {
          throw new Error('Expected output "embedding" not found in ONNX backbone output');
        }
        // Copy out of WASM-backed memory before disposing the tensors.
        return Array.from(embTensor.data);
      } finally {
        // Explicit release keeps WASM heap flat on mobile (iOS Safari / Android Chrome tab limits).
        try {
          tensor.dispose();
        } catch (_) {}
        if (output) {
          for (const k of Object.keys(output)) {
            try {
              output[k].dispose();
            } catch (_) {}
          }
        }
      }
    }
  }

  const featureExtractor = new FeatureExtractor();

  // =========================================================================
  // 4. ON-DEVICE CLASSIFIER OPTIMIZER (FEDPROX + RMS NOISE)
  // =========================================================================
  class EdgeOptimizer {
    static softmax(logits, temperature = 1.0) {
      const C = logits.length;
      let maxVal = -Infinity;
      for (let c = 0; c < C; c++) {
        if (logits[c] > maxVal) maxVal = logits[c];
      }
      let sumExp = 0.0;
      const exps = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        exps[c] = Math.exp((logits[c] - maxVal) / temperature);
        sumExp += exps[c];
      }
      const probs = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        probs[c] = exps[c] / (sumExp + 1e-12);
      }
      return probs;
    }

    static computeOODEnergy(logits, temperature = 1.0) {
      let maxVal = -Infinity;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxVal) maxVal = logits[i];
      }
      let sumExp = 0.0;
      for (let i = 0; i < logits.length; i++) {
        sumExp += Math.exp((logits[i] - maxVal) / temperature);
      }
      return -temperature * (maxVal + Math.log(sumExp + 1e-12));
    }

    static generateGaussianNoise(length) {
      const arr = new Float32Array(length);
      const uintArr = new Uint32Array(2);
      for (let i = 0; i < length; i += 2) {
        window.crypto.getRandomValues(uintArr);
        const u1 = (uintArr[0] + 1.0) / 4294967297.0;
        const u2 = (uintArr[1] + 1.0) / 4294967297.0;
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
        arr[i] = z0;
        if (i + 1 < length) arr[i + 1] = z1;
      }
      return arr;
    }

    /**
     * Train classification head using Softmax CrossEntropy + FedProx proximal loss.
     */
    static async trainHead({
      dataset, // Array of { embedding: number[], label: number }
      baseWeights, // { weight: number[][], bias: number[] } (canonical base weights = FedProx anchor w_t)
      canonicalClasses, // string[]
      activeClasses, // string[] (may include local user-created classes)
      embeddingDim,
      initWeights = null, // optional warm start { weight: (number[]|null)[], bias: (number|null)[] } aligned to activeClasses
      maxRelativeDelta = 0.2, // client-side delta clipping; prevents drastic divergence
      epochs = 5,
      lr = 0.001,
      fedproxMu = 0.2,
      noiseSigma = 0.0,
      batchSize = 16,
    }) {
      const numActive = activeClasses.length;
      const numCanonical = canonicalClasses.length;
      const D = embeddingDim;
      const N = dataset.length;

      if (N === 0) {
        throw new Error('No local samples provided for on-device training.');
      }

      // Initialize trainable parameters
      // Matrix W: [numActive, D], Bias b: [numActive]
      const W = new Float32Array(numActive * D);
      const b = new Float32Array(numActive);

      // Copy base weights for canonical classes
      for (let c = 0; c < numCanonical; c++) {
        b[c] = baseWeights.bias[c] || 0.0;
        const baseRow = baseWeights.weight[c] || [];
        for (let d = 0; d < D; d++) {
          W[c * D + d] = baseRow[d] || 0.0;
        }
      }

      // Initialize extra classes with small random values
      if (numActive > numCanonical) {
        const extraNoise = this.generateGaussianNoise((numActive - numCanonical) * D);
        const scale = Math.sqrt(2.0 / D);
        for (let c = numCanonical; c < numActive; c++) {
          b[c] = 0.0;
          for (let d = 0; d < D; d++) {
            W[c * D + d] = extraNoise[(c - numCanonical) * D + d] * scale * 0.1;
          }
        }
      }

      // Warm start: continue from previously learned local weights so that earlier sessions
      // (whose photos may have been auto-cleared) are not discarded. The FedProx anchor stays w_t.
      if (initWeights && initWeights.weight && initWeights.bias) {
        for (let c = 0; c < numActive; c++) {
          const row = initWeights.weight[c];
          if (row && row.length === D && Number.isFinite(initWeights.bias[c])) {
            b[c] = initWeights.bias[c];
            for (let d = 0; d < D; d++) W[c * D + d] = row[d];
          }
        }
      }

      // Frozen reference for classes that have no local samples in this session
      const W0 = Float32Array.from(W);
      const b0 = Float32Array.from(b);
      const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

      // Class frequency and missing classes computation
      const classCounts = new Array(numActive).fill(0);
      dataset.forEach((item) => classCounts[item.label]++);

      const presentClasses = [];
      const missingClasses = [];
      for (let c = 0; c < numActive; c++) {
        if (classCounts[c] > 0) {
          presentClasses.push(c);
        } else {
          missingClasses.push(c);
        }
      }

      // Effective Number of Samples weighting (beta = 0.999)
      const beta = 0.999;
      const classWeights = new Float32Array(numActive).fill(1.0);
      let sumWeights = 0.0;
      presentClasses.forEach((c) => {
        const n = classCounts[c];
        const eff = (1.0 - beta) / (1.0 - Math.pow(beta, n) + 1e-12);
        classWeights[c] = eff;
        sumWeights += eff;
      });
      presentClasses.forEach((c) => {
        classWeights[c] = (classWeights[c] / (sumWeights + 1e-12)) * presentClasses.length;
      });

      // AdamW optimizer state buffers
      const mW = new Float32Array(numActive * D);
      const vW = new Float32Array(numActive * D);
      const mb = new Float32Array(numActive);
      const vb = new Float32Array(numActive);

      const beta1 = 0.9;
      const beta2 = 0.999;
      const eps = 1e-8;
      const labelSmoothing = 0.05;
      let optStep = 0;

      // Training loop over epochs
      let lastLoss = 0.0;
      for (let ep = 0; ep < epochs; ep++) {
        // Shuffle indices
        const indices = Array.from({ length: N }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        let epochLoss = 0.0;

        for (let start = 0; start < N; start += batchSize) {
          const batchIndices = indices.slice(start, start + batchSize);
          const B = batchIndices.length;

          // Gradient accumulators
          const gradW = new Float32Array(numActive * D);
          const gradb = new Float32Array(numActive);

          for (let bIdx = 0; bIdx < B; bIdx++) {
            const sample = dataset[batchIndices[bIdx]];
            const z = sample.embedding;
            const y = sample.label;
            const sampleWeight = classWeights[y] || 1.0;

            // Forward: logits = W * z + b
            const logits = new Float32Array(numActive);
            for (let c = 0; c < numActive; c++) {
              let dot = b[c];
              const rowOffset = c * D;
              for (let d = 0; d < D; d++) {
                dot += W[rowOffset + d] * z[d];
              }
              logits[c] = dot;
            }

            // Softmax
            const probs = this.softmax(logits);

            // Cross-Entropy Loss with Label Smoothing
            for (let c = 0; c < numActive; c++) {
              const targetP = (1.0 - labelSmoothing) * (c === y ? 1.0 : 0.0) + labelSmoothing / numActive;
              epochLoss -= targetP * Math.log(probs[c] + 1e-12) * sampleWeight;

              // Backpropagation gradient
              const g_c = sampleWeight * (probs[c] - targetP);
              gradb[c] += g_c / B;
              const rowOffset = c * D;
              for (let d = 0; d < D; d++) {
                gradW[rowOffset + d] += (g_c * z[d]) / B;
              }
            }
          }

          // FedProx proximal regularization term on canonical classes: mu * (W - W_base)
          for (let c = 0; c < numCanonical; c++) {
            const baseRow = baseWeights.weight[c] || [];
            const rowOffset = c * D;
            const deltaB = b[c] - (baseWeights.bias[c] || 0.0);
            gradb[c] += fedproxMu * deltaB;
            for (let d = 0; d < D; d++) {
              const deltaW = W[rowOffset + d] - (baseRow[d] || 0.0);
              gradW[rowOffset + d] += fedproxMu * deltaW;
            }
          }

          // Gradient Masking: zero out gradients for missing classes
          missingClasses.forEach((c) => {
            gradb[c] = 0.0;
            const rowOffset = c * D;
            for (let d = 0; d < D; d++) {
              gradW[rowOffset + d] = 0.0;
            }
          });

          // Global L2 Gradient Clipping (max_norm = 1.0)
          let gradNormSq = 0.0;
          for (let i = 0; i < gradW.length; i++) gradNormSq += gradW[i] * gradW[i];
          for (let i = 0; i < gradb.length; i++) gradNormSq += gradb[i] * gradb[i];
          const gradNorm = Math.sqrt(gradNormSq);
          if (gradNorm > 1.0) {
            const clipFactor = 1.0 / gradNorm;
            for (let i = 0; i < gradW.length; i++) gradW[i] *= clipFactor;
            for (let i = 0; i < gradb.length; i++) gradb[i] *= clipFactor;
          }

          // AdamW parameter update
          optStep++;
          const lrCorrection = lr * (Math.sqrt(1.0 - Math.pow(beta2, optStep)) / (1.0 - Math.pow(beta1, optStep)));

          for (let c = 0; c < numActive; c++) {
            // Bias update
            mb[c] = beta1 * mb[c] + (1.0 - beta1) * gradb[c];
            vb[c] = beta2 * vb[c] + (1.0 - beta2) * gradb[c] * gradb[c];
            b[c] -= (lrCorrection * mb[c]) / (Math.sqrt(vb[c]) + eps);

            // Weight update
            const rowOffset = c * D;
            for (let d = 0; d < D; d++) {
              const idx = rowOffset + d;
              mW[idx] = beta1 * mW[idx] + (1.0 - beta1) * gradW[idx];
              vW[idx] = beta2 * vW[idx] + (1.0 - beta2) * gradW[idx] * gradW[idx];
              W[idx] -= (lrCorrection * mW[idx]) / (Math.sqrt(vW[idx]) + eps);
            }
          }

          // Hold classes without local samples strictly at base weights (preserve pre-trained knowledge)
          missingClasses.forEach((c) => {
            if (c < numCanonical) {
              b[c] = baseWeights.bias[c] || 0.0;
              const baseRow = baseWeights.weight[c] || [];
              const rowOffset = c * D;
              for (let d = 0; d < D; d++) {
                W[rowOffset + d] = baseRow[d] || 0.0;
              }
            } else {
              b[c] = b0[c];
              const rowOffset = c * D;
              for (let d = 0; d < D; d++) {
                W[rowOffset + d] = W0[rowOffset + d];
              }
            }
          });

          // Keep the UI thread responsive on slow phones (progress bars, touch events)
          await yieldToUI();
        }
        lastLoss = epochLoss / N;
      }

      // Client-side delta clipping on canonical rows: bounds the update (model-poisoning surface and
      // server-side 422 rejections) *before* noise is added, i.e. clip-then-noise.
      {
        let dSq = 0.0;
        let bSq = 0.0;
        for (let c = 0; c < numCanonical; c++) {
          const baseRow = baseWeights.weight[c] || [];
          const dB = b[c] - (baseWeights.bias[c] || 0.0);
          dSq += dB * dB;
          bSq += (baseWeights.bias[c] || 0.0) ** 2;
          for (let d = 0; d < D; d++) {
            const bw = baseRow[d] || 0.0;
            const dW = W[c * D + d] - bw;
            dSq += dW * dW;
            bSq += bw * bw;
          }
        }
        const limit = maxRelativeDelta * Math.sqrt(bSq);
        const dNorm = Math.sqrt(dSq);
        if (limit > 0 && dNorm > limit) {
          const s = limit / dNorm;
          for (let c = 0; c < numCanonical; c++) {
            const baseRow = baseWeights.weight[c] || [];
            const bb = baseWeights.bias[c] || 0.0;
            b[c] = bb + s * (b[c] - bb);
            for (let d = 0; d < D; d++) {
              const bw = baseRow[d] || 0.0;
              W[c * D + d] = bw + s * (W[c * D + d] - bw);
            }
          }
        }
      }

      // Post-training Adaptive RMS noise injection (Differential Privacy protection)
      if (noiseSigma > 0.0) {
        // Compute RMS of W and b
        let sumWsq = 0.0;
        for (let i = 0; i < W.length; i++) sumWsq += W[i] * W[i];
        const rmsW = Math.max(Math.sqrt(sumWsq / W.length), 1e-4);

        let sumBsq = 0.0;
        for (let i = 0; i < b.length; i++) sumBsq += b[i] * b[i];
        const rmsB = Math.max(Math.sqrt(sumBsq / b.length), 1e-4);

        const noiseW = this.generateGaussianNoise(W.length);
        const noiseB = this.generateGaussianNoise(b.length);

        for (let i = 0; i < W.length; i++) {
          W[i] += noiseW[i] * noiseSigma * rmsW;
        }
        for (let i = 0; i < b.length; i++) {
          b[i] += noiseB[i] * noiseSigma * rmsB;
        }

        // Re-apply hold on missing classes (strictly keep canonical base weights)
        missingClasses.forEach((c) => {
          if (c < numCanonical) {
            b[c] = baseWeights.bias[c] || 0.0;
            const baseRow = baseWeights.weight[c] || [];
            const rowOffset = c * D;
            for (let d = 0; d < D; d++) {
              W[rowOffset + d] = baseRow[d] || 0.0;
            }
          } else {
            b[c] = b0[c];
            const rowOffset = c * D;
            for (let d = 0; d < D; d++) {
              W[rowOffset + d] = W0[rowOffset + d];
            }
          }
        });
      }

      // Calculate accuracy and divergence norms on canonical classes
      let correct = 0;
      for (let i = 0; i < N; i++) {
        const sample = dataset[i];
        const z = sample.embedding;
        let maxL = -Infinity;
        let pred = -1;
        for (let c = 0; c < numActive; c++) {
          let dot = b[c];
          const rowOffset = c * D;
          for (let d = 0; d < D; d++) dot += W[rowOffset + d] * z[d];
          if (dot > maxL) {
            maxL = dot;
            pred = c;
          }
        }
        if (pred === sample.label) correct++;
      }
      const accuracy = (correct / N) * 100.0;

      // Delta norm on canonical classes
      let deltaNormSq = 0.0;
      let baseNormSq = 0.0;
      for (let c = 0; c < numCanonical; c++) {
        const baseRow = baseWeights.weight[c] || [];
        const baseB = baseWeights.bias[c] || 0.0;
        const dB = b[c] - baseB;
        deltaNormSq += dB * dB;
        baseNormSq += baseB * baseB;

        const rowOffset = c * D;
        for (let d = 0; d < D; d++) {
          const dW = W[rowOffset + d] - (baseRow[d] || 0.0);
          deltaNormSq += dW * dW;
          baseNormSq += (baseRow[d] || 0.0) * (baseRow[d] || 0.0);
        }
      }
      const deltaNorm = Math.sqrt(deltaNormSq);
      const relativeNorm = deltaNorm / (Math.sqrt(baseNormSq) + 1e-12);

      // Pack back into nested arrays
      const packedW = [];
      for (let c = 0; c < numActive; c++) {
        const row = [];
        const rowOffset = c * D;
        for (let d = 0; d < D; d++) {
          row.push(W[rowOffset + d]);
        }
        packedW.push(row);
      }
      const packedB = Array.from(b);

      return {
        weights: {
          weight: packedW,
          bias: packedB,
        },
        canonical_projection: {
          weight: packedW.slice(0, numCanonical),
          bias: packedB.slice(0, numCanonical),
        },
        missing_classes: missingClasses.filter((c) => c < numCanonical),
        present_classes: presentClasses,
        stats: {
          epochs,
          num_samples: N,
          final_loss: Math.round(lastLoss * 10000) / 10000,
          accuracy: Math.round(accuracy * 100) / 100,
          delta_norm: Math.round(deltaNorm * 10000) / 10000,
          relative_norm: Math.round(relativeNorm * 10000) / 10000,
        },
      };
    }
  }

  // =========================================================================
  // 5. MAIN EDGE FEDERATED LEARNING ENGINE (KhaosEdgeFL)
  // =========================================================================
  class KhaosEdgeFLClient {
    constructor() {
      this.activeSlug = null;
      this.modelBundleCache = new Map();
    }

    /**
     * Download or load cached edge model bundle from server.
     */
    async ensureModelBundle(modelSlug) {
      if (this.modelBundleCache.has(modelSlug)) {
        return this.modelBundleCache.get(modelSlug);
      }

      // Check IndexedDB state
      const savedState = await dbManager.getModelState(modelSlug);
      let res;
      try {
        res = await fetch(`/federate/model-bundle?model_slug=${encodeURIComponent(modelSlug)}`);
      } catch (err) {
        // If offline and we have saved state, return offline bundle
        if (savedState) {
          console.warn('[KhaosEdgeFL] Operating offline using saved model state:', modelSlug);
          return savedState;
        }
        throw new Error('Could not connect to server and no local model bundle is cached: ' + err.message);
      }

      if (!res.ok) {
        if (savedState) return savedState;
        throw new Error(`Failed to fetch model bundle (${res.status}): ${await res.text()}`);
      }

      const bundle = await res.json();

      // Check if server hash differs from the lineage the local head was *trained* against.
      // (Previously only LOCAL_DIRTY was flagged, so a FEDERATED_PENDING/SYNCED head trained on an
      // old base could silently be re-labelled with the new base hash and bypass the 409 check.)
      let syncState = 'SYNCED_CLEAN';
      const trainedBase = savedState ? savedState.trained_base_sha256 || (savedState.local_weights ? savedState.base_model_sha256 : null) : null;
      const lineageStale = Boolean(savedState && savedState.local_weights && trainedBase && trainedBase !== bundle.base_model_sha256);
      if (savedState && savedState.base_model_sha256 !== bundle.base_model_sha256 && savedState.is_pinned) {
        syncState = 'PINNED_ISOLATED';
      } else if (lineageStale) {
        syncState = savedState.is_pinned ? 'PINNED_ISOLATED' : 'CONFLICT_DIRTY_OUTDATED';
      } else if (savedState && savedState.is_pinned) {
        syncState = 'PINNED_ISOLATED';
      } else if (savedState && savedState.sync_state) {
        syncState = savedState.sync_state;
      }

      const stateObj = {
        model_slug: modelSlug,
        base_model_sha256: bundle.base_model_sha256,
        trained_base_sha256: lineageStale || (savedState && savedState.local_weights) ? trainedBase : null,
        cumulative_samples: savedState ? savedState.cumulative_samples || 0 : 0,
        missing_classes: savedState ? savedState.missing_classes || [] : [],
        architecture: bundle.architecture,
        classes: bundle.classes,
        num_classes: bundle.num_classes,
        embedding_dim: bundle.embedding_dim,
        head_weights: bundle.head_weights,
        onnx_download_url: bundle.onnx_download_url,
        preprocessing: bundle.preprocessing,
        local_weights: savedState ? savedState.local_weights : null,
        training_stats: savedState ? savedState.training_stats : null,
        sync_state: syncState,
        is_pinned: savedState ? Boolean(savedState.is_pinned) : false,
        active_version_id: savedState ? savedState.active_version_id : null,
        version_tag: savedState ? savedState.version_tag : null,
        custom_name: savedState ? savedState.custom_name : null,
        updated_at: new Date().toISOString(),
      };

      await dbManager.saveModelState(stateObj);
      this.modelBundleCache.set(modelSlug, stateObj);
      return stateObj;
    }

    /**
     * Add new photo samples into IndexedDB for a given class (Zero Data Leakage).
     */
    async addLocalSamples(modelSlug, className, fileList) {
      if (!fileList || fileList.length === 0) return 0;
      let count = 0;

      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        if (!file.type.startsWith('image/')) continue;

        const thumb = await Preprocessor.createThumbnail(file);
        const sampleId = `${modelSlug}::${className}::${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;

        const sample = {
          id: sampleId,
          model_slug: modelSlug,
          className: className,
          filename: file.name || `photo_${Date.now()}.jpg`,
          blob: file,
          thumbnailDataUrl: thumb,
          created_at: new Date().toISOString(),
        };

        await dbManager.saveSample(sample);
        count++;
      }
      return count;
    }

    /**
     * Quick summary of local activity for a model (used by "Mis Modelos" hub tab).
     */
    async getLocalSummary(modelSlug) {
      try {
        const samples = await dbManager.getSamplesForModel(modelSlug);
        const state = await dbManager.getModelState(modelSlug);
        const versions = await dbManager.getVersionsForModel(modelSlug);
        const hasTrained = Boolean(state && state.local_weights);
        const sampleCount = samples ? samples.length : 0;
        const versionCount = versions ? versions.length : 0;
        const isAdopted = Boolean(state && state.adopted_at);
        const isActive = sampleCount > 0 || hasTrained || versionCount > 0 || isAdopted;
        return {
          isActive,
          sampleCount,
          hasTrained,
          versionCount,
          activeVersion: (state && state.version_tag) || (hasTrained ? 'v1.0-local' : null),
          accuracy: (state && state.training_stats && state.training_stats.accuracy) || null,
          syncState: (state && state.sync_state) || 'SYNCED_CLEAN',
        };
      } catch (e) {
        return { isActive: false, sampleCount: 0, hasTrained: false, versionCount: 0 };
      }
    }

    /**
     * Explicitly adopt a server model into the expert's local workspace.
     */
    async adoptModel(modelSlug) {
      let state = await dbManager.getModelState(modelSlug);
      if (!state) {
        try {
          const bundle = await this.ensureModelBundle(modelSlug);
          state = {
            model_slug: modelSlug,
            adopted_at: new Date().toISOString(),
            base_model_sha256: bundle ? bundle.base_model_sha256 : '',
            sync_state: 'SYNCED_CLEAN',
          };
          await dbManager.saveModelState(state);
        } catch (bundleErr) {
          console.warn(`[KhaosEdgeFL] Bundle not available for '${modelSlug}', saving minimal state:`, bundleErr.message);
          state = {
            model_slug: modelSlug,
            adopted_at: new Date().toISOString(),
            base_model_sha256: '',
            sync_state: 'SYNCED_CLEAN',
          };
          await dbManager.saveModelState(state);
        }
      } else if (!state.adopted_at) {
        state.adopted_at = new Date().toISOString();
        await dbManager.saveModelState(state);
      }
      return state;
    }

    /**
     * Get all local workspace classes, counts, and sample thumbnails for UI rendering.
     */
    async getLocalWorkspace(modelSlug) {
      const bundle = await this.ensureModelBundle(modelSlug);
      const samples = await dbManager.getSamplesForModel(modelSlug);
      const state = await dbManager.getModelState(modelSlug);
      const savedVersions = await this.listVersions(modelSlug);

      // Collect all classes (canonical + any user created local folders)
      const classMap = new Map();
      bundle.classes.forEach((c) => {
        classMap.set(c, { name: c, count: 0, samples: [] });
      });

      // Populate samples
      samples.forEach((s) => {
        if (!classMap.has(s.className)) {
          classMap.set(s.className, { name: s.className, count: 0, samples: [] });
        }
        const item = classMap.get(s.className);
        item.count++;
        item.samples.push({
          id: s.id,
          filename: s.filename,
          url: s.thumbnailDataUrl,
          fullUrl: s.blob ? URL.createObjectURL(s.blob) : s.thumbnailDataUrl,
        });
      });

      const classesList = Array.from(classMap.values());
      const totalSamples = samples.length;
      const activeVer = savedVersions.find((v) => v.is_active) || null;

      return {
        model_slug: modelSlug,
        architecture: bundle.architecture,
        base_model_sha256: bundle.base_model_sha256,
        classes: classesList,
        total_samples: totalSamples,
        is_pinned: Boolean(state && state.is_pinned),
        active_version_id: state ? state.active_version_id : null,
        saved_versions: savedVersions,
        local_model: state && state.local_weights ? {
          exists: true,
          created_at: state.updated_at,
          num_samples: state.training_stats ? state.training_stats.num_samples : totalSamples,
          final_accuracy: state.training_stats ? state.training_stats.accuracy : null,
          final_loss: state.training_stats ? state.training_stats.final_loss : null,
          delta_norm: state.training_stats ? state.training_stats.delta_norm : null,
          sync_state: state.sync_state,
          is_pinned: Boolean(state.is_pinned),
          custom_name: state.custom_name || (activeVer ? activeVer.custom_name : null),
          version_tag: state.version_tag || (activeVer ? activeVer.version_tag : null),
          download_url: activeVer ? activeVer.download_pt_url : null,
        } : { exists: false },
        sync_state: state ? state.sync_state : 'SYNCED_CLEAN',
      };
    }

    /**
     * Delete a single local sample by ID.
     */
    async deleteLocalSample(sampleId) {
      return dbManager.deleteSample(sampleId);
    }

    /**
     * Move a local sample to a new class folder.
     */
    async moveLocalSample(modelSlug, sampleId, newClassName, filename) {
      await dbManager.updateSampleClass(sampleId, newClassName);
      try {
        const params = new URLSearchParams({
          to_class: newClassName,
          filename: filename || sampleId,
        });
        await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/move-class-sample?${params.toString()}`, {
          method: 'POST',
        });
      } catch (e) {
        console.warn('[PWA] Server sync for move-sample skipped:', e);
      }
      return true;
    }

    /**
     * Clear all local samples for a model to free device storage while keeping retrained weights intact.
     */
    async clearLocalSamples(modelSlug) {
      const count = await dbManager.clearAllSamplesForModel(modelSlug);
      try {
        await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/clear-samples`, {
          method: 'DELETE',
        });
      } catch (e) {
        console.warn('[PWA] Server sync for clear-samples skipped:', e);
      }
      return count;
    }

    /**
     * Clear all samples for a class.
     */
    async clearClassSamples(modelSlug, className) {
      return dbManager.clearSamplesForClass(modelSlug, className);
    }

    /**
     * Extract embeddings incrementally for all local samples, caching in IndexedDB.
     */
    async extractLocalEmbeddings(modelSlug, onProgress) {
      const bundle = await this.ensureModelBundle(modelSlug);
      const samples = await dbManager.getSamplesForModel(modelSlug);
      const total = samples.length;
      if (total === 0) return [];

      const dataset = [];
      const onnxUrl = bundle.onnx_download_url;
      const baseSha = bundle.base_model_sha256;

      // Case-insensitive canonical class mapping
      const canonicalMap = new Map();
      bundle.classes.forEach((c) => canonicalMap.set(c.toLowerCase(), c));

      // Map class name to active class index
      const activeClasses = Array.from(new Set([
        ...bundle.classes,
        ...samples.map((s) => {
          const lower = (s.className || '').trim().toLowerCase();
          return canonicalMap.has(lower) ? canonicalMap.get(lower) : s.className;
        }),
      ]));

      for (let i = 0; i < total; i++) {
        const s = samples[i];
        const lower = (s.className || '').trim().toLowerCase();
        const effectiveClass = canonicalMap.has(lower) ? canonicalMap.get(lower) : s.className;
        if (onProgress) {
          onProgress({ current: i + 1, total, stage: 'feature_extraction' });
        }

        let emb = await dbManager.getCachedEmbedding(s.id, baseSha);
        if (!emb) {
          const tensorNCHW = await Preprocessor.processImageToTensor(s.blob, 224, 256);
          emb = await featureExtractor.extractEmbedding(modelSlug, onnxUrl, tensorNCHW);
          await dbManager.saveEmbedding({
            id: `${s.id}::${baseSha}`,
            sample_id: s.id,
            model_slug: modelSlug,
            base_model_sha256: baseSha,
            className: effectiveClass,
            embedding: emb,
          });
        }

        const labelIdx = activeClasses.indexOf(effectiveClass);
        dataset.push({ embedding: emb, label: labelIdx, sampleId: s.id });
      }

      return { dataset, activeClasses, bundle };
    }

    /**
     * BOTÓN 1: Reentrenar Modelo en Local (FedProx on Head, Zero Data Leakage).
     */
    async trainLocalHead(modelSlug, options = {}, onProgress) {
      const t0 = performance.now();
      const { dataset, activeClasses, bundle } = await this.extractLocalEmbeddings(modelSlug, onProgress);

      if (dataset.length === 0) {
        throw new Error('Añade al menos 1 o más fotos a las carpetas para reentrenar tu modelo local.');
      }

      if (onProgress) {
        onProgress({ stage: 'training', message: 'Optimizando cabeza clasificadora con FedProx...' });
      }

      // Warm start only if the previous local head was trained against the *current* base lineage.
      const prevState = await dbManager.getModelState(modelSlug);
      let initWeights = null;
      let cumulativeSamples = 0;
      const prevTrainedBase = prevState && (prevState.trained_base_sha256 || prevState.base_model_sha256);
      if (
        prevState &&
        prevState.local_weights &&
        prevState.local_weights.classes &&
        prevTrainedBase === bundle.base_model_sha256
      ) {
        const prevIdx = new Map(prevState.local_weights.classes.map((n, i) => [n, i]));
        initWeights = {
          weight: activeClasses.map((n) => (prevIdx.has(n) ? prevState.local_weights.weight[prevIdx.get(n)] : null)),
          bias: activeClasses.map((n) => (prevIdx.has(n) ? prevState.local_weights.bias[prevIdx.get(n)] : null)),
        };
        cumulativeSamples = prevState.cumulative_samples || 0;
      }

      const trainResult = await EdgeOptimizer.trainHead({
        dataset,
        baseWeights: bundle.head_weights,
        canonicalClasses: bundle.classes,
        activeClasses: activeClasses,
        embeddingDim: bundle.embedding_dim,
        initWeights,
        epochs: options.epochs || 5,
        lr: options.lr || (dataset.length <= 5 ? 0.0008 : 0.001),
        fedproxMu: options.fedproxMu !== undefined ? options.fedproxMu : 0.2,
        noiseSigma: options.noiseSigma !== undefined ? options.noiseSigma : 0.0,
        maxRelativeDelta: options.maxRelativeDelta || 0.2,
        batchSize: options.batchSize || 16,
      });
      cumulativeSamples += dataset.length;
      trainResult.stats.cumulative_samples = cumulativeSamples;

      const elapsedSec = Math.round((performance.now() - t0) / 10) / 100;
      trainResult.stats.training_time_seconds = elapsedSec;

      // Save local weights in IndexedDB
      const stateObj = {
        model_slug: modelSlug,
        base_model_sha256: bundle.base_model_sha256,
        trained_base_sha256: bundle.base_model_sha256, // lineage the local head was actually trained against
        cumulative_samples: cumulativeSamples,
        architecture: bundle.architecture,
        classes: bundle.classes,
        num_classes: bundle.num_classes,
        embedding_dim: bundle.embedding_dim,
        head_weights: bundle.head_weights,
        onnx_download_url: bundle.onnx_download_url,
        preprocessing: bundle.preprocessing,
        local_weights: {
          weight: trainResult.weights.weight,
          bias: trainResult.weights.bias,
          classes: activeClasses,
        },
        training_stats: trainResult.stats,
        missing_classes: trainResult.missing_classes,
        sync_state: 'LOCAL_DIRTY',
        updated_at: new Date().toISOString(),
      };

      await dbManager.saveModelState(stateObj);
      this.modelBundleCache.set(modelSlug, stateObj);

      return {
        success: true,
        stats: trainResult.stats,
        activeClasses,
        training_time_seconds: elapsedSec,
      };
    }

    /**
     * BOTÓN 2: Usar y Probar Modelo (Comparativa Global vs Local 100% Offline).
     */
    async runDualInference(modelSlug, imageFileOrBlob) {
      const bundle = await this.ensureModelBundle(modelSlug);
      const state = await dbManager.getModelState(modelSlug);

      const tensorNCHW = await Preprocessor.processImageToTensor(imageFileOrBlob, 224, 256);
      const z = await featureExtractor.extractEmbedding(modelSlug, bundle.onnx_download_url, tensorNCHW);

      const D = bundle.embedding_dim;

      // 1. Global Model Inference
      const globalClasses = bundle.classes;
      const baseW = bundle.head_weights.weight;
      const baseB = bundle.head_weights.bias;
      const globalLogits = new Float32Array(globalClasses.length);

      for (let c = 0; c < globalClasses.length; c++) {
        let dot = baseB[c] || 0.0;
        const row = baseW[c] || [];
        for (let d = 0; d < D; d++) dot += (row[d] || 0.0) * z[d];
        globalLogits[c] = dot;
      }
      const globalProbs = EdgeOptimizer.softmax(globalLogits);
      const globalOOD = EdgeOptimizer.computeOODEnergy(globalLogits);

      let globalTopIdx = 0;
      let globalMaxP = -1;
      for (let c = 0; c < globalClasses.length; c++) {
        if (globalProbs[c] > globalMaxP) {
          globalMaxP = globalProbs[c];
          globalTopIdx = c;
        }
      }

      const globalResult = {
        predicted_class: globalClasses[globalTopIdx],
        confidence: Math.round(globalMaxP * 1000) / 1000,
        energy_score: Math.round(globalOOD * 100) / 100,
        all_probabilities: globalClasses.map((cName, idx) => ({
          class_name: cName,
          probability: Math.round(globalProbs[idx] * 1000) / 1000,
        })).sort((a, b) => b.probability - a.probability),
      };

      // 2. Local Model Inference (if exists)
      const hasLocalModel = Boolean(state && state.local_weights);
      let localResult = null;
      let verdictChanged = false;
      let confDelta = 0.0;

      if (hasLocalModel) {
        const localW = state.local_weights.weight;
        const localB = state.local_weights.bias;
        const localClasses = state.local_weights.classes || globalClasses;
        const localLogits = new Float32Array(localClasses.length);

        for (let c = 0; c < localClasses.length; c++) {
          let dot = localB[c] || 0.0;
          const row = localW[c] || [];
          for (let d = 0; d < D; d++) dot += (row[d] || 0.0) * z[d];
          localLogits[c] = dot;
        }

        const localProbs = EdgeOptimizer.softmax(localLogits);
        const localOOD = EdgeOptimizer.computeOODEnergy(localLogits);

        let localTopIdx = 0;
        let localMaxP = -1;
        for (let c = 0; c < localClasses.length; c++) {
          if (localProbs[c] > localMaxP) {
            localMaxP = localProbs[c];
            localTopIdx = c;
          }
        }

        localResult = {
          predicted_class: localClasses[localTopIdx],
          confidence: Math.round(localMaxP * 1000) / 1000,
          energy_score: Math.round(localOOD * 100) / 100,
          all_probabilities: localClasses.map((cName, idx) => ({
            class_name: cName,
            probability: Math.round(localProbs[idx] * 1000) / 1000,
          })).sort((a, b) => b.probability - a.probability),
        };

        verdictChanged = localResult.predicted_class !== globalResult.predicted_class;
        confDelta = localResult.confidence - globalResult.confidence;
      } else {
        // Before local retraining, the local model is identical to the base checkpoint.
        // Providing the baseline result prevents "Sin predicción" in UI diagnostic sandboxes.
        localResult = {
          predicted_class: globalResult.predicted_class,
          confidence: globalResult.confidence,
          energy_score: globalResult.energy_score,
          all_probabilities: [...globalResult.all_probabilities],
          is_base_baseline: true,
        };
        verdictChanged = false;
        confDelta = 0.0;
      }

      return {
        has_local_model: hasLocalModel,
        verdict_changed: verdictChanged,
        confidence_delta: Math.round(confDelta * 1000) / 1000,
        global_model: {
          version: 'Global (Servidor)',
          result: globalResult,
        },
        local_model: {
          version: hasLocalModel ? 'Local (Reentrenado en tu Dispositivo)' : 'Local (Base Inicial)',
          result: localResult,
        },
      };
    }

    /**
     * Retrieve the original Blob for a stored local sample by ID.
     */
    async getLocalSampleBlob(sampleId) {
      const s = await dbManager.getSample(sampleId);
      return s ? s.blob : null;
    }

    /**
     * Run on-device benchmark evaluation on official test samples.
     * Evaluates Base vs Local Adapted Model completely in the browser via WASM/ONNX.
     * ZERO DATA LEAKAGE: Weights and inference never leave the user's device.
     */
    async runOnDeviceBenchmark(modelSlug, onProgress) {
      await this.ensureModelBundle(modelSlug);
      const state = await dbManager.getModelState(modelSlug);
      const hasLocalModel = Boolean(state && state.local_weights);

      // 1. Fetch benchmark test samples from server
      const manifestRes = await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/benchmark-test-samples`);
      if (!manifestRes.ok) {
        throw new Error('No se pudieron obtener las muestras de test del servidor');
      }
      const manifest = await manifestRes.json();
      const samples = manifest.samples || [];

      if (samples.length === 0) {
        return {
          success: false,
          message: 'No hay muestras de prueba disponibles en el servidor para este modelo.',
          total_samples: 0,
          samples: [],
        };
      }

      const evaluatedSamples = [];
      let baseCorrect = 0;
      let localCorrect = 0;
      let stable = 0;
      let improved = 0;
      let regressed = 0;
      let bothWrong = 0;

      for (let i = 0; i < samples.length; i++) {
        const s = samples[i];
        if (onProgress) {
          onProgress({ current: i + 1, total: samples.length, sample: s });
        }

        try {
          const imgRes = await fetch(s.image_url);
          if (!imgRes.ok) throw new Error('Error al descargar muestra de test');
          const blob = await imgRes.blob();

          const dual = await this.runDualInference(modelSlug, blob);
          const gRes = dual.global_model.result;
          const lRes = (dual.local_model && dual.local_model.result) ? dual.local_model.result : gRes;

          const trueCls = (s.true_class || '').toLowerCase();
          const basePred = s.base_pred || s.base_prediction || gRes.predicted_class || '';
          const baseConf = s.base_conf !== undefined ? s.base_conf : (s.base_confidence !== undefined ? s.base_confidence : (Math.round((gRes.confidence || 0) * 1000) / 10));
          const isBaseOk = s.base_ok !== undefined ? Boolean(s.base_ok) : (basePred.toLowerCase() === trueCls);

          const localPred = (hasLocalModel && lRes && lRes.predicted_class) ? lRes.predicted_class : basePred;
          const localConf = (hasLocalModel && lRes && lRes.confidence !== undefined) ? (Math.round(lRes.confidence * 1000) / 10) : baseConf;
          const isLocalOk = hasLocalModel ? ((localPred || '').toLowerCase() === trueCls) : isBaseOk;

          if (isBaseOk) baseCorrect++;
          if (isLocalOk) localCorrect++;

          let status = 'BOTH_WRONG';
          if (!hasLocalModel) {
            status = isBaseOk ? 'STABLE' : 'BOTH_WRONG';
            if (isBaseOk) stable++;
            else bothWrong++;
          } else {
            if (isBaseOk && isLocalOk) {
              status = 'STABLE';
              stable++;
            } else if (!isBaseOk && isLocalOk) {
              status = 'IMPROVED';
              improved++;
            } else if (isBaseOk && !isLocalOk) {
              status = 'REGRESSED';
              regressed++;
            } else {
              bothWrong++;
            }
          }

          evaluatedSamples.push({
            id: s.id,
            image_url: s.image_url,
            true_class: s.true_class,
            base_pred: basePred,
            base_prediction: basePred,
            base_conf: baseConf,
            base_confidence: baseConf,
            base_ok: isBaseOk,
            base_correct: isBaseOk,
            local_pred: localPred,
            local_prediction: localPred,
            local_conf: localConf,
            local_confidence: localConf,
            local_ok: isLocalOk,
            local_correct: isLocalOk,
            status: status,
            verdict_changed: (localPred || '').toLowerCase() !== (basePred || '').toLowerCase(),
          });
        } catch (itemErr) {
          console.warn('[PWA] Error evaluating on-device sample:', s.id, itemErr);
        }
      }

      const total = evaluatedSamples.length;
      const baseAcc = (manifest.base_accuracy !== undefined && manifest.base_accuracy !== null)
        ? manifest.base_accuracy
        : (total > 0 ? Math.round((baseCorrect / total) * 1000) / 10 : 0.0);
      let localAcc = total > 0 ? Math.round((localCorrect / total) * 1000) / 10 : 0.0;
      let retention = baseCorrect > 0 ? Math.round((stable / baseCorrect) * 1000) / 10 : 100.0;
      let deltaAcc = Math.round((localAcc - baseAcc) * 10) / 10;

      if (!hasLocalModel) {
        localAcc = baseAcc;
        deltaAcc = 0.0;
        retention = 100.0;
      }

      // Compute MATLAB plotconfusion style confusion matrices locally (Zero Data Leakage)
      const allLabels = (manifest.classes && manifest.classes.length > 0)
        ? manifest.classes
        : ((this.activeBundle && this.activeBundle.classes) || []);

      const yTrue = evaluatedSamples.map(s => s.true_class);
      const yBase = evaluatedSamples.map(s => s.base_pred);
      const yLocal = evaluatedSamples.map(s => s.local_pred);

      const cmBase = this.computeConfusionMatrixData(allLabels, yTrue, yBase);
      const cmLocal = this.computeConfusionMatrixData(allLabels, yTrue, yLocal);

      return {
        success: true,
        on_device: true,
        slug: modelSlug,
        total_samples: total,
        base_accuracy: baseAcc,
        local_accuracy: localAcc,
        delta_accuracy: deltaAcc,
        retention_rate: retention,
        summary: {
          stable: stable,
          improved: improved,
          regressed: regressed,
          both_wrong: bothWrong,
        },
        samples: evaluatedSamples,
        confusion_matrices: {
          labels: cmBase.labels,
          base: cmBase,
          local: cmLocal,
        },
      };
    }

    /**
     * Compute MATLAB-style confusion matrix and classification metrics directly in browser memory.
     */
    computeConfusionMatrixData(labels, yTrue, yPred) {
      if (!labels || labels.length === 0) {
        labels = Array.from(new Set([...yTrue, ...yPred])).sort();
      }
      const n = labels.length;
      const labelMap = new Map();
      labels.forEach((lbl, idx) => labelMap.set(lbl.toLowerCase(), idx));

      // Transposed matrix [predicted_idx][true_idx] (MATLAB plotconfusion style)
      const matrix = Array.from({ length: n }, () => Array(n).fill(0));

      for (let i = 0; i < yTrue.length; i++) {
        const tCls = (yTrue[i] || '').toLowerCase();
        const pCls = (yPred[i] || '').toLowerCase();
        const tIdx = labelMap.get(tCls);
        const pIdx = labelMap.get(pCls);
        if (tIdx !== undefined && pIdx !== undefined) {
          matrix[pIdx][tIdx]++;
        }
      }

      const sumLin = [];
      const sumCol = Array(n).fill(0);
      for (let r = 0; r < n; r++) {
        let rowSum = 0;
        for (let c = 0; c < n; c++) {
          rowSum += matrix[r][c];
          sumCol[c] += matrix[r][c];
        }
        sumLin.push(rowSum);
      }
      const totAll = sumCol.reduce((a, b) => a + b, 0);

      // User Accuracy (Precision per row: hit / sumLin)
      const userAccuracy = [];
      const userError = [];
      for (let r = 0; r < n; r++) {
        const hit = matrix[r][r];
        const tot = sumLin[r];
        const acc = tot > 0 ? Math.round((hit / tot) * 1000) / 10 : 0.0;
        userAccuracy.push(acc);
        userError.push(tot > 0 ? Math.round((100.0 - acc) * 10) / 10 : 0.0);
      }

      // Producer Accuracy (Recall per col: hit / sumCol)
      const producerAccuracy = [];
      const producerError = [];
      for (let c = 0; c < n; c++) {
        const hit = matrix[c][c];
        const tot = sumCol[c];
        const acc = tot > 0 ? Math.round((hit / tot) * 1000) / 10 : 0.0;
        producerAccuracy.push(acc);
        producerError.push(tot > 0 ? Math.round((100.0 - acc) * 10) / 10 : 0.0);
      }

      let diagHits = 0;
      for (let i = 0; i < n; i++) diagHits += matrix[i][i];
      const overallAccuracy = totAll > 0 ? Math.round((diagHits / totAll) * 1000) / 10 : 0.0;
      const overallError = totAll > 0 ? Math.round((100.0 - overallAccuracy) * 10) / 10 : 0.0;

      const classMetrics = labels.map((lbl, idx) => {
        const prec = userAccuracy[idx];
        const rec = producerAccuracy[idx];
        const f1 = (prec + rec > 0) ? Math.round(((2 * prec * rec) / (prec + rec)) * 10) / 10 : 0.0;
        return {
          class_name: lbl,
          precision: prec,
          recall: rec,
          f1_score: f1,
          support: sumCol[idx],
        };
      });

      return {
        labels: labels,
        matrix: matrix,
        sum_lin: sumLin,
        sum_col: sumCol,
        tot_all: totAll,
        user_accuracy: userAccuracy,
        user_error: userError,
        producer_accuracy: producerAccuracy,
        producer_error: producerError,
        overall_accuracy: overallAccuracy,
        overall_error: overallError,
        class_metrics: classMetrics,
      };
    }

    /**
     * Compute SHA-256 hex digest using Web Crypto API.
     */
    async sha256Bytes(buffer) {
      const digest = await window.crypto.subtle.digest('SHA-256', buffer);
      return Array.from(new Uint8Array(digest))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    }

    /**
     * Compute HMAC-SHA256 hex signature using Web Crypto API.
     */
    async computeHmacSha256(keyStr, messageStr) {
      const enc = new TextEncoder();
      const key = await window.crypto.subtle.importKey(
        'raw',
        enc.encode(keyStr),
        { name: 'HMAC', hash: 'SHA-256' },
        false,
        ['sign']
      );
      const signature = await window.crypto.subtle.sign('HMAC', key, enc.encode(messageStr));
      return Array.from(new Uint8Array(signature))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    }

    /**
     * BOTÓN 3: Federar al Servidor Central (Canonical Projection + HMAC + Zero Leakage).
     */
    async submitFederatedUpdate(modelSlug, clientId = 'web_client_01', options = {}) {
      const state = await dbManager.getModelState(modelSlug);
      if (!state || !state.local_weights) {
        throw new Error('Primero debes entrenar tu modelo local pulsando "1. Reentrenar" antes de federar.');
      }

      const bundle = await this.ensureModelBundle(modelSlug);
      const numCanonical = bundle.classes.length;
      const canonicalClasses = bundle.classes;

      // Lineage guard: the head must be submitted against the base it was trained on, never against
      // whatever base the server currently serves.
      const trainedBase = state.trained_base_sha256 || state.base_model_sha256;
      if (trainedBase !== bundle.base_model_sha256) {
        state.sync_state = 'CONFLICT_DIRTY_OUTDATED';
        await dbManager.saveModelState(state);
        throw new Error('CONFLICT_409: Tu modelo local se entrenó sobre una versión anterior del modelo global. Sincroniza antes de aportar.');
      }

      // Project weights to canonical base classes (Anti-Ontology Pollution)
      const fullW = state.local_weights.weight;
      const fullB = state.local_weights.bias;
      const fullClasses = state.local_weights.classes || canonicalClasses;

      const canonW = [];
      const canonB = [];
      for (let c = 0; c < numCanonical; c++) {
        const cName = canonicalClasses[c];
        const localIdx = fullClasses.indexOf(cName);
        if (localIdx !== -1) {
          canonW.push(fullW[localIdx]);
          canonB.push(fullB[localIdx]);
        } else {
          canonW.push(bundle.head_weights.weight[c]);
          canonB.push(bundle.head_weights.bias[c]);
        }
      }

      // Convert canonical weights to binary Float32Array buffer for deterministic SHA256
      const flatFloats = [];
      canonW.forEach((row) => row.forEach((v) => flatFloats.push(v)));
      canonB.forEach((v) => flatFloats.push(v));
      const f32Buf = new Float32Array(flatFloats).buffer;
      const payloadSha256 = await this.sha256Bytes(f32Buf);

      const numSamples = Math.max(1, Math.round(state.cumulative_samples || (state.training_stats && state.training_stats.num_samples) || 1));
      const missingClasses = state.missing_classes || [];

      // Optional HMAC computation if server requires HMAC key
      let hmacSig = null;
      if (options.hmacKey) {
        const manifestStr = `sha256=${payloadSha256}|base_sha256=${bundle.base_model_sha256}|scope=head_only|client_id=${clientId}|num_samples=${numSamples}`;
        hmacSig = await this.computeHmacSha256(options.hmacKey, manifestStr);
      }

      const payload = {
        client_id: clientId,
        model_slug: modelSlug,
        base_model_sha256: bundle.base_model_sha256,
        training_scope: 'head_only',
        num_samples: numSamples,
        classes: canonicalClasses,
        missing_classes: missingClasses,
        head_tensors: {
          weight: canonW,
          bias: canonB,
        },
        training_stats: state.training_stats || {},
        payload_sha256: payloadSha256,
        hmac_signature: hmacSig,
      };

      const headers = { 'Content-Type': 'application/json' };
      if (options.authToken) {
        headers['Authorization'] = `Bearer ${options.authToken}`;
      }

      const res = await fetch('/federate/submit-update', {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      if (res.status === 409) {
        // Lineage conflict: Global model updated on server!
        state.sync_state = 'CONFLICT_DIRTY_OUTDATED';
        await dbManager.saveModelState(state);
        throw new Error('CONFLICT_409: La versión del modelo global ha cambiado en el servidor. Es necesario sincronizar.');
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Error al federar el modelo (${res.status})`);
      }

      const resData = await res.json();

      // Mark model state as federated pending
      state.sync_state = 'FEDERATED_PENDING';
      await dbManager.saveModelState(state);

      return resData;
    }

    /**
     * Resolve synchronization conflict when server updates global model.
     * Choice: 'readapt' (download new global base and auto-retrain with local photos) or 'adopt_global'
     */
    async resolveSyncConflict(modelSlug, choice, onProgress) {
      if (choice === 'adopt_global') {
        // Clear local weights and adopt fresh bundle
        this.modelBundleCache.delete(modelSlug);
        const bundle = await this.ensureModelBundle(modelSlug);
        const state = await dbManager.getModelState(modelSlug);
        state.local_weights = null;
        state.training_stats = null;
        state.trained_base_sha256 = null;
        state.cumulative_samples = 0;
        state.missing_classes = [];
        state.sync_state = 'SYNCED_CLEAN';
        await dbManager.saveModelState(state);
        this.modelBundleCache.delete(modelSlug);
        return { success: true, message: 'Modelo global adoptado.' };
      } else if (choice === 'readapt') {
        // Download new global bundle and re-run trainLocalHead
        this.modelBundleCache.delete(modelSlug);
        await this.ensureModelBundle(modelSlug);
        const retrainRes = await this.trainLocalHead(modelSlug, {}, onProgress);
        return { success: true, message: 'Modelo re-adaptado con éxito sobre la nueva versión global.', stats: retrainRes.stats };
      }
    }

    /**
     * Freeze and save current local model weights as a personal version snapshot.
     */
    async saveVersionSnapshot(modelSlug, customName, description = '', isPinned = false, options = {}) {
      if (options && typeof options.receive_updates === 'boolean') {
        isPinned = !options.receive_updates;
      }
      const state = await dbManager.getModelState(modelSlug);
      if (!state || !state.local_weights) {
        throw new Error('No hay ningún modelo entrenado en local para guardar. Primero pulsa "1. Reentrenar".');
      }

      const existingVersions = await dbManager.getVersionsForModel(modelSlug);
      const nextIndex = existingVersions.length + 1;
      const versionTag = `v1.${nextIndex}-local`;
      const versionId = `v_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      const nowIso = new Date().toISOString();

      // Deactivate previous versions in local store
      for (const v of existingVersions) {
        if (v.is_active) {
          v.is_active = false;
          await dbManager.saveVersion(v);
        }
      }

      const headWeightsClean = {
        weight: state.local_weights.weight,
        bias: state.local_weights.bias,
      };
      const classesList = state.local_weights.classes || state.classes || [];

      const snapshot = {
        version_id: versionId,
        model_slug: modelSlug,
        version_tag: versionTag,
        custom_name: customName || `Versión ${versionTag} (${(state.training_stats && state.training_stats.num_samples) || 0} fotos)`,
        description: description || '',
        is_pinned: Boolean(isPinned),
        receive_updates: !Boolean(isPinned),
        is_active: true,
        base_model_sha256: state.base_model_sha256 || null,
        base_version: state.base_version || 'v1',
        num_samples: (state.training_stats && state.training_stats.num_samples) || 0,
        accuracy: (state.training_stats && state.training_stats.accuracy) || 0.0,
        final_loss: (state.training_stats && state.training_stats.final_loss) || 0.0,
        delta_norm: (state.training_stats && state.training_stats.delta_norm) || 0.0,
        head_weights: headWeightsClean,
        classes: classesList,
        created_at: nowIso,
        download_pt_url: null,
      };

      // Save locally in IndexedDB
      await dbManager.saveVersion(snapshot);

      // Update active model state in IndexedDB
      state.active_version_id = versionId;
      state.version_tag = versionTag;
      state.custom_name = snapshot.custom_name;
      state.is_pinned = Boolean(isPinned);
      state.receive_updates = !Boolean(isPinned);
      if (isPinned) {
        state.sync_state = 'PINNED_ISOLATED';
      }
      await dbManager.saveModelState(state);

      // Backup snapshot to server profile
      try {
        const serverPayload = {
          custom_name: snapshot.custom_name,
          description: snapshot.description,
          version_tag: snapshot.version_tag,
          base_model_sha256: snapshot.base_model_sha256,
          num_samples: snapshot.num_samples,
          accuracy: snapshot.accuracy,
          final_loss: snapshot.final_loss,
          delta_norm: snapshot.delta_norm,
          classes: snapshot.classes,
          head_weights: snapshot.head_weights,
          is_pinned: snapshot.is_pinned,
        };
        const res = await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(serverPayload),
        });
        if (res.ok) {
          const sData = await res.json();
          snapshot.server_version_id = sData.version_id;
          snapshot.download_pt_url = `/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions/${sData.version_id}/download-pt`;
          await dbManager.saveVersion(snapshot);
        }
      } catch (syncErr) {
        console.warn('[PWA] Server sync for saved version failed (offline mode active):', syncErr);
      }

      return snapshot;
    }

    /**
     * List all saved versions for this model (local IndexedDB merged with server profile).
     */
    async listVersions(modelSlug) {
      const localVersions = await dbManager.getVersionsForModel(modelSlug);
      let serverVersions = [];
      try {
        const res = await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions`);
        if (res.ok) {
          serverVersions = await res.json();
        }
      } catch (err) {
        // Offline or unauthenticated
      }

      const vMap = new Map();
      localVersions.forEach((lv) => {
        vMap.set(lv.version_id, lv);
      });

      serverVersions.forEach((sv) => {
        const svId = sv.version_id;
        const dlUrl = `/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions/${svId}/download-pt`;
        if (!vMap.has(svId)) {
          vMap.set(svId, {
            version_id: svId,
            model_slug: sv.model_slug || modelSlug,
            version_tag: sv.version_tag,
            custom_name: sv.custom_name,
            description: sv.description || '',
            is_pinned: Boolean(sv.is_pinned),
            receive_updates: !Boolean(sv.is_pinned),
            is_active: Boolean(sv.is_active),
            base_model_sha256: sv.base_model_sha256,
            num_samples: sv.num_samples || 0,
            accuracy: sv.accuracy || 0.0,
            final_loss: sv.final_loss || 0.0,
            delta_norm: sv.delta_norm || 0.0,
            head_weights: sv.head_weights,
            classes: sv.classes || [],
            created_at: sv.created_at,
            server_version_id: svId,
            download_pt_url: dlUrl,
            original_model_status: sv.original_model_status || 'available',
            original_model_diff: sv.original_model_diff || 'synced',
            original_model_active_version: sv.original_model_active_version || 'v1',
          });
        } else {
          const lv = vMap.get(svId);
          lv.server_version_id = svId;
          lv.download_pt_url = dlUrl;
          if (sv.is_pinned !== undefined) {
            lv.is_pinned = Boolean(sv.is_pinned);
            lv.receive_updates = !Boolean(sv.is_pinned);
          }
          if (sv.is_active !== undefined) lv.is_active = Boolean(sv.is_active);
          lv.original_model_status = sv.original_model_status || lv.original_model_status || 'available';
          lv.original_model_diff = sv.original_model_diff || lv.original_model_diff || 'synced';
          lv.original_model_active_version = sv.original_model_active_version || lv.original_model_active_version || 'v1';
        }
      });

      const merged = Array.from(vMap.values());
      merged.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      return merged;
    }

    /**
     * Activate a saved version, replacing active model weights with this snapshot.
     */
    async activateVersion(modelSlug, versionId) {
      let ver = await dbManager.getVersion(versionId);
      if (!ver) {
        try {
          const res = await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions`);
          if (res.ok) {
            const list = await res.json();
            ver = list.find((v) => v.version_id === versionId);
          }
        } catch (e) {}
      }
      if (!ver) {
        throw new Error(`Versión "${versionId}" no encontrada.`);
      }

      // Update IndexedDB states
      const allVersions = await dbManager.getVersionsForModel(modelSlug);
      for (const v of allVersions) {
        v.is_active = (v.version_id === versionId);
        await dbManager.saveVersion(v);
      }

      const state = await dbManager.getModelState(modelSlug);
      if (state) {
        state.local_weights = ver.head_weights;
        // A restored version keeps the lineage it was trained on, not the current server base.
        state.trained_base_sha256 = ver.base_model_sha256 || state.trained_base_sha256 || state.base_model_sha256;
        state.cumulative_samples = ver.num_samples || state.cumulative_samples || 0;
        state.training_stats = {
          num_samples: ver.num_samples,
          accuracy: ver.accuracy,
          final_loss: ver.final_loss,
          delta_norm: ver.delta_norm,
        };
        state.active_version_id = versionId;
        state.version_tag = ver.version_tag;
        state.custom_name = ver.custom_name;
        state.is_pinned = Boolean(ver.is_pinned);
        state.sync_state = ver.is_pinned ? 'PINNED_ISOLATED' : 'LOCAL_DIRTY';
        state.updated_at = new Date().toISOString();
        await dbManager.saveModelState(state);
      }

      // Sync active state to server
      try {
        const targetId = ver.server_version_id || versionId;
        await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions/${encodeURIComponent(targetId)}/activate`, {
          method: 'POST',
        });
      } catch (e) {
        console.warn('[PWA] Failed to sync version activation to server:', e);
      }

      return { success: true, version: ver };
    }

    /**
     * Toggle the pinned state of an expert version to prevent automatic global overwrites.
     */
    async togglePinVersion(modelSlug, versionId, isPinned) {
      const ver = await dbManager.getVersion(versionId);
      if (ver) {
        ver.is_pinned = Boolean(isPinned);
        await dbManager.saveVersion(ver);
      }

      const state = await dbManager.getModelState(modelSlug);
      if (state && state.active_version_id === versionId) {
        state.is_pinned = Boolean(isPinned);
        state.sync_state = isPinned ? 'PINNED_ISOLATED' : 'LOCAL_DIRTY';
        await dbManager.saveModelState(state);
      }

      // Sync to server
      try {
        const targetId = (ver && ver.server_version_id) || versionId;
        await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions/${encodeURIComponent(targetId)}/pin`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ is_pinned: Boolean(isPinned) }),
        });
      } catch (e) {
        console.warn('[PWA] Failed to sync pin toggle to server:', e);
      }

      return { success: true, is_pinned: Boolean(isPinned) };
    }

    /**
     * Delete a saved version snapshot from local store and server profile.
     */
    async deleteVersion(modelSlug, versionId) {
      const ver = await dbManager.getVersion(versionId);
      await dbManager.deleteVersion(versionId);

      const state = await dbManager.getModelState(modelSlug);
      if (state && state.active_version_id === versionId) {
        state.active_version_id = null;
        state.version_tag = null;
        state.custom_name = null;
        state.is_pinned = false;
        await dbManager.saveModelState(state);
      }

      // Sync to server
      try {
        const targetId = (ver && ver.server_version_id) || versionId;
        await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/saved-versions/${encodeURIComponent(targetId)}`, {
          method: 'DELETE',
        });
      } catch (e) {
        console.warn('[PWA] Failed to sync version deletion to server:', e);
      }

      return { success: true, deleted_id: versionId };
    }

    /**
     * Reset local model adaptations back to pristine base model state (0.0% delta).
     */
    async resetLocalModel(modelSlug) {
      const state = await dbManager.getModelState(modelSlug);
      if (state) {
        state.local_weights = null;
        state.training_stats = null;
        state.trained_base_sha256 = null;
        state.cumulative_samples = 0;
        state.missing_classes = [];
        state.sync_state = 'SYNCED_CLEAN';
        state.active_version_id = null;
        state.is_pinned = false;
        state.updated_at = new Date().toISOString();
        await dbManager.saveModelState(state);
      }
      this.modelBundleCache.delete(modelSlug);
      return { success: true };
    }

    /**
     * Delete model completely from local device (IndexedDB) and sync workspace deletion to server.
     */
    async deleteLocalModel(modelSlug) {
      await dbManager.purgeModelData(modelSlug);
      this.modelBundleCache.delete(modelSlug);

      // Sync server workspace deletion
      try {
        await fetch(`/api/expert/models/${encodeURIComponent(modelSlug)}/client-workspace`, {
          method: 'DELETE',
        });
      } catch (e) {
        console.warn('[PWA] Failed to delete server client workspace:', e);
      }

      return { success: true, model_slug: modelSlug };
    }
  }

  const client = new KhaosEdgeFLClient();
  // Test seam: pure numeric internals (no DOM/IndexedDB) for headless verification under Node.
  Object.defineProperty(client, '_internals', { value: { EdgeOptimizer }, enumerable: false });
  if (typeof window !== 'undefined') {
    window.KhaosEdgeFL = client;
  }
  return client;
});

