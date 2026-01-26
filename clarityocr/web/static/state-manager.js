/* =============================================================================
   State Manager - durable UI state (localStorage)
   ============================================================================= */

const StateManager = {
  KEYS: {
    THEME: "ocr-theme",
    PIPELINE: "ocr-pipeline",
    SETTINGS: "ocr-settings",
    SESSION: "ocr-session",
    TRANSITION: "ocr-transition",
  },

  getJSON(key, fallback = null) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return fallback;
      return JSON.parse(raw);
    } catch {
      return fallback;
    }
  },

  setJSON(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {}
  },

  remove(key) {
    try {
      localStorage.removeItem(key);
    } catch {}
  },

  setTransition(data) {
    const payload = {
      ...data,
      timestamp: Date.now(),
    };
    this.setJSON(this.KEYS.TRANSITION, payload);
  },

  consumeTransition(maxAgeMs = 5 * 60 * 1000) {
    const data = this.getJSON(this.KEYS.TRANSITION, null);
    this.remove(this.KEYS.TRANSITION);
    if (!data || !data.timestamp) return null;
    if (Date.now() - data.timestamp > maxAgeMs) return null;
    return data;
  },
};
