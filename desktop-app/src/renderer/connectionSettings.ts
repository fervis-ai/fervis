export const DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1";

const STORAGE_KEY = "fervis.desktop.connection.v1";

export interface ConnectionSettings {
  readonly baseUrl: string;
}

export function loadConnectionSettings(
  storage: Storage = window.localStorage
): ConnectionSettings {
  try {
    const rawValue = storage.getItem(STORAGE_KEY);
    if (rawValue === null) {
      return defaultConnectionSettings();
    }
    return decodeConnectionSettings(JSON.parse(rawValue));
  } catch {
    return defaultConnectionSettings();
  }
}

export function saveConnectionSettings(
  settings: ConnectionSettings,
  storage: Storage = window.localStorage
): void {
  const normalized = normalizeConnectionSettings(settings);
  storage.setItem(STORAGE_KEY, JSON.stringify(normalized));
}

export function normalizeConnectionSettings(
  settings: ConnectionSettings
): ConnectionSettings {
  const baseUrl = settings.baseUrl.trim();
  return {
    baseUrl: baseUrl === "" ? DEFAULT_BASE_URL : baseUrl
  };
}

function decodeConnectionSettings(payload: unknown): ConnectionSettings {
  if (
    typeof payload === "object" &&
    payload !== null &&
    "baseUrl" in payload &&
    typeof payload.baseUrl === "string"
  ) {
    return normalizeConnectionSettings({ baseUrl: payload.baseUrl });
  }
  return defaultConnectionSettings();
}

function defaultConnectionSettings(): ConnectionSettings {
  return { baseUrl: DEFAULT_BASE_URL };
}
