import { useState } from "react";

export function SettingsModal({
  initialBaseUrl,
  onClose,
  onSave
}: {
  readonly initialBaseUrl: string;
  readonly onClose: () => void;
  readonly onSave: (connection: { readonly baseUrl: string; readonly authToken: string }) => void;
}) {
  const [baseUrl, setBaseUrl] = useState(initialBaseUrl);
  const [authToken, setAuthToken] = useState("");

  return (
    <div className="modal-backdrop" role="presentation">
      <section
        aria-label="Connection settings"
        className="settings-modal"
        role="dialog"
      >
        <h2>Connection settings</h2>
        <p className="modal-subtitle">Demo credentials stay in this session.</p>
        <label>
          Base API URL
          <input
            value={baseUrl}
            onChange={(event) => setBaseUrl(event.currentTarget.value)}
          />
        </label>
        <label>
          Auth token
          <input
            placeholder="Bearer token"
            type="password"
            value={authToken}
            onChange={(event) => setAuthToken(event.currentTarget.value)}
          />
        </label>
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="save"
            type="button"
            onClick={() => onSave({ authToken, baseUrl })}
          >
            Save
          </button>
        </div>
      </section>
    </div>
  );
}
