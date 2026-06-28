import type { ThemeMode } from "../viewTypes";

export function TopBar({
  themeMode,
  onToggleTheme
}: {
  readonly themeMode: ThemeMode;
  readonly onToggleTheme: () => void;
}) {
  return (
    <header className="topbar">
      <div className="brand">
        <img alt="" src="/fervis-logo-sm.png" />
        <span>Fervis</span>
      </div>
      <div className="topbar-meta">desktop alpha</div>
      <button
        aria-label={`Switch theme from ${themeMode}`}
        className="theme-toggle"
        title={`Theme: ${themeMode}`}
        type="button"
        onClick={onToggleTheme}
      >
        {themeIcon(themeMode)}
      </button>
    </header>
  );
}

function themeIcon(mode: ThemeMode): string {
  if (mode === "dark") {
    return "☾";
  }
  if (mode === "light") {
    return "☼";
  }
  return "◐";
}

