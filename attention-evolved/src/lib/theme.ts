export type ThemeMode = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "theme";

export function getStoredThemeMode(): ThemeMode {
  if (typeof window === "undefined") return "system";
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "light" || stored === "dark" ? stored : "system";
}

export function resolveTheme(mode: ThemeMode): ResolvedTheme {
  if (mode === "light" || mode === "dark") return mode;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(mode: ThemeMode) {
  const resolved = resolveTheme(mode);
  document.documentElement.setAttribute("data-theme", resolved);
  document.documentElement.setAttribute("data-theme-mode", mode);
}

export function cycleThemeMode(current: ThemeMode): ThemeMode {
  if (current === "system") return "light";
  if (current === "light") return "dark";
  return "system";
}

export function persistThemeMode(mode: ThemeMode) {
  if (mode === "system") {
    localStorage.removeItem(STORAGE_KEY);
  } else {
    localStorage.setItem(STORAGE_KEY, mode);
  }
}

export function themeToggleTitle(mode: ThemeMode): string {
  if (mode === "system") return "Theme: System (click for light)";
  if (mode === "light") return "Theme: Light (click for dark)";
  return "Theme: Dark (click for system)";
}
