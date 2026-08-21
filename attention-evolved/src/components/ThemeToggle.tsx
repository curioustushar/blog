"use client";

import { useEffect, useState } from "react";
import {
  applyTheme,
  cycleThemeMode,
  getStoredThemeMode,
  persistThemeMode,
  themeToggleTitle,
  type ThemeMode,
} from "@/lib/theme";

export function ThemeToggle() {
  const [mode, setMode] = useState<ThemeMode>("system");

  useEffect(() => {
    const initial = getStoredThemeMode();
    setMode(initial);
    applyTheme(initial);

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      if (getStoredThemeMode() === "system") {
        applyTheme("system");
      }
    };

    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const handleClick = () => {
    const next = cycleThemeMode(mode);
    persistThemeMode(next);
    applyTheme(next);
    setMode(next);
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="theme-toggle inline-flex items-center justify-center w-9 h-9 rounded-lg border border-border bg-muted text-muted-foreground hover:text-accent hover:bg-accent-light transition-colors"
      aria-label="Toggle color theme"
      title={themeToggleTitle(mode)}
    >
      <span className={mode === "system" ? "inline-flex" : "hidden"} aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="2" y="3" width="20" height="14" rx="2" />
          <path d="M8 21h8M12 17v4" />
        </svg>
      </span>
      <span className={mode === "light" ? "inline-flex" : "hidden"} aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="5" />
          <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
        </svg>
      </span>
      <span className={mode === "dark" ? "inline-flex" : "hidden"} aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      </span>
    </button>
  );
}
