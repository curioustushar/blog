(function () {
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (!btn) return;

  var icons = {
    system: btn.querySelector(".theme-icon-system"),
    sun: btn.querySelector(".theme-icon-sun"),
    moon: btn.querySelector(".theme-icon-moon"),
  };

  function resolvedTheme() {
    var stored = localStorage.getItem("theme");
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function mode() {
    var stored = localStorage.getItem("theme");
    return stored === "light" || stored === "dark" ? stored : "system";
  }

  function apply(theme, themeMode) {
    root.setAttribute("data-theme", theme);
    root.setAttribute("data-theme-mode", themeMode);
    icons.system.classList.add("d-none");
    icons.sun.classList.add("d-none");
    icons.moon.classList.add("d-none");
    if (themeMode === "system") {
      icons.system.classList.remove("d-none");
    } else if (theme === "light") {
      icons.sun.classList.remove("d-none");
    } else {
      icons.moon.classList.remove("d-none");
    }
    btn.title = themeMode === "system"
      ? "Theme: System (click for light)"
      : theme === "light"
        ? "Theme: Light (click for dark)"
        : "Theme: Dark (click for system)";
  }

  apply(resolvedTheme(), mode());

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    if (mode() === "system") apply(resolvedTheme(), "system");
  });

  btn.addEventListener("click", function () {
    var current = mode();
    var next;
    if (current === "system") {
      next = "light";
    } else if (current === "light") {
      next = "dark";
    } else {
      next = "system";
      localStorage.removeItem("theme");
      apply(resolvedTheme(), "system");
      return;
    }
    localStorage.setItem("theme", next);
    apply(next, next);
  });
})();
