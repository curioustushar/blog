export function ThemeScript() {
  const script = `(function(){var s=localStorage.getItem("theme");var d=window.matchMedia("(prefers-color-scheme: dark)").matches;var t=s==="light"||s==="dark"?s:(d?"dark":"light");var m=s==="light"||s==="dark"?s:"system";document.documentElement.setAttribute("data-theme",t);document.documentElement.setAttribute("data-theme-mode",m);})();`;

  return <script dangerouslySetInnerHTML={{ __html: script }} />;
}
