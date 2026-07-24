(function () {
  var root = document.documentElement;
  var btn = document.getElementById("nav-sticky-toggle");
  if (!btn) return;

  var iconPinned = btn.querySelector(".nav-sticky-icon-pinned");
  var iconUnpinned = btn.querySelector(".nav-sticky-icon-unpinned");

  function isSticky() {
    return localStorage.getItem("nav-sticky") !== "false";
  }

  function updateNavHeight() {
    var nav = document.querySelector(".site-nav");
    if (!nav || !isSticky()) {
      root.style.removeProperty("--site-nav-height");
      document.body.style.paddingTop = "";
      return;
    }

    var height = nav.offsetHeight;
    root.style.setProperty("--site-nav-height", height + "px");
    document.body.style.paddingTop = height + "px";
  }

  function apply(sticky) {
    root.setAttribute("data-nav-sticky", sticky ? "true" : "false");
    iconPinned.classList.toggle("d-none", !sticky);
    iconUnpinned.classList.toggle("d-none", sticky);
    btn.title = sticky
      ? "Navbar pinned (click to unpin)"
      : "Navbar scrolls (click to pin)";
    btn.setAttribute("aria-pressed", sticky ? "true" : "false");
    updateNavHeight();
  }

  apply(isSticky());

  btn.addEventListener("click", function () {
    var next = !isSticky();
    if (next) {
      localStorage.removeItem("nav-sticky");
    } else {
      localStorage.setItem("nav-sticky", "false");
    }
    apply(next);
  });

  window.addEventListener("resize", updateNavHeight);

  var navbar = document.getElementById("navbar");
  if (navbar) {
    navbar.addEventListener("shown.bs.collapse", updateNavHeight);
    navbar.addEventListener("hidden.bs.collapse", updateNavHeight);
  }

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(updateNavHeight);
  }
})();
