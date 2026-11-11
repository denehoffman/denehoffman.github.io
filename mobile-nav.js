(() => {
  "use strict";

  const siteNav = document.querySelector("#site-nav");
  const navList = siteNav?.querySelector("nav > ul");
  const homeItem = navList?.querySelector("#home");
  if (!siteNav || !navList || !homeItem) return;

  if (!navList.id) navList.id = "site-nav-links";

  const toggleItem = document.createElement("li");
  toggleItem.id = "mobile-nav-toggle";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "circle";
  toggle.setAttribute("aria-controls", navList.id);
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-label", "Open navigation");
  toggle.innerHTML = '<span class="mobile-nav-icon" aria-hidden="true"></span>';
  toggleItem.append(toggle);
  homeItem.after(toggleItem);
  siteNav.classList.add("mobile-nav-ready");

  function setOpen(open) {
    siteNav.classList.toggle("mobile-nav-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  }

  toggle.addEventListener("click", () => {
    setOpen(!siteNav.classList.contains("mobile-nav-open"));
  });

  navList.addEventListener("click", (event) => {
    if (event.target.closest("a")) setOpen(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setOpen(false);
  });

  const mobileQuery = window.matchMedia("(max-width: 600px)");
  mobileQuery.addEventListener("change", (event) => {
    if (!event.matches) setOpen(false);
  });
})();
