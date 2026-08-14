(() => {
  const rail = document.querySelector(".publication-years");
  if (!rail) return;

  const links = [...rail.querySelectorAll("a[href^='#']")];
  const sections = links
    .map((link) => document.getElementById(link.hash.slice(1)))
    .filter(Boolean);

  if (!sections.length) return;

  const setCurrentYear = (heading) => {
    links.forEach((link) => {
      const current = link.getAttribute("href") === `#${heading.id}`;
      link.classList.toggle("is-current", current);
      if (current) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  };

  const updateCurrentYear = () => {
    const threshold = Math.min(rail.getBoundingClientRect().bottom + 24, 180);
    let current = sections[0];

    sections.forEach((heading) => {
      if (heading.getBoundingClientRect().top <= threshold) current = heading;
    });

    setCurrentYear(current);
  };

  updateCurrentYear();
  window.addEventListener("scroll", updateCurrentYear, { passive: true });
  window.addEventListener("resize", updateCurrentYear);
})();
