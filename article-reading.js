(() => {
  const content = document.querySelector("[data-reading-content]");
  const progress = document.querySelector("[data-reading-progress] span");
  const toc = document.querySelector("[data-reading-toc]");

  if (!content || !progress || !toc) return;

  const desktop = window.matchMedia("(min-width: 1180px)");
  const syncTocState = () => {
    if (desktop.matches) toc.open = true;
  };

  const updateProgress = () => {
    const start = content.getBoundingClientRect().top + window.scrollY;
    const distance = Math.max(content.offsetHeight - window.innerHeight, 1);
    const amount = Math.min(Math.max((window.scrollY - start) / distance, 0), 1);
    progress.style.transform = `scaleX(${amount})`;
  };

  const links = [...toc.querySelectorAll("a[href*='#']")];
  const sections = links
    .map((link) => {
      const id = decodeURIComponent(link.hash.slice(1));
      return { link, heading: document.getElementById(id) };
    })
    .filter(({ heading }) => heading);

  const updateCurrentSection = () => {
    const threshold = Math.min(window.innerHeight * 0.28, 180);
    let current = sections[0];

    sections.forEach((section) => {
      if (section.heading.getBoundingClientRect().top <= threshold) current = section;
    });

    links.forEach((link) => {
      const active = current && link === current.link;
      link.classList.toggle("is-current", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };

  let scheduled = false;
  const update = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      updateProgress();
      updateCurrentSection();
      scheduled = false;
    });
  };

  syncTocState();
  desktop.addEventListener("change", syncTocState);
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
})();
