(() => {
  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const input = document.createElement("textarea");
    input.value = text;
    input.setAttribute("readonly", "");
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.append(input);
    input.select();
    document.execCommand("copy");
    input.remove();
  };

  document.querySelectorAll("[data-copy-paper]").forEach((button) => {
    const label = button.textContent;
    button.addEventListener("click", async () => {
      const citation = button.closest(".paper-citation");
      let value =
        button.dataset.copyPaper === "bibtex"
          ? citation?.dataset.bibtexCopy
          : citation?.dataset.citationCopy;
      if (!value) return;
      if (button.dataset.copyPaper === "bibtex") value = value.replaceAll("\\n", "\n");

      try {
        await copyText(value);
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = label;
        }, 1600);
      } catch {
        button.textContent = "Copy failed";
      }
    });
  });
})();
