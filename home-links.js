document.querySelectorAll("[data-contact-user]").forEach((link) => {
  const address = `${link.dataset.contactUser}@${link.dataset.contactHost}.${link.dataset.contactDomain}`;
  link.href = `mailto:${address}`;
});
