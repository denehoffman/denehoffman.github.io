(() => {
  const publications = {
    "10.1038/s41467-023-39602-2": {
      venue: "Nature Communications · 14, 3960",
      citation:
        "M. ElKabbash et al., “Fano resonant optical coatings platform for full gamut and high purity structural colors,” Nature Communications 14, 3960 (2023). https://doi.org/10.1038/s41467-023-39602-2",
      bibtex: `@article{ElKabbash2023,
  author = {ElKabbash, Mohamed and Hoffman, Nathaniel and Lininger, Andrew R. and Jalil, Sohail A. and Letsou, Theodore and Hinczewski, Michael and Strangi, Giuseppe and Guo, Chunlei},
  title = {Fano resonant optical coatings platform for full gamut and high purity structural colors},
  journal = {Nature Communications},
  volume = {14},
  pages = {3960},
  year = {2023},
  doi = {10.1038/s41467-023-39602-2}
}`,
    },
    "10.1007/s11661-021-06182-z": {
      venue: "Metallurgical and Materials Transactions A · 52, 1551–1558",
      citation:
        "N. Hoffman and M. Widom, “Cluster variation method analysis of correlations and entropy in BCC solid solutions,” Metallurgical and Materials Transactions A 52, 1551–1558 (2021). https://doi.org/10.1007/s11661-021-06182-z",
      bibtex: `@article{Hoffman2021,
  author = {Hoffman, Nathaniel and Widom, Michael},
  title = {Cluster Variation Method Analysis of Correlations and Entropy in BCC Solid Solutions},
  journal = {Metallurgical and Materials Transactions A},
  volume = {52},
  number = {5},
  pages = {1551--1558},
  year = {2021},
  doi = {10.1007/s11661-021-06182-z}
}`,
    },
    "10.1038/s41565-020-00841-9": {
      venue: "Nature Nanotechnology · 16, 440–446",
      citation:
        "M. ElKabbash et al., “Fano-resonant ultrathin film optical coatings,” Nature Nanotechnology 16, 440–446 (2021). https://doi.org/10.1038/s41565-020-00841-9",
      bibtex: `@article{ElKabbash2021,
  author = {ElKabbash, Mohamed and Letsou, Theodore and Jalil, Sohail A. and Hoffman, Nathaniel and Zhang, Jihua and Rutledge, James and Lininger, Andrew R. and Fann, Chun-Hao and Hinczewski, Michael and Strangi, Giuseppe and Guo, Chunlei},
  title = {Fano-resonant ultrathin film optical coatings},
  journal = {Nature Nanotechnology},
  volume = {16},
  number = {4},
  pages = {440--446},
  year = {2021},
  doi = {10.1038/s41565-020-00841-9}
}`,
    },
    "10.1088/1361-6501/ab9fd8": {
      venue: "Measurement Science and Technology · 31, 115201",
      citation:
        "M. ElKabbash et al., “Ultrathin-film optical coating for angle-independent remote hydrogen sensing,” Measurement Science and Technology 31, 115201 (2020). https://doi.org/10.1088/1361-6501/ab9fd8",
      bibtex: `@article{ElKabbash2020,
  author = {ElKabbash, Mohamed and Sreekanth, Kandammathe Valiyaveedu and Fraiwan, Arwa and Cole, Jonathan and Alapan, Yunus and Letsou, Theodore and Hoffman, Nathaniel and Guo, Chunlei and Sankaran, R. Mohan and Gurkan, Umut A. and Hinczewski, Michael and Strangi, Giuseppe},
  title = {Ultrathin-film optical coating for angle-independent remote hydrogen sensing},
  journal = {Measurement Science and Technology},
  volume = {31},
  number = {11},
  pages = {115201},
  year = {2020},
  doi = {10.1088/1361-6501/ab9fd8}
}`,
    },
    "10.1002/adom.201700617": {
      venue: "Advanced Optical Materials · 5, 1700617",
      citation:
        "M. ElKabbash et al., “Tunable black gold: Controlling the near-field coupling of immobilized Au nanoparticles embedded in mesoporous silica capsules,” Advanced Optical Materials 5, 1700617 (2017). https://doi.org/10.1002/adom.201700617",
      bibtex: `@article{ElKabbash2017BlackGold,
  author = {ElKabbash, Mohamed and Sousa-Castillo, Ana and Nguyen, Quang and Marino-Fernandez, Rosalia and Hoffman, Nathaniel and Correa-Duarte, Miguel A. and Strangi, Giuseppe},
  title = {Tunable Black Gold: Controlling the Near-Field Coupling of Immobilized Au Nanoparticles Embedded in Mesoporous Silica Capsules},
  journal = {Advanced Optical Materials},
  volume = {5},
  number = {21},
  pages = {1700617},
  year = {2017},
  doi = {10.1002/adom.201700617}
}`,
    },
    "10.1364/ol.42.003598": {
      venue: "Optics Letters · 42, 3598–3601",
      citation:
        "M. ElKabbash et al., “Iridescence-free and narrowband perfect light absorption in critically coupled metal high-index dielectric cavities,” Optics Letters 42, 3598–3601 (2017). https://doi.org/10.1364/OL.42.003598",
      bibtex: `@article{ElKabbash2017Iridescence,
  author = {ElKabbash, M. and Ilker, E. and Letsou, T. and Hoffman, N. and Yaney, A. and Hinczewski, M. and Strangi, G.},
  title = {Iridescence-free and narrowband perfect light absorption in critically coupled metal high-index dielectric cavities},
  journal = {Optics Letters},
  volume = {42},
  number = {18},
  pages = {3598--3601},
  year = {2017},
  doi = {10.1364/OL.42.003598}
}`,
    },
  };

  const getDoi = (link) => {
    try {
      return new URL(link.href).pathname.slice(1).toLowerCase();
    } catch {
      return "";
    }
  };

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

  const makeAction = (label, href) => {
    const action = document.createElement(href ? "a" : "button");
    action.className = "publication-action";
    action.textContent = label;
    if (href) action.href = href;
    else action.type = "button";
    return action;
  };

  document.querySelectorAll("main > h2").forEach((heading) => {
    const titleLink = heading.querySelector("a[href*='doi.org']");
    if (!titleLink) return;

    const publication = publications[getDoi(titleLink)];
    if (!publication) return;

    const record = document.createElement("article");
    record.className = "publication-record";
    heading.before(record);

    let node = heading;
    while (
      node &&
      !(
        node !== heading &&
        (/^(H1|H2)$/.test(node.nodeName) || node.matches?.(".collaboration-publications"))
      )
    ) {
      const next = node.nextSibling;
      record.append(node);
      node = next;
    }

    const arxivLink = record.querySelector("a[href*='arxiv.org']");
    const arxivHref = arxivLink?.href;
    arxivLink?.closest("p")?.remove();

    const metadata = document.createElement("p");
    metadata.className = "publication-record__venue";
    metadata.textContent = publication.venue;
    heading.after(metadata);

    const actions = document.createElement("div");
    actions.className = "publication-record__actions";
    actions.setAttribute("aria-label", "Publication links and citation actions");
    actions.append(makeAction("DOI", titleLink.href));
    if (arxivHref) actions.append(makeAction("arXiv", arxivHref));

    [
      ["Copy citation", publication.citation],
      ["Copy BibTeX", publication.bibtex],
    ].forEach(([label, text]) => {
      const button = makeAction(label);
      button.addEventListener("click", async () => {
        try {
          await copyText(text);
          button.textContent = "Copied";
          window.setTimeout(() => {
            button.textContent = label;
          }, 1600);
        } catch {
          button.textContent = "Copy failed";
        }
      });
      actions.append(button);
    });

    const abstract = record.querySelector("details");
    if (abstract) abstract.before(actions);
    else record.append(actions);
  });
})();
