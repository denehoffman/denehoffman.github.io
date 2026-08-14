const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;

const pages = [
  ["/", "Dr. Dene Hoffman"],
  ["/projects/", "Projects"],
  ["/projects/laddu/", "laddu"],
  ["/projects/yamloom/", "yamloom"],
  ["/projects/pdg-rs/", "pdg-rs"],
  ["/projects/splotrs/", "splotrs"],
  ["/projects/see/", "see"],
  ["/projects/hddm-rs/", "hddm-rs"],
  ["/publications/", "Publications"],
  ["/blog/the-bfgs-algorithm-family-in-rust-part-3/", "The BFGS Algorithm Family in Rust (Part 3)"],
];

for (const [path, heading] of pages) {
  test(`${path} renders without serious accessibility or overflow problems`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: heading }).filter({ visible: true })).toBeVisible();
    const dimensions = await page.evaluate(() => {
      const clientWidth = document.documentElement.clientWidth;
      const overflowing = [...document.querySelectorAll("body *")]
        .map((element) => {
          const rect = element.getBoundingClientRect();
          return {
            selector: `${element.tagName.toLowerCase()}${element.id ? `#${element.id}` : ""}${[...element.classList].map((name) => `.${name}`).join("")}`,
            left: Math.round(rect.left),
            right: Math.round(rect.right),
            width: Math.round(rect.width),
          };
        })
        .filter(({ left, right }) => left < -1 || right > clientWidth + 1)
        .slice(0, 12);
      return {
        clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
        overflowing,
      };
    });
    expect(dimensions, `Overflowing elements: ${JSON.stringify(dimensions.overflowing)}`).toMatchObject({
      clientWidth: dimensions.scrollWidth,
    });

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .exclude(".giallo")
      .analyze();
    const serious = results.violations.filter(({ impact }) => impact === "serious" || impact === "critical");
    expect(serious.map(({ id, nodes }) => ({
      id,
      targets: nodes.map((node) => node.target.join(" ")),
    }))).toEqual([]);
  });
}

test("project status badges are reserved for exceptional states", async ({ page }) => {
  await page.goto("/projects/");
  await expect(page.locator(".project-directory__card")).toHaveCount(11);
  await expect(page.locator(".project-directory__status", { hasText: /^Archived$/ })).toHaveCount(2);
  await expect(page.locator(".project-directory__status", { hasText: /^Experimental$/ })).toHaveCount(1);
  await expect(page.locator(".project-directory__status", { hasText: /^Active/ })).toHaveCount(0);
});

test("publication topics filter records and report the result", async ({ page }) => {
  await page.goto("/publications/");
  await page.getByRole("button", { name: "Materials" }).click();
  await expect(page.locator(".publication-filters__status")).toHaveText("2 publications");
  await expect(page.getByRole("button", { name: "Materials" })).toHaveAttribute("aria-pressed", "true");
});

test("custom scripts are loaded only on relevant pages", async ({ page }) => {
  const scriptNames = () => page.locator("script[src]").evaluateAll((scripts) =>
    scripts.map((script) => new URL(script.src).pathname.split("/").pop()),
  );

  await page.goto("/");
  expect(await scriptNames()).toContain("home-links.js");
  expect(await scriptNames()).not.toContain("publication-records.js");

  await page.goto("/publications/");
  expect(await scriptNames()).toContain("publication-records.js");
  expect(await scriptNames()).not.toContain("laddu-interference.js");

  await page.goto("/projects/laddu/");
  expect(await scriptNames()).toContain("laddu-interference.js");
  expect(await scriptNames()).not.toContain("publication-records.js");
});
