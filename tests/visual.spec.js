const { test, expect } = require("@playwright/test");

const pages = [
  ["home", "/"],
  ["projects", "/projects/"],
  ["publications", "/publications/"],
  ["laddu", "/projects/laddu/"],
];

for (const [name, path] of pages) {
  test(`${name} visual baseline @visual`, async ({ page }) => {
    await page.goto(path);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect(page).toHaveScreenshot(`${name}.png`, {
      animations: "disabled",
      fullPage: true,
      maxDiffPixelRatio: 0.01,
    });
  });
}
