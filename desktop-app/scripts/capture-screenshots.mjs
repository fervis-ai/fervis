import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

import { chromium } from "playwright";

const baseUrl = process.env.FERVIS_DESKTOP_APP_URL ?? "http://127.0.0.1:5173/?demo=1";
const outputDir = resolve(
  process.cwd(),
  "../docs/desktop-app/test-runs/2026-06-27-visual-verification/screenshots"
);

await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch();
try {
  await captureCompletedDesktopLight(browser);
  await captureCompletedDesktopDark(browser);
  await captureCompletedTabletLight(browser);
  await captureCompletedTabletDark(browser);
  await captureRunning(browser);
  await captureChoiceClarification(browser);
  await captureTextClarification(browser);
  await captureFailed(browser);
  await captureSettingsLight(browser);
  await captureSettingsDark(browser);
} finally {
  await browser.close();
}

async function newPage(browser, viewport) {
  const page = await browser.newPage({ viewport });
  await page.goto(baseUrl);
  await page.getByText("18 in-person sales happened this month.").waitFor();
  return page;
}

async function captureCompletedDesktopLight(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByLabel("Switch theme from system").click();
  await screenshot(page, "desktop-light-completed.png");
  await page.close();
}

async function captureCompletedDesktopDark(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByLabel("Switch theme from system").click();
  await page.getByLabel("Switch theme from light").click();
  await screenshot(page, "desktop-dark-completed.png");
  await page.close();
}

async function captureCompletedTabletLight(browser) {
  const page = await newPage(browser, { width: 900, height: 1024 });
  await page.getByLabel("Switch theme from system").click();
  await screenshot(page, "tablet-light-completed.png");
  await page.close();
}

async function captureCompletedTabletDark(browser) {
  const page = await newPage(browser, { width: 900, height: 1024 });
  await page.getByLabel("Switch theme from system").click();
  await page.getByLabel("Switch theme from light").click();
  await screenshot(page, "tablet-dark-completed.png");
  await page.close();
}

async function captureRunning(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByText("Which store has the most inventory at risk today?").click();
  await page.getByText("Selecting the sales endpoint for this month.").waitFor();
  await screenshot(page, "desktop-light-running.png");
  await page.close();
}

async function captureChoiceClarification(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByText("run_clarify").click();
  await page.locator(".clar-question", { hasText: "Which store do you mean?" }).waitFor();
  await screenshot(page, "desktop-light-choice-clarification.png");
  await page.close();
}

async function captureTextClarification(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByRole("button", { name: /What were sales for BBS last month?/ }).click();
  await page.locator(".clar-question", { hasText: "Which March should I use?" }).waitFor();
  await screenshot(page, "desktop-light-text-clarification.png");
  await page.close();
}

async function captureFailed(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByRole("button", {
    name: /Which returns endpoint failed during settlement review?/
  }).click();
  await page.getByText(/provider_runtime_failed/).waitFor();
  await screenshot(page, "desktop-light-failed.png");
  await page.close();
}

async function captureSettingsLight(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByLabel("Switch theme from system").click();
  await page.getByLabel("Open connection settings").click();
  await page.getByRole("dialog", { name: "Connection settings" }).waitFor();
  await screenshot(page, "desktop-light-settings.png", false);
  await page.close();
}

async function captureSettingsDark(browser) {
  const page = await newPage(browser, { width: 1440, height: 900 });
  await page.getByLabel("Switch theme from system").click();
  await page.getByLabel("Switch theme from light").click();
  await page.getByLabel("Open connection settings").click();
  await page.getByRole("dialog", { name: "Connection settings" }).waitFor();
  await screenshot(page, "desktop-dark-settings.png", false);
  await page.close();
}

async function screenshot(page, name, fullPage = true) {
  await page.screenshot({
    fullPage,
    path: resolve(outputDir, name)
  });
}
