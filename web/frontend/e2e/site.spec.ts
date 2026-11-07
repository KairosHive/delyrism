import { expect, Page, test } from "@playwright/test";

// The online page, tab by tab: a space builds, every tab renders without an
// error, the Live tab answers a context and a query, and a space the server
// lost is rebuilt without the user seeing a 404.

const external = !!process.env.E2E_URL;

/** Console errors and failed API calls, collected per test. */
function watch(page: Page) {
  const problems: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !/favicon|DevTools/.test(m.text())) problems.push(`console: ${m.text()}`);
  });
  page.on("response", (r) => {
    const u = new URL(r.url());
    if (r.status() >= 500 || (r.status() === 404 && !u.pathname.includes("favicon"))) {
      problems.push(`${r.status()} ${r.request().method()} ${u.pathname}`);
    }
  });
  return problems;
}

async function build(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: /^(Build|Rebuild) space$/ }).click();
  await expect(page.getByRole("button", { name: "Rebuild space" })).toBeVisible();
  await expect(page.getByText("Ranked archetypes")).toBeVisible();
}

async function openTab(page: Page, name: string) {
  await page.getByRole("button", { name: new RegExp(`^${name}\\b`) }).click();
}

test("the page loads with its presets", async ({ page }) => {
  const problems = watch(page);
  await page.goto("/");
  await expect(page.getByRole("button", { name: /^(Build|Rebuild) space$/ })).toBeVisible();
  const presets = page.locator("select").filter({ has: page.locator('option[value="elements"]') });
  await expect(presets).toHaveCount(1);
  expect(problems).toEqual([]);
});

test("a space builds and the Explorer fills in", async ({ page }) => {
  const problems = watch(page);
  await build(page);
  await page.getByPlaceholder(/flooding spirits/).fill("flooding spirits dancing around floating suns");
  await expect(page.getByText("Failed to fetch")).toHaveCount(0);
  await page.waitForTimeout(1500);
  expect(problems).toEqual([]);
});

for (const tab of ["Topology", "Story", "Builder"]) {
  test(`the ${tab} tab renders`, async ({ page }) => {
    const problems = watch(page);
    await build(page);
    await openTab(page, tab);
    // wait for the tab's requests (Topology computes for a while) so the next test starts on an idle server
    await page.waitForLoadState("networkidle", { timeout: 120_000 });
    await expect(page.getByText(/unknown space_id|Internal Server Error/)).toHaveCount(0);
    expect(problems).toEqual([]);
  });
}

test("Live answers a context and a typed query", async ({ page }) => {
  const problems = watch(page);
  await build(page);
  await openTab(page, "Live");
  await expect(page.getByText(/^ready · \d+ edges/)).toBeVisible();
  await page.getByPlaceholder("a slow fire under ice…").fill("flooding spirits dancing around floating suns");
  await expect(page.getByText(/^ready · [1-9]\d* edges/)).toBeVisible();
  await page.getByPlaceholder(/what are you looking for in it/).fill("rain over the river");
  await expect(page.getByText(/text query on/)).toBeVisible();
  expect(problems).toEqual([]);
});

test("a space the server lost is rebuilt without a visible error", async ({ page, request }) => {
  test.skip(external, "needs the test server's /__test__ routes");
  await build(page);
  await openTab(page, "Live");
  await expect(page.getByText(/^ready · \d+ edges/)).toBeVisible();

  const forgot = await (await request.post("/__test__/forget-spaces")).json();
  expect(forgot.forgot).toBeGreaterThan(0);

  // a call with the lost id: the page rebuilds the space and retries
  const rebuilt = page.waitForResponse((r) => r.url().endsWith("/spaces") && r.request().method() === "POST");
  await openTab(page, "Topology");
  expect((await rebuilt).status()).toBe(200);
  await page.waitForTimeout(1500);
  await expect(page.getByText(/unknown space_id/)).toHaveCount(0);

  // the Live socket reconnects to the rebuilt space
  await openTab(page, "Live");
  await expect(page.getByText(/^ready · \d+ edges/)).toBeVisible();
});
