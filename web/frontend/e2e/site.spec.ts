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

test("Live speech mode transcribes the microphone into a spoken query", async ({ page }) => {
  test.skip(external, "uses the test server's stand-in transcriber and Chromium's fake microphone");
  const problems = watch(page);
  await build(page);
  await openTab(page, "Live");
  await expect(page.getByText(/^ready · \d+ edges/)).toBeVisible();
  await page.getByPlaceholder("a slow fire under ice…").fill("flooding spirits dancing around floating suns");
  await page.getByRole("button", { name: "speech → text" }).click();
  await page.getByRole("button", { name: "● listen" }).click();
  await expect(page.getByText("the water rises")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/speech query on/)).toBeVisible();
  await page.getByRole("button", { name: "■ stop" }).click();
  expect(problems).toEqual([]);
});

test("an audio-capable space offers the microphone in Live", async ({ page }) => {
  test.skip(external, "builds with the test server's stand-in for Gemini");
  await page.goto("/");
  await page.getByText("Embedding Model", { exact: true }).click();      // the sidebar section holding the dropdown
  const backend = page.locator("select").filter({ has: page.locator('option[value="gemini-embedding-2"]') });
  await expect(backend).toHaveCount(1);
  await backend.selectOption("gemini-embedding-2");
  await page.getByRole("button", { name: /^(Build|Rebuild) space$/ }).click();
  await expect(page.getByRole("button", { name: "Rebuild space" })).toBeVisible();
  await openTab(page, "Live");
  await page.getByRole("button", { name: "Gemini · mic" }).click();
  await expect(page.getByText("centre on this session")).toBeVisible();
  await page.getByRole("button", { name: "● listen" }).click();
  // Chromium's fake microphone beeps: after five windows the query is centred on the session
  await expect(page.getByText(/centred on \d+ queries|nothing varies yet/)).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "■ stop" }).click();
});

test("a Live session records and opens its result panel", async ({ page }) => {
  test.skip(external, "a 30 s recording against the test server");
  test.setTimeout(120_000);
  const problems = watch(page);
  await build(page);
  await openTab(page, "Live");
  await expect(page.getByText(/^ready · \d+ edges/)).toBeVisible();
  const context = page.getByPlaceholder("a slow fire under ice…");
  const query = page.getByPlaceholder(/what are you looking for in it/);
  await context.fill("flooding spirits dancing around floating suns");
  await page.getByPlaceholder("name (optional)").fill("e2e");
  await page.getByRole("button", { name: "● Start session" }).click();
  await expect(page.getByText("recording")).toBeVisible();
  const steps = [["rain over the river", ""], ["deep water and tears", ""], ["embers, smoke and fury", "a fire in the night"],
                 ["the forge and the sun", ""], ["wind and open sky", "a storm over the plain"], ["rain over the river", ""]];
  for (const [q, c] of steps) {
    if (c) await context.fill(c);
    await query.fill(q);
    await page.waitForTimeout(5_000);
  }
  await page.getByRole("button", { name: "■ End session" }).click();
  const panel = page.getByTestId("session-panel");
  await expect(panel).toBeVisible({ timeout: 60_000 });   // the first session on a fresh server also fits the UMAP map
  for (const title of ["Trajectory", "Territory", "Returns", "Topology over time", "Words", "Session archetypes", "Fingerprint"])
    await expect(panel.getByRole("heading", { name: title, exact: true })).toBeVisible();
  await expect(panel.getByRole("button", { name: /embers, smoke and fury/ })).toBeVisible();   // the words, in order
  const raw = await page.request.get(await panel.getByRole("link", { name: "↓ raw data" }).getAttribute("href") as string);
  expect(raw.ok()).toBeTruthy();
  expect((await raw.json()).raw.events.filter((e: any) => e.type === "query").length).toBeGreaterThanOrEqual(5);
  await panel.getByRole("button", { name: /weave a text/ }).click();
  await expect(panel.locator("blockquote")).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 3200 });
  await panel.screenshot({ path: "test-results/session-panel.png" });
  // the playhead: click the session strip at 40 %, every view follows
  const strip = panel.locator("svg.cursor-ew-resize");
  const sb = await strip.boundingBox();
  if (sb) await page.mouse.click(sb.x + sb.width * 0.4, sb.y + sb.height / 2);
  await panel.screenshot({ path: "test-results/session-panel-mid.png" });
  expect(problems).toEqual([]);
});
