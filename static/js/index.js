const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector(".nav-links");
const leaderboardSortSelect = document.querySelector("#leaderboard-sort-select");

const leaderboardMetrics = {
  relativeProfit: {
    title: "Relative Profit",
    rule: "Higher is better",
    copy: "Seller-side total profit, normalized against GPT-3.5 as the 1.0x baseline.",
    suffix: "x",
    decimals: 1,
    sort: "desc"
  },
  sellerPrr: {
    title: "Seller PRR",
    rule: "Lower is better",
    copy: "Seller price reduction rate measures how much the seller lowers the price; lower values indicate stronger seller-side outcomes.",
    suffix: "%",
    decimals: 2,
    sort: "asc"
  },
  buyerPrr: {
    title: "Buyer PRR",
    rule: "Higher is better",
    copy: "Buyer price reduction rate measures the discount the buyer obtains from the seller's initial offer; higher values indicate stronger buyer-side outcomes.",
    suffix: "%",
    decimals: 2,
    sort: "desc"
  }
};

const leaderboardRows = [
  { model: "GPT-4.1", relativeProfit: 13.3, sellerPrr: 8.80, buyerPrr: 12.76 },
  { model: "DeepSeek-R1", relativeProfit: 12.0, sellerPrr: 10.18, buyerPrr: 11.85 },
  { model: "o4-mini", relativeProfit: 10.5, sellerPrr: 10.58, buyerPrr: 13.02 },
  { model: "o3", relativeProfit: 10.2, sellerPrr: 7.69, buyerPrr: 13.43 },
  { model: "GPT-4o-mini", relativeProfit: 9.2, sellerPrr: 8.68, buyerPrr: 11.65 },
  { model: "DeepSeek-V3", relativeProfit: 8.4, sellerPrr: 10.08, buyerPrr: 11.60 },
  { model: "Qwen2.5-7B", relativeProfit: 6.3, sellerPrr: 17.16, buyerPrr: 11.39 },
  { model: "Qwen2.5-14B", relativeProfit: 6.1, sellerPrr: 15.86, buyerPrr: 11.67 },
  { model: "GPT-3.5", relativeProfit: 1.0, sellerPrr: 19.21, buyerPrr: 10.20 }
];

function formatMetric(value, metricKey) {
  const metric = leaderboardMetrics[metricKey];
  return `${value.toFixed(metric.decimals)}${metric.suffix}`;
}

function renderLeaderboard(metricKey = "relativeProfit") {
  const metric = leaderboardMetrics[metricKey];
  const tableBody = document.querySelector("#leaderboard-table-body");
  if (!metric || !tableBody) return;

  document.querySelector("#leaderboard-metric-title").textContent = metric.title;
  document.querySelector("#leaderboard-metric-rule").textContent = metric.rule;
  document.querySelector("#leaderboard-metric-copy").textContent = metric.copy;

  const rows = [...leaderboardRows].sort((a, b) => {
    return metric.sort === "asc" ? a[metricKey] - b[metricKey] : b[metricKey] - a[metricKey];
  });

  tableBody.innerHTML = rows.map((row, index) => `
    <tr>
      <td class="rank-cell">#${index + 1}</td>
      <td>${row.model}</td>
      <td class="metric-value ${metricKey === "relativeProfit" ? "sorted-metric" : ""}">${formatMetric(row.relativeProfit, "relativeProfit")}</td>
      <td class="metric-value ${metricKey === "sellerPrr" ? "sorted-metric" : ""}">${formatMetric(row.sellerPrr, "sellerPrr")}</td>
      <td class="metric-value ${metricKey === "buyerPrr" ? "sorted-metric" : ""}">${formatMetric(row.buyerPrr, "buyerPrr")}</td>
    </tr>
  `).join("");

  if (leaderboardSortSelect) {
    leaderboardSortSelect.value = metricKey;
  }

  document.querySelectorAll("[data-sort-column]").forEach((heading) => {
    const isSorted = heading.dataset.sortColumn === metricKey;
    heading.classList.toggle("sorted-column", isSorted);
    heading.setAttribute("aria-sort", isSorted ? (metric.sort === "asc" ? "ascending" : "descending") : "none");
  });
}

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const expanded = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!expanded));
    navLinks.classList.toggle("open");
  });

  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  });
}

if (leaderboardSortSelect) {
  leaderboardSortSelect.addEventListener("change", () => {
    renderLeaderboard(leaderboardSortSelect.value);
  });
}

renderLeaderboard("relativeProfit");
