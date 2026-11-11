const fs = require('fs');

const reportPath = process.argv[2] || './lh-desktop.json';
if (!fs.existsSync(reportPath)) {
  console.error(`Error: Report file not found at ${reportPath}`);
  process.exit(1);
}

const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));

const scores = {};
if (report.categories) {
  for (const [key, cat] of Object.entries(report.categories)) {
    scores[cat.title] = Math.round(cat.score * 100);
  }
}

const metrics = {};
const metricIds = [
  'first-contentful-paint',
  'largest-contentful-paint',
  'total-blocking-time',
  'cumulative-layout-shift',
  'speed-index',
  'interactive'
];

for (const id of metricIds) {
  if (report.audits && report.audits[id]) {
    metrics[id] = {
      score: report.audits[id].score,
      displayValue: report.audits[id].displayValue || report.audits[id].numericValue
    };
  }
}

const failedAudits = [];
if (report.audits) {
  for (const [id, audit] of Object.entries(report.audits)) {
    if (
      audit.score !== null &&
      audit.score < 1 &&
      audit.scoreDisplayMode !== 'notApplicable' &&
      audit.scoreDisplayMode !== 'informative' &&
      audit.scoreDisplayMode !== 'manual'
    ) {
      let items = [];
      if (Array.isArray(audit.details?.items)) {
        items = audit.details.items.slice(0, 5).map(it => {
          if (typeof it === 'string') return it;
          return it.node?.snippet || it.url || it.label || it.description || it.text || JSON.stringify(it);
        });
      }
      failedAudits.push({
        id,
        title: audit.title,
        score: audit.score,
        displayValue: audit.displayValue || '',
        items
      });
    }
  }
}

console.log(JSON.stringify({ scores, metrics, failedAuditsCount: failedAudits.length, failedAudits }, null, 2));

