/**
 * NHL Draft Simulator — k6 Load Test
 *
 * Tests the two most expensive simulation endpoints under realistic load.
 *
 * Usage:
 *   # Against local dev
 *   BASE_URL=http://localhost:8000 k6 run tests/load/k6_script.js
 *
 *   # Against staging with HTML report
 *   BASE_URL=https://staging.nhl-draft.example.com \
 *     k6 run --out json=load_test_results.json tests/load/k6_script.js
 *
 * Scenarios:
 *   load_test  — Ramp to 50 VUs over 30s, hold 2 min, ramp down (steady state)
 *   spike_test — Sudden 100-VU burst after 3 min (resilience check)
 *
 * Thresholds (CI gate — test fails if these are breached):
 *   p95 lottery latency < 2 000 ms
 *   p95 draft latency   < 5 000 ms  (ML inference is heavier)
 *   error rate          < 1 %
 *   HTTP failure rate   < 1 %
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ── Custom metrics ─────────────────────────────────────────────────────────────
const errorRate       = new Rate('errors');
const lotteryDuration = new Trend('lottery_duration_ms', true);
const draftDuration   = new Trend('draft_duration_ms',   true);

// ── Test configuration ─────────────────────────────────────────────────────────
export const options = {
  scenarios: {
    load_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 50  },   // ramp up
        { duration: '2m',  target: 50  },   // steady state
        { duration: '30s', target: 0   },   // ramp down
      ],
    },
    spike_test: {
      executor: 'ramping-vus',
      startTime: '3m30s',  // start after load_test ramp-down
      startVUs: 0,
      stages: [
        { duration: '10s', target: 100 },   // sudden spike
        { duration: '30s', target: 100 },   // hold spike
        { duration: '10s', target: 0   },   // release
      ],
    },
  },

  thresholds: {
    // Lottery simulation: p95 < 2s
    'lottery_duration_ms': ['p(95)<2000'],
    // Draft simulation (ML inference): p95 < 5s
    'draft_duration_ms':   ['p(95)<5000'],
    // Overall error rate < 1%
    'errors':              ['rate<0.01'],
    // HTTP-level failures (4xx/5xx that aren't expected) < 1%
    'http_req_failed':     ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// A realistic 32-team pick order (matches the actual NHL abbreviations used in
// the draft simulation endpoint). In production this would come from the lottery
// result, but for load testing a fixed order is fine.
const MOCK_LOTTERY_PICK_ORDER = [
  'SJS', 'CHI', 'ANA', 'NSH', 'PHI', 'BUF', 'MTL', 'DET',
  'STL', 'OTT', 'VAN', 'PIT', 'MIN', 'CBJ', 'NYI', 'WSH',
  'CAR', 'FLA', 'NJD', 'TOR', 'CGY', 'EDM', 'BOS', 'NYR',
  'TBL', 'COL', 'DAL', 'ARI', 'SEA', 'VGK', 'LAK', 'WPG',
];

const JSON_HEADERS = { 'Content-Type': 'application/json' };

// ── Test entrypoint ────────────────────────────────────────────────────────────
export default function () {
  const rand = Math.random();

  if (rand < 0.50) {
    // ── Lottery simulation (50%) ─────────────────────────────────────────────
    const start = Date.now();
    const res = http.post(
      `${BASE_URL}/api/lottery/simulate`,
      JSON.stringify({}),
      { headers: JSON_HEADERS },
    );
    lotteryDuration.add(Date.now() - start);

    const ok = check(res, {
      'lottery: status 200':        (r) => r.status === 200,
      'lottery: has pick_order':    (r) => {
        try { return JSON.parse(r.body).pick_order !== undefined; }
        catch { return false; }
      },
      'lottery: pick_order length': (r) => {
        try { return JSON.parse(r.body).pick_order.length === 32; }
        catch { return false; }
      },
    });
    errorRate.add(!ok);

  } else if (rand < 0.80) {
    // ── Draft simulation (30%) — most expensive, uses ML model ───────────────
    const start = Date.now();
    const res = http.post(
      `${BASE_URL}/api/draft/simulate`,
      JSON.stringify({ pick_order: MOCK_LOTTERY_PICK_ORDER }),
      { headers: JSON_HEADERS },
    );
    draftDuration.add(Date.now() - start);

    const ok = check(res, {
      // 200 = success, 503 = model not loaded (acceptable in test env)
      'draft: status 200 or 503': (r) => r.status === 200 || r.status === 503,
      'draft: has picks or detail': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.picks !== undefined || body.detail !== undefined;
        } catch { return false; }
      },
    });
    errorRate.add(!ok);

  } else if (rand < 0.90) {
    // ── Prospect search (10%) ────────────────────────────────────────────────
    const positions = ['C', 'LW', 'RW', 'D', 'G'];
    const pos = positions[Math.floor(Math.random() * positions.length)];
    const res = http.get(`${BASE_URL}/api/prospects?position=${pos}&limit=20`);

    const ok = check(res, {
      'prospects: status 200': (r) => r.status === 200,
    });
    errorRate.add(!ok);

  } else {
    // ── Health + metrics (10%) ───────────────────────────────────────────────
    const endpoint = Math.random() < 0.5 ? '/health' : '/metrics';
    const res = http.get(`${BASE_URL}${endpoint}`);

    const ok = check(res, {
      'infra: status 200': (r) => r.status === 200,
    });
    errorRate.add(!ok);
  }

  // Think time: 0–500 ms (models a realistic user pacing between actions)
  sleep(Math.random() * 0.5);
}

// ── Summary output ─────────────────────────────────────────────────────────────
export function handleSummary(data) {
  return {
    // Write full JSON results for CI artifact upload / diff tracking
    'load_test_results.json': JSON.stringify(data, null, 2),
    // Print a human-readable summary to stdout
    stdout: _textSummary(data),
  };
}

function _textSummary(data) {
  const m = data.metrics;
  const p = (metric, pct) => {
    const v = m[metric];
    if (!v) return 'N/A';
    return v.values[`p(${pct})`]
      ? `${v.values[`p(${pct})`].toFixed(0)} ms`
      : 'N/A';
  };

  return [
    '\n=== NHL Draft Simulator Load Test Summary ===',
    `Lottery p50/p95: ${p('lottery_duration_ms', 50)} / ${p('lottery_duration_ms', 95)}`,
    `Draft   p50/p95: ${p('draft_duration_ms',   50)} / ${p('draft_duration_ms',   95)}`,
    `Error rate:      ${(m.errors?.values?.rate * 100 || 0).toFixed(2)}%`,
    `HTTP failures:   ${(m.http_req_failed?.values?.rate * 100 || 0).toFixed(2)}%`,
    `Total requests:  ${m.http_reqs?.values?.count || 0}`,
    '==============================================\n',
  ].join('\n');
}
