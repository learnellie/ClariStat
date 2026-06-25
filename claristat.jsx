import React, { useState, useMemo, useEffect, useRef } from "react";

/* ============================================================
   CLARISTAT — biomedical statistics: learn → ask → analyze
   Palette "lab notebook at dusk":
     ink #1B2A2F · paper #F7F4EC · sage #5B7B6F
     amber #C97B3E · slate #3E5C6E · line #DAD3C2 · faint #EFEADD
   ============================================================ */

const C = {
  ink: "#1B2A2F", paper: "#F7F4EC", card: "#FFFFFF", sage: "#5B7B6F",
  amber: "#C97B3E", slate: "#3E5C6E", line: "#DAD3C2", faint: "#EFEADD",
  ok: "#2E7D55", warn: "#B5532A",
};
const mono = "'IBM Plex Mono', ui-monospace, monospace";
const serif = "'Fraunces', Georgia, serif";
const sans = "'Inter', system-ui, sans-serif";

/* ---------------- math core ---------------- */
const sum = (a) => a.reduce((s, x) => s + x, 0);
const mean = (a) => sum(a) / a.length;
const geomean = (a) => Math.exp(mean(a.map((x) => Math.log(x))));
const variance = (a, samp = true) => {
  const m = mean(a);
  return sum(a.map((x) => (x - m) ** 2)) / (a.length - (samp ? 1 : 0));
};
const sd = (a, samp = true) => Math.sqrt(variance(a, samp));
const median = (a) => {
  const s = [...a].sort((x, y) => x - y), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};
const quantile = (a, q) => {
  const s = [...a].sort((x, y) => x - y);
  const pos = (s.length - 1) * q, b = Math.floor(pos), r = pos - b;
  return s[b + 1] !== undefined ? s[b] + r * (s[b + 1] - s[b]) : s[b];
};

/* log-gamma & incomplete beta -> exact Student-t two-sided p */
function logGamma(x) {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let xx = x, y = x, tmp = x + 5.5;
  tmp -= (xx + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += g[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / xx);
}
function betacf(a, b, x) {
  const EPS = 3e-12, FPMIN = 1e-300;
  let qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d; let h = d;
  for (let m = 1; m <= 200; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}
function ibeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) +
    a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? (bt * betacf(a, b, x)) / a
    : 1 - (bt * betacf(b, a, 1 - x)) / b;
}
const tTwoSidedP = (t, df) => ibeta(df / (df + t * t), df / 2, 0.5);
const normCdf = (z) => {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
};
/* inverse t via bisection on two-sided tail prob */
function tInv(p2sided, df) {
  let lo = 0, hi = 1000;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    tTwoSidedP(mid, df) > p2sided ? (lo = mid) : (hi = mid);
  }
  return (lo + hi) / 2;
}
const tCrit = (df, conf = 0.95) => tInv(1 - conf, df);

/* Welch two-sample t */
function welch(a, b) {
  const ma = mean(a), mb = mean(b), va = variance(a), vb = variance(b);
  const na = a.length, nb = b.length;
  const se = Math.sqrt(va / na + vb / nb);
  const t = (ma - mb) / se;
  const df = (va / na + vb / nb) ** 2 /
    ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1));
  const p = tTwoSidedP(Math.abs(t), df);
  const sp = Math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2));
  const d = (ma - mb) / sp; // Cohen's d
  return { ma, mb, va, vb, na, nb, se, t, df, p, diff: ma - mb, d };
}
/* paired t */
function pairedT(a, b) {
  const d = a.map((x, i) => x - b[i]);
  const md = mean(d), s = sd(d), n = d.length;
  const se = s / Math.sqrt(n), t = md / se, df = n - 1;
  return { md, s, n, se, t, df, p: tTwoSidedP(Math.abs(t), df), dz: md / s };
}
/* one-way ANOVA */
function anova(groups) {
  const all = groups.flat(), N = all.length, k = groups.length;
  const gm = mean(all);
  const ssb = sum(groups.map((g) => g.length * (mean(g) - gm) ** 2));
  const ssw = sum(groups.map((g) => sum(g.map((x) => (x - mean(g)) ** 2))));
  const dfb = k - 1, dfw = N - k;
  const msb = ssb / dfb, msw = ssw / dfw, F = msb / msw;
  const p = 1 - fCdf(F, dfb, dfw);
  const eta2 = ssb / (ssb + ssw);
  return { F, dfb, dfw, p, eta2, msb, msw, k };
}
const fCdf = (f, d1, d2) => f <= 0 ? 0 : 1 - ibeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);

/* chi-square 2x2 / contingency */
function chiSq(table) {
  const rows = table.length, cols = table[0].length;
  const rt = table.map(sum), ct = table[0].map((_, j) => sum(table.map((r) => r[j])));
  const N = sum(rt);
  let chi = 0;
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const e = (rt[i] * ct[j]) / N;
      chi += (table[i][j] - e) ** 2 / e;
    }
  const df = (rows - 1) * (cols - 1);
  const p = 1 - chiSqCdf(chi, df);
  return { chi, df, p, N };
}
const chiSqCdf = (x, k) => x <= 0 ? 0 : lowerGamma(k / 2, x / 2) / Math.exp(logGamma(k / 2));
function lowerGamma(s, x) {
  let sum = 1 / s, term = sum;
  for (let n = 1; n < 200; n++) { term *= x / (s + n); sum += term; if (term < 1e-12) break; }
  return Math.pow(x, s) * Math.exp(-x) * sum;
}
/* linear regression */
function linreg(x, y) {
  const n = x.length, mx = mean(x), my = mean(y);
  const sxx = sum(x.map((xi) => (xi - mx) ** 2));
  const sxy = sum(x.map((xi, i) => (xi - mx) * (y[i] - my)));
  const slope = sxy / sxx, intercept = my - slope * mx;
  const yhat = x.map((xi) => slope * xi + intercept);
  const ssr = sum(y.map((yi, i) => (yi - yhat[i]) ** 2));
  const sst = sum(y.map((yi) => (yi - my) ** 2));
  const r2 = 1 - ssr / sst;
  const r = sxy / Math.sqrt(sxx * sst);
  const seSlope = Math.sqrt(ssr / (n - 2) / sxx);
  const tSlope = slope / seSlope;
  const p = tTwoSidedP(Math.abs(tSlope), n - 2);
  return { slope, intercept, r2, r, p, n, seSlope };
}

/* Shapiro–Wilk (approx, n 3..50) -> returns W and a normality verdict */
function shapiroApprox(a) {
  const n = a.length;
  if (n < 3) return null;
  const s = [...a].sort((x, y) => x - y);
  const m = s.map((_, i) => {
    const p = (i + 1 - 0.375) / (n + 0.25);
    // inverse normal (Beasley-Springer-Moro light)
    return invNorm(p);
  });
  const mm = Math.sqrt(sum(m.map((x) => x * x)));
  const c = m.map((x) => x / mm);
  const xb = mean(s);
  const num = sum(c.map((ci, i) => ci * s[i])) ** 2;
  const den = sum(s.map((x) => (x - xb) ** 2));
  const W = num / den;
  return { W, normalish: W > 0.9 };
}
function invNorm(p) {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1];
  const cc = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  if (p <= 1 - pl) { const q = p - 0.5, r = q*q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1); }
  const q = Math.sqrt(-2*Math.log(1-p)); return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
}

/* power for two-sample t (approx via normal) */
function powerTwoSample(d, n, alpha = 0.05) {
  const ncp = d * Math.sqrt(n / 2);
  const zc = invNorm(1 - alpha / 2);
  return 1 - normCdf(zc - ncp) + normCdf(-zc - ncp);
}

const fmt = (x, k = 2) => (isFinite(x) ? x.toFixed(k) : "—");
const fmtP = (p) => (p < 0.0001 ? "<0.0001" : p.toFixed(4));
const parse = (s) => s.split(/[\s,;\t\n]+/).map(Number).filter((x) => !isNaN(x));

/* ---------------- shared UI ---------------- */
const Eyebrow = ({ children }) => (
  <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: C.amber, marginBottom: 8 }}>{children}</div>
);
const Card = ({ children, style }) => (
  <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 4, padding: 22, ...style }}>{children}</div>
);
const Formula = ({ children }) => (
  <div style={{ fontFamily: mono, fontSize: 14.5, background: C.faint, border: `1px solid ${C.line}`, borderRadius: 3, padding: "12px 16px", margin: "12px 0", overflowX: "auto", lineHeight: 1.7 }}>{children}</div>
);
const Stat = ({ label, v, color = C.ink, hint }) => (
  <div style={{ background: C.faint, borderRadius: 3, padding: "10px 12px", minWidth: 92 }} title={hint || ""}>
    <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", color: C.sage }}>{label}</div>
    <div style={{ fontFamily: serif, fontSize: 21, color, marginTop: 2 }}>{v}</div>
  </div>
);
const Interpret = ({ children }) => (
  <div style={{ borderLeft: `3px solid ${C.sage}`, paddingLeft: 14, marginTop: 10, color: "#3a4a44", fontFamily: sans, fontSize: 14.5, lineHeight: 1.6 }}>{children}</div>
);
const proofBox = { background: C.faint, border: `1px solid ${C.line}`, borderLeft: `3px solid ${C.sage}`, borderRadius: 3, padding: "14px 16px", margin: "14px 0" };

/* glossary tooltip term */
const GLOSSARY = {
  "p-value": "Probability, assuming H₀ is true, of data at least as extreme as observed.",
  SEM: "Standard error of the mean = s/√n. Precision of the estimated mean.",
  SD: "Standard deviation. Spread of individual data points.",
  "Cohen's d": "Standardized effect size: mean difference in SD units.",
  "Welch": "A t-test that does not assume the two groups share a variance.",
  CI: "Confidence interval: a range of plausible parameter values from the data.",
};
const Term = ({ children }) => {
  const k = typeof children === "string" ? children : "";
  const def = GLOSSARY[k];
  if (!def) return <>{children}</>;
  return (
    <span style={{ borderBottom: `1px dotted ${C.amber}`, cursor: "help" }} title={def}>{children}</span>
  );
};

/* ============================================================
   MODULE 1 — LEARN  (six concepts, expandable)
   ============================================================ */
const LESSONS = [
  { id: "mean", title: "Mean", blurb: "Arithmetic, weighted & geometric — and when each is honest.", render: () => (<>
    <p style={pS}>The <b>mean</b> is a center of mass. Three flavors recur in biomedical work; the wrong one quietly distorts results.</p>
    <h4 style={h4}>Arithmetic</h4><Formula>x̄ = (1/n)·Σxᵢ</Formula>
    <p style={pS}>For symmetric, additive quantities (BP, age, counts). Minimizes squared deviation — which is why it pairs with variance and the normal curve.</p>
    <h4 style={h4}>Weighted</h4><Formula>x̄_w = Σ(wᵢxᵢ)/Σwᵢ</Formula>
    <p style={pS}>When observations carry unequal authority — pooling clinic means of different sizes, or inverse-variance weighting in meta-analysis.</p>
    <h4 style={h4}>Geometric</h4><Formula>x̄_g = (Πxᵢ)^(1/n) = exp((1/n)Σ ln xᵢ)</Formula>
    <p style={pS}>The correct center for <b>multiplicative / log-normal</b> data: titres, dilutions, viral loads, fold-changes. Dilution 2, 8, 32 → arithmetic mean 14 but geometric mean 8.</p>
    <div style={proofBox}><Eyebrow>Property (AM–GM)</Eyebrow><p style={{ ...pS, margin: 0 }}>x̄_g ≤ x̄ for positive data, equal only when all values match. Using the arithmetic mean on skewed multiplicative data systematically overstates the center.</p></div></>) },
  { id: "variance", title: "Variance", blurb: "Population vs sample, and why n−1.", render: () => (<>
    <Formula>Population: σ² = (1/N)Σ(xᵢ−μ)²</Formula><Formula>Sample: s² = (1/(n−1))Σ(xᵢ−x̄)²</Formula>
    <h4 style={h4}>Bessel's correction</h4>
    <p style={pS}>x̄ is computed from the data and sits as close to the points as possible, so Σ(xᵢ−x̄)² runs small. The expectation is exactly E[Σ(xᵢ−x̄)²] = (n−1)σ², so dividing by n−1 gives an unbiased s². One degree of freedom is "spent" estimating x̄.</p>
    <h4 style={h4}>Computational form</h4><Formula>Σ(xᵢ−x̄)² = Σxᵢ² − (Σxᵢ)²/n</Formula>
    <p style={pS}>One pass with Σx and Σx². Numerically fragile for large near-equal values — prefer Welford online updates in code.</p></>) },
  { id: "sd", title: "Standard Deviation", blurb: "Empirical rule, SD vs SEM, CV.", render: () => (<>
    <Formula>s = √s²  (back in original units)</Formula>
    <h4 style={h4}>Empirical rule</h4><p style={pS}>≈68% within 1 SD, ≈95% within 2 SD, ≈99.7% within 3 SD (roughly normal data).</p>
    <h4 style={h4}>SD vs SEM</h4><SDSEM /><p style={pS}>SEM is smaller, so authors abuse it to shrink error bars. Describe a sample → SD. Estimate a parameter → SEM/CI.</p>
    <h4 style={h4}>Coefficient of variation</h4><Formula>CV = s/x̄ (×100%)</Formula>
    <p style={pS}>Unitless relative spread — compare an assay in ng/mL with one in cells/µL. Assay reproducibility often wants CV &lt; 15%.</p></>) },
  { id: "ci", title: "Confidence Intervals", blurb: "Z vs t, proportions, what they really mean.", render: () => (<>
    <Formula>x̄ ± (critical value)·SE</Formula>
    <h4 style={h4}>Z vs t</h4><p style={pS}>Z (1.96 at 95%) when σ known / n large. t when σ is estimated from a small sample — heavier tails, wider intervals. t→Z as df grows.</p>
    <Formula>Mean (t): x̄ ± t*₍df=n−1₎ · s/√n</Formula>
    <h4 style={h4}>Proportion (Wald)</h4><Formula>p̂ ± Z*·√(p̂(1−p̂)/n)</Formula>
    <p style={pS}>Poor near 0/1 or small n — use Wilson or Clopper–Pearson there.</p>
    <div style={proofBox}><Eyebrow>Is / isn't</Eyebrow><p style={{ ...pS, margin: "0 0 8px" }}><b>Is:</b> a procedure capturing the true value 95% of the time over many samples.</p><p style={{ ...pS, margin: 0 }}><b>Isn't:</b> "95% probability the value is in this interval." This interval either contains it or not.</p></div>
    <h4 style={h4}>Duality with testing</h4><p style={pS}>A 95% CI for a difference excludes 0 ⟺ the two-sided test rejects at α=0.05 — but the CI also gives the effect size and its precision.</p></>) },
  { id: "cochran", title: "Cochran's Formula", blurb: "Sample size, from the CI you want.", render: () => (<>
    <h4 style={h4}>Derivation (proportion)</h4><Formula>E = Z*·√(p(1−p)/n)</Formula><Formula>→ n₀ = Z²·p(1−p)/E²</Formula>
    <p style={pS}>Use p=0.5 when unknown; it maximizes p(1−p) and gives the safest (largest) n.</p>
    <h4 style={h4}>Finite population correction</h4><Formula>n = n₀ / (1 + (n₀−1)/N)</Formula>
    <p style={pS}>Sampling a big slice of a small population is itself informative, so you need fewer subjects.</p>
    <h4 style={h4}>For a mean</h4><Formula>n = (Z*·σ/E)²</Formula><p style={pS}>σ from a pilot or literature. Same logic, proportion SE swapped for σ/√n.</p></>) },
  { id: "pvalue", title: "P-value, Errors & Power", blurb: "Definition, Type I/II, multiplicity, ASA.", render: () => (<>
    <div style={proofBox}><Eyebrow>Formal definition</Eyebrow><p style={{ ...pS, margin: 0 }}>P(data at least this extreme | H₀ true). It is P(data | H₀), never P(H₀ | data).</p></div>
    <h4 style={h4}>The two errors</h4><ErrTbl />
    <p style={pS}><b>α</b> = P(Type I) false positive rate. <b>β</b> = P(Type II) missing a real effect.</p>
    <h4 style={h4}>Power</h4><Formula>Power = 1 − β = P(reject H₀ | H₁ true)</Formula>
    <p style={pS}>Driven by effect size, n, variability, α. Design for 80–90%. Underpowered studies waste subjects and don't replicate.</p>
    <h4 style={h4}>Multiple testing</h4><Formula>Bonferroni: α_adj = α/m  (family-wise)</Formula><Formula>Benjamini–Hochberg: controls false discovery rate</Formula>
    <div style={{ ...proofBox, borderColor: C.amber }}><Eyebrow>ASA 2016 on misuse</Eyebrow><p style={{ ...pS, margin: 0 }}>p-values don't measure effect size or importance; p&lt;0.05 isn't proof of an effect and p&gt;0.05 isn't proof of none; don't decide on a threshold alone; reporting only tests that "worked" (p-hacking) destroys meaning. Report effect sizes, CIs, and your full plan.</p></div></>) },
];
function SDSEM() {
  const rows = [["Measures", "Spread of data points", "Precision of the mean"], ["Formula", "s = √[Σ(x−x̄)²/(n−1)]", "s/√n"], ["As n grows", "→ true σ", "→ 0"], ["Use when", "Describing a sample", "Estimating the mean"]];
  return (<div style={{ overflowX: "auto", margin: "12px 0" }}><table style={tbl}><thead><tr><th style={th}></th><th style={th}>SD</th><th style={th}>SEM</th></tr></thead><tbody>{rows.map((r, i) => <tr key={i}><td style={{ ...td, fontWeight: 600, color: C.sage }}>{r[0]}</td><td style={td}>{r[1]}</td><td style={td}>{r[2]}</td></tr>)}</tbody></table></div>);
}
function ErrTbl() {
  return (<div style={{ overflowX: "auto", margin: "12px 0" }}><table style={tbl}><thead><tr><th style={th}></th><th style={th}>H₀ true</th><th style={th}>H₀ false</th></tr></thead><tbody>
    <tr><td style={{ ...td, fontWeight: 600, color: C.sage }}>Reject</td><td style={{ ...td, background: "#F6E2D6" }}>Type I (α)</td><td style={{ ...td, background: "#E2EDE6" }}>Power ✓</td></tr>
    <tr><td style={{ ...td, fontWeight: 600, color: C.sage }}>Don't reject</td><td style={{ ...td, background: "#E2EDE6" }}>✓</td><td style={{ ...td, background: "#F6E2D6" }}>Type II (β)</td></tr></tbody></table></div>);
}

/* ============================================================
   CODE EXPORT — R / Python side by side, learner picks lang
   ============================================================ */
function CodePanel({ lang, setLang, r, py }) {
  const [copied, setCopied] = useState(false);
  const code = lang === "r" ? r : py;
  const copy = () => { navigator.clipboard?.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1200); };
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8 }}>
        <Eyebrow>Reproduce this in code</Eyebrow>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          {[["r", "R"], ["py", "Python"]].map(([k, l]) => (
            <button key={k} onClick={() => setLang(k)} style={{ fontFamily: mono, fontSize: 12, padding: "4px 12px", borderRadius: 3, border: `1px solid ${C.line}`, cursor: "pointer", background: lang === k ? C.ink : C.card, color: lang === k ? C.paper : C.ink }}>{l}</button>
          ))}
          <button onClick={copy} style={{ fontFamily: mono, fontSize: 12, padding: "4px 12px", borderRadius: 3, border: `1px solid ${C.line}`, cursor: "pointer", background: C.amber, color: "#fff" }}>{copied ? "copied" : "copy"}</button>
        </div>
      </div>
      <pre style={{ fontFamily: mono, fontSize: 12.5, background: C.ink, color: "#E7E2D4", padding: 16, borderRadius: 4, overflowX: "auto", lineHeight: 1.55, margin: 0 }}>{code}</pre>
    </div>
  );
}

/* ============================================================
   MODULE 2 — ASK (PICOT + FINER)
   ============================================================ */
function QuestionBuilder() {
  const [f, setF] = useState({ p: "", i: "", c: "", o: "", t: "" });
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const q = f.p || f.i || f.o ? `In ${f.p || "[population]"}, does ${f.i || "[intervention]"} compared with ${f.c || "[comparison]"} affect ${f.o || "[outcome]"}${f.t ? ` over ${f.t}` : ""}?` : "";
  const fields = [["p", "Population / Problem", "adults with diabetic foot ulcers"], ["i", "Intervention / Exposure", "topical antimicrobial dressing"], ["c", "Comparison", "standard saline dressing"], ["o", "Outcome", "wound closure at 12 weeks"], ["t", "Time / Setting", "a 12-week single-centre trial"]];
  return (<div>
    <Eyebrow>Module 2 — Frame the question</Eyebrow><h2 style={h2}>Research question builder</h2>
    <p style={lead}>Fill the PICOT slots — each is a design decision you'd otherwise make later, less deliberately.</p>
    <Card>{fields.map(([k, label, ph]) => (<div key={k} style={{ marginBottom: 14 }}><label style={lbl}>{label}</label><input value={f[k]} onChange={set(k)} placeholder={`e.g. ${ph}`} style={inp} /></div>))}</Card>
    {q && <Card style={{ marginTop: 16, background: C.faint, borderColor: C.amber }}><Eyebrow>Your draft question</Eyebrow><p style={{ fontFamily: serif, fontSize: 19, lineHeight: 1.5, margin: 0 }}>{q}</p></Card>}
    <Card style={{ marginTop: 16 }}><Eyebrow>FINER check</Eyebrow>{[["Feasible", "Enough subjects, time, funding, scope?"], ["Interesting", "Does the answer matter to you and the field?"], ["Novel", "Confirms, refutes, or extends prior work?"], ["Ethical", "Approvable by your IRB / ethics board?"], ["Relevant", "Will it change practice or future research?"]].map(([t, d]) => (<div key={t} style={{ display: "flex", gap: 12, padding: "8px 0", borderBottom: `1px solid ${C.faint}` }}><span style={{ fontFamily: mono, fontSize: 10, color: "#fff", background: C.sage, padding: "3px 8px", borderRadius: 3, height: "fit-content" }}>{t[0]}</span><div><b>{t}</b> — <span style={{ color: "#556" }}>{d}</span></div></div>))}</Card>
  </div>);
}

/* ============================================================
   MODULE 3 — ANALYZE (wizard + assumptions + tests + viz + code + report)
   ============================================================ */
function Analyzer({ lang, setLang }) {
  const [a, setA] = useState("128 122 130 119 124 127 121 133 118 125");
  const [b, setB] = useState("138 142 135 145 140 137 144 139 141 136");
  const [design, setDesign] = useState("two"); // one | two | paired | anova | reg | cat
  const [c, setC] = useState("131 129 134 127 132 128 135 130");
  const [xv, setXv] = useState("1 2 3 4 5 6 7 8");
  const [yv, setYv] = useState("2.1 3.9 6.2 7.8 10.1 12.2 13.8 16.1");
  const [tab, setTab] = useState({ ad: 12, ag: 8, bd: 5, bg: 15 }); // 2x2

  const A = parse(a), B = parse(b), Cc = parse(c), X = parse(xv), Y = parse(yv);
  const fileRef = useRef();

  function onFile(e) {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const lines = reader.result.trim().split(/\r?\n/).map((l) => l.split(/[,;\t]/));
      const cols = lines[0].length;
      const c0 = [], c1 = [];
      const start = isNaN(Number(lines[0][0])) ? 1 : 0;
      for (let i = start; i < lines.length; i++) { if (lines[i][0] !== undefined && lines[i][0] !== "") c0.push(lines[i][0]); if (cols > 1 && lines[i][1] !== undefined) c1.push(lines[i][1]); }
      setA(c0.join(" ")); if (cols > 1) setB(c1.join(" "));
    };
    reader.readAsText(file);
  }

  return (<div>
    <Eyebrow>Module 3 — Analyze your data</Eyebrow><h2 style={h2}>Data analysis workbench</h2>
    <p style={lead}>Pick what your data looks like; the right test, its assumptions, the result, a plain-language reading, an effect size, and runnable R/Python code all follow.</p>

    <Card>
      <label style={lbl}>1 · What is your data shape?</label>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
        {[["one", "One group"], ["two", "Two groups (unpaired)"], ["paired", "Paired / before-after"], ["anova", "3+ groups"], ["reg", "X vs Y (regression)"], ["cat", "Counts (2×2 table)"]].map(([k, l]) => (
          <button key={k} onClick={() => setDesign(k)} style={chipBtn(design === k)}>{l}</button>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 10 }}>
        <button onClick={() => fileRef.current?.click()} style={chipBtn(false)}>Upload CSV…</button>
        <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" onChange={onFile} style={{ display: "none" }} />
        <span style={{ fontFamily: mono, fontSize: 11, color: C.sage }}>or paste below (commas / spaces / newlines)</span>
      </div>

      {design === "cat" ? (
        <Cat2x2 tab={tab} setTab={setTab} />
      ) : design === "reg" ? (<>
        <label style={lbl}>X (predictor)</label><textarea value={xv} onChange={(e) => setXv(e.target.value)} style={ta} rows={2} />
        <label style={lbl}>Y (outcome)</label><textarea value={yv} onChange={(e) => setYv(e.target.value)} style={ta} rows={2} />
      </>) : (<>
        <label style={lbl}>{design === "one" ? "Your data" : "Group A"}</label>
        <textarea value={a} onChange={(e) => setA(e.target.value)} style={ta} rows={2} />
        {(design === "two" || design === "paired" || design === "anova") && (<>
          <label style={lbl}>Group B</label><textarea value={b} onChange={(e) => setB(e.target.value)} style={ta} rows={2} />
        </>)}
        {design === "anova" && (<>
          <label style={lbl}>Group C</label><textarea value={c} onChange={(e) => setC(e.target.value)} style={ta} rows={2} />
        </>)}
      </>)}
    </Card>

    {design === "one" && <OneGroup a={A} lang={lang} setLang={setLang} />}
    {design === "two" && <TwoGroup a={A} b={B} lang={lang} setLang={setLang} />}
    {design === "paired" && <Paired a={A} b={B} lang={lang} setLang={setLang} />}
    {design === "anova" && <Anova groups={[A, B, Cc].filter((g) => g.length > 1)} lang={lang} setLang={setLang} />}
    {design === "reg" && <Regression x={X} y={Y} lang={lang} setLang={setLang} />}
    {design === "cat" && <CatResult tab={tab} lang={lang} setLang={setLang} />}

    <Card style={{ marginTop: 16, background: C.faint }}>
      <Eyebrow>How this was analyzed</Eyebrow>
      <ol style={{ margin: 0, paddingLeft: 20, color: "#445", lineHeight: 1.7, fontSize: 14 }}>
        <li>Descriptives use the sample (n−1) formulas from Module 1.</li>
        <li>Intervals use an exact <b>Student-t</b> critical value (incomplete-beta), so they match R / Prism.</li>
        <li>Two unpaired groups use <Term>Welch</Term>'s t-test (no equal-variance assumption — the safer default).</li>
        <li>Every test reports an effect size, because significance ≠ importance.</li>
        <li>The assumptions panel runs <i>before</i> you trust the result.</li>
      </ol>
    </Card>
  </div>);
}

function chipBtn(active) {
  return { fontFamily: mono, fontSize: 12.5, padding: "7px 13px", borderRadius: 3, border: `1px solid ${active ? C.ink : C.line}`, cursor: "pointer", background: active ? C.ink : C.card, color: active ? C.paper : C.ink };
}

/* assumptions panel */
function Assumptions({ groups }) {
  const checks = groups.map((g, i) => {
    const sw = shapiroApprox(g);
    const out = outliers(g);
    return { i, sw, out, n: g.length };
  });
  const eqVar = groups.length === 2 ? (() => {
    const r = variance(groups[0]) / variance(groups[1]);
    const ratio = r < 1 ? 1 / r : r;
    return ratio;
  })() : null;
  return (
    <Card style={{ marginTop: 16, borderColor: C.sage }}>
      <Eyebrow>Check assumptions first</Eyebrow>
      {checks.map((c) => (
        <div key={c.i} style={{ display: "flex", gap: 10, alignItems: "center", padding: "6px 0", borderBottom: `1px solid ${C.faint}`, fontSize: 13.5 }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: C.sage, minWidth: 64 }}>Group {String.fromCharCode(65 + c.i)}</span>
          {c.sw ? <Badge ok={c.sw.normalish}>{c.sw.normalish ? "≈ normal" : "non-normal?"} (W={fmt(c.sw.W, 2)})</Badge> : <Badge ok={true}>n too small to test</Badge>}
          <Badge ok={c.out.length === 0}>{c.out.length === 0 ? "no extreme outliers" : `${c.out.length} possible outlier(s)`}</Badge>
        </div>
      ))}
      {eqVar && <div style={{ marginTop: 8, fontSize: 13.5 }}><Badge ok={eqVar < 3}>variance ratio ≈ {fmt(eqVar, 1)}× {eqVar < 3 ? "(comparable)" : "(unequal — Welch handles it)"}</Badge></div>}
      <p style={{ ...pS, fontSize: 12.5, marginTop: 10, marginBottom: 0 }}>If groups look non-normal or tiny, prefer a transformation (e.g. log for titres) or a non-parametric test (Mann–Whitney / Wilcoxon). Outliers flagged at 1.5×IQR — investigate, don't auto-delete.</p>
    </Card>
  );
}
function outliers(a) {
  const q1 = quantile(a, 0.25), q3 = quantile(a, 0.75), iqr = q3 - q1;
  return a.filter((x) => x < q1 - 1.5 * iqr || x > q3 + 1.5 * iqr);
}
const Badge = ({ ok, children }) => (
  <span style={{ fontFamily: mono, fontSize: 11, padding: "3px 9px", borderRadius: 10, background: ok ? "#E2EDE6" : "#F6E2D6", color: ok ? C.ok : C.warn }}>{children}</span>
);

/* one group */
function OneGroup({ a, lang, setLang }) {
  if (a.length < 2) return <Note>Enter at least 2 numbers.</Note>;
  const m = mean(a), s = sd(a), n = a.length, sem = s / Math.sqrt(n), tc = tCrit(n - 1);
  const lo = m - tc * sem, hi = m + tc * sem, cv = (s / m) * 100;
  const gm = a.every((x) => x > 0) ? geomean(a) : null;
  const r = `x <- c(${a.join(", ")})\nmean(x); sd(x)\nt.test(x)            # 95% CI for the mean`;
  const py = `import numpy as np, scipy.stats as st\nx = np.array([${a.join(", ")}])\nprint(x.mean(), x.std(ddof=1))\nprint(st.t.interval(0.95, len(x)-1, x.mean(), st.sem(x)))`;
  return (<>
    <Assumptions groups={[a]} />
    <Card style={{ marginTop: 16 }}>
      <Eyebrow>Descriptive statistics</Eyebrow>
      <div style={grid}><Stat label="n" v={n} /><Stat label="Mean" v={fmt(m)} /><Stat label="Median" v={fmt(median(a))} /><Stat label="SD" v={fmt(s)} hint={GLOSSARY.SD} /><Stat label="SEM" v={fmt(sem)} hint={GLOSSARY.SEM} /><Stat label="CV %" v={fmt(cv, 1)} />{gm && <Stat label="Geo.mean" v={fmt(gm)} />}</div>
      <Formula>95% CI for mean = [{fmt(lo)}, {fmt(hi)}] (t*={fmt(tc, 3)}, df={n - 1})</Formula>
      <Interpret>Average <b>{fmt(m)}</b>; values scatter ±{fmt(s)} (1 SD). We're 95% confident the true mean is between {fmt(lo)} and {fmt(hi)}. {cv > 30 ? "CV>30% — high relative variability; check for skew/outliers." : "CV indicates fairly consistent measurements."}</Interpret>
      <CodePanel lang={lang} setLang={setLang} r={r} py={py} />
      <ReportLine text={`Mean ${fmt(m)} (SD ${fmt(s)}, n=${n}); 95% CI ${fmt(lo)}–${fmt(hi)}.`} />
    </Card>
    <Hist a={a} />
  </>);
}

/* two group */
function TwoGroup({ a, b, lang, setLang }) {
  if (a.length < 2 || b.length < 2) return <Note>Enter at least 2 numbers per group.</Note>;
  const w = welch(a, b), tc = tCrit(w.df), lo = w.diff - tc * w.se, hi = w.diff + tc * w.se;
  const r = `a <- c(${a.join(", ")})\nb <- c(${b.join(", ")})\nt.test(a, b)                    # Welch by default\nlibrary(effsize); cohen.d(a, b) # effect size`;
  const py = `import numpy as np, scipy.stats as st\na = np.array([${a.join(", ")}]); b = np.array([${b.join(", ")}])\nt, p = st.ttest_ind(a, b, equal_var=False)\nprint(f"t={t:.3f} p={p:.4f}")`;
  return (<>
    <Assumptions groups={[a, b]} />
    <Card style={{ marginTop: 16, borderColor: C.amber }}>
      <Eyebrow>Two-group comparison — <Term>Welch</Term>'s t-test</Eyebrow>
      <div style={grid}><Stat label="Mean A" v={fmt(w.ma)} /><Stat label="Mean B" v={fmt(w.mb)} /><Stat label="Diff" v={fmt(w.diff)} color={C.amber} /><Stat label="t" v={fmt(w.t)} /><Stat label="df" v={fmt(w.df, 1)} /><Stat label="p" v={fmtP(w.p)} color={w.p < 0.05 ? C.ok : C.ink} hint={GLOSSARY["p-value"]} /><Stat label="Cohen's d" v={fmt(w.d)} hint={GLOSSARY["Cohen's d"]} /></div>
      <Formula>95% CI for the difference = [{fmt(lo)}, {fmt(hi)}]</Formula>
      <DiffNumberLine lo={lo} hi={hi} diff={w.diff} />
      <Interpret>A's mean is {fmt(Math.abs(w.diff))} {w.diff < 0 ? "lower" : "higher"} than B's. {lo * hi > 0 ? <>The CI <b>excludes 0</b> → significant at α=0.05 (p={fmtP(w.p)}).</> : <>The CI <b>includes 0</b> → can't rule out no difference (p={fmtP(w.p)}); possibly underpowered, not proof of equality.</>} Effect size d={fmt(w.d)} ({Math.abs(w.d) < 0.2 ? "negligible" : Math.abs(w.d) < 0.5 ? "small" : Math.abs(w.d) < 0.8 ? "medium" : "large"}). Significance ≠ importance — judge the {fmt(Math.abs(w.diff), 1)}-unit difference clinically.</Interpret>
      <CodePanel lang={lang} setLang={setLang} r={r} py={py} />
      <ReportLine text={`Group A (M=${fmt(w.ma)}) vs B (M=${fmt(w.mb)}): mean difference ${fmt(w.diff)} (95% CI ${fmt(lo)} to ${fmt(hi)}), Welch t(${fmt(w.df, 1)})=${fmt(w.t)}, p=${fmtP(w.p)}, Cohen's d=${fmt(w.d)}.`} />
    </Card>
    <Hist2 a={a} b={b} />
  </>);
}

/* paired */
function Paired({ a, b, lang, setLang }) {
  if (a.length < 2 || a.length !== b.length) return <Note>Paired data needs equal-length groups (one row per subject).</Note>;
  const p = pairedT(a, b), tc = tCrit(p.df), lo = p.md - tc * p.se, hi = p.md + tc * p.se;
  const r = `before <- c(${a.join(", ")})\nafter  <- c(${b.join(", ")})\nt.test(before, after, paired = TRUE)`;
  const py = `import numpy as np, scipy.stats as st\nbefore=np.array([${a.join(", ")}]); after=np.array([${b.join(", ")}])\nprint(st.ttest_rel(before, after))`;
  return (<Card style={{ marginTop: 16, borderColor: C.amber }}>
    <Eyebrow>Paired t-test (within-subject)</Eyebrow>
    <div style={grid}><Stat label="Mean Δ" v={fmt(p.md)} color={C.amber} /><Stat label="SD Δ" v={fmt(p.s)} /><Stat label="n pairs" v={p.n} /><Stat label="t" v={fmt(p.t)} /><Stat label="df" v={p.df} /><Stat label="p" v={fmtP(p.p)} color={p.p < 0.05 ? C.ok : C.ink} /><Stat label="d_z" v={fmt(p.dz)} /></div>
    <Formula>95% CI for the mean change = [{fmt(lo)}, {fmt(hi)}]</Formula>
    <Interpret>Mean within-subject change is {fmt(p.md)}. {lo * hi > 0 ? <>CI excludes 0 → significant (p={fmtP(p.p)}).</> : <>CI includes 0 → not significant (p={fmtP(p.p)}).</>} Pairing removes between-subject variation, which is why it's more powerful than an unpaired test when subjects are their own controls.</Interpret>
    <CodePanel lang={lang} setLang={setLang} r={r} py={py} />
    <ReportLine text={`Mean change ${fmt(p.md)} (95% CI ${fmt(lo)}–${fmt(hi)}), paired t(${p.df})=${fmt(p.t)}, p=${fmtP(p.p)}.`} />
  </Card>);
}

/* anova */
function Anova({ groups, lang, setLang }) {
  if (groups.length < 2) return <Note>Enter at least two groups.</Note>;
  const r = anova(groups);
  const Rcode = `vals <- c(${groups.map((g) => g.join(", ")).join(",  ")})\ngrp  <- factor(rep(c(${groups.map((_, i) => `"${String.fromCharCode(65 + i)}"`).join(", ")}), c(${groups.map((g) => g.length).join(", ")})))\nsummary(aov(vals ~ grp))\nTukeyHSD(aov(vals ~ grp))   # which pairs differ`;
  const py = `import numpy as np, scipy.stats as st\ngroups=[${groups.map((g) => `[${g.join(", ")}]`).join(", ")}]\nprint(st.f_oneway(*[np.array(g) for g in groups]))`;
  return (<Card style={{ marginTop: 16, borderColor: C.amber }}>
    <Eyebrow>One-way ANOVA ({r.k} groups)</Eyebrow>
    <div style={grid}><Stat label="F" v={fmt(r.F)} /><Stat label="df between" v={r.dfb} /><Stat label="df within" v={r.dfw} /><Stat label="p" v={fmtP(r.p)} color={r.p < 0.05 ? C.ok : C.ink} /><Stat label="η²" v={fmt(r.eta2, 3)} /></div>
    <Interpret>ANOVA asks whether <i>any</i> group mean differs. {r.p < 0.05 ? <>p={fmtP(r.p)} → at least one group differs. Follow with a post-hoc test (Tukey HSD) to find which pairs; η²={fmt(r.eta2, 2)} of variance is explained by group.</> : <>p={fmtP(r.p)} → no significant difference among the groups.</>} ANOVA replaces running many t-tests, which would inflate the false-positive rate.</Interpret>
    <CodePanel lang={lang} setLang={setLang} r={Rcode} py={py} />
    <ReportLine text={`One-way ANOVA: F(${r.dfb},${r.dfw})=${fmt(r.F)}, p=${fmtP(r.p)}, η²=${fmt(r.eta2, 2)}.`} />
  </Card>);
}

/* regression */
function Regression({ x, y, lang, setLang }) {
  if (x.length < 3 || x.length !== y.length) return <Note>Regression needs equal-length X and Y, n≥3.</Note>;
  const m = linreg(x, y);
  const r = `x <- c(${x.join(", ")}); y <- c(${y.join(", ")})\nfit <- lm(y ~ x); summary(fit)`;
  const py = `import numpy as np, scipy.stats as st\nx=np.array([${x.join(", ")}]); y=np.array([${y.join(", ")}])\nprint(st.linregress(x, y))`;
  return (<Card style={{ marginTop: 16, borderColor: C.amber }}>
    <Eyebrow>Linear regression  y = a + b·x</Eyebrow>
    <div style={grid}><Stat label="Slope" v={fmt(m.slope, 3)} color={C.amber} /><Stat label="Intercept" v={fmt(m.intercept, 3)} /><Stat label="r" v={fmt(m.r, 3)} /><Stat label="R²" v={fmt(m.r2, 3)} /><Stat label="p (slope)" v={fmtP(m.p)} color={m.p < 0.05 ? C.ok : C.ink} /></div>
    <Scatter x={x} y={y} m={m} />
    <Interpret>Each 1-unit rise in X changes Y by {fmt(m.slope, 2)}. R²={fmt(m.r2, 2)} → {fmt(m.r2 * 100, 0)}% of Y's variation is explained by X. {m.p < 0.05 ? "The slope differs significantly from 0." : "The slope is not significantly different from 0."} Correlation isn't causation — check confounders.</Interpret>
    <CodePanel lang={lang} setLang={setLang} r={r} py={py} />
    <ReportLine text={`y = ${fmt(m.intercept, 2)} + ${fmt(m.slope, 2)}·x; R²=${fmt(m.r2, 2)}, p=${fmtP(m.p)}.`} />
  </Card>);
}

/* categorical 2x2 */
function Cat2x2({ tab, setTab }) {
  const cell = (k) => (<input type="number" value={tab[k]} onChange={(e) => setTab({ ...tab, [k]: Number(e.target.value) })} style={{ ...inp, width: 80, textAlign: "center", marginBottom: 0 }} />);
  return (<div style={{ marginTop: 8 }}>
    <table style={{ borderCollapse: "collapse" }}><tbody>
      <tr><td style={cellTd}></td><td style={{ ...cellTd, fontFamily: mono, fontSize: 11, color: C.sage }}>Outcome +</td><td style={{ ...cellTd, fontFamily: mono, fontSize: 11, color: C.sage }}>Outcome −</td></tr>
      <tr><td style={{ ...cellTd, fontFamily: mono, fontSize: 11, color: C.sage }}>Group A</td><td style={cellTd}>{cell("ad")}</td><td style={cellTd}>{cell("ag")}</td></tr>
      <tr><td style={{ ...cellTd, fontFamily: mono, fontSize: 11, color: C.sage }}>Group B</td><td style={cellTd}>{cell("bd")}</td><td style={cellTd}>{cell("bg")}</td></tr>
    </tbody></table>
  </div>);
}
const cellTd = { padding: 5, border: `1px solid ${C.line}` };
function CatResult({ tab, lang, setLang }) {
  const t = [[tab.ad, tab.ag], [tab.bd, tab.bg]];
  const r = chiSq(t);
  const a = tab.ad, bb = tab.ag, cc = tab.bd, d = tab.bg;
  const rrA = a / (a + bb), rrB = cc / (cc + d);
  const RR = rrA / rrB, OR = (a * d) / (bb * cc);
  const Rcode = `tbl <- matrix(c(${tab.ad}, ${tab.ag}, ${tab.bd}, ${tab.bg}), nrow=2, byrow=TRUE)\nchisq.test(tbl)\nfisher.test(tbl)   # exact, for small counts`;
  const py = `import numpy as np, scipy.stats as st\ntbl=np.array([[${tab.ad},${tab.ag}],[${tab.bd},${tab.bg}]])\nprint(st.chi2_contingency(tbl))\nprint(st.fisher_exact(tbl))`;
  const small = [a, bb, cc, d].some((v) => v < 5);
  return (<Card style={{ marginTop: 16, borderColor: C.amber }}>
    <Eyebrow>2×2 categorical — χ² test</Eyebrow>
    <div style={grid}><Stat label="χ²" v={fmt(r.chi)} /><Stat label="df" v={r.df} /><Stat label="p" v={fmtP(r.p)} color={r.p < 0.05 ? C.ok : C.ink} /><Stat label="Risk ratio" v={fmt(RR)} color={C.amber} /><Stat label="Odds ratio" v={fmt(OR)} /></div>
    <Interpret>Risk in A is {fmt(rrA * 100, 0)}% vs {fmt(rrB * 100, 0)}% in B → RR={fmt(RR)}. {r.p < 0.05 ? "The association is statistically significant." : "No significant association."} {small && <b> Some cell counts &lt;5 — use Fisher's exact test instead of χ².</b>}</Interpret>
    <CodePanel lang={lang} setLang={setLang} r={Rcode} py={py} />
    <ReportLine text={`RR=${fmt(RR)}, OR=${fmt(OR)}, χ²(${r.df})=${fmt(r.chi)}, p=${fmtP(r.p)}.`} />
  </Card>);
}

/* report line */
function ReportLine({ text }) {
  const [copied, setCopied] = useState(false);
  return (<div style={{ marginTop: 14, background: C.faint, border: `1px dashed ${C.sage}`, borderRadius: 4, padding: "12px 14px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
      <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", color: C.sage }}>Results sentence (journal style)</span>
      <button onClick={() => { navigator.clipboard?.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1200); }} style={{ fontFamily: mono, fontSize: 11, padding: "3px 10px", border: `1px solid ${C.line}`, borderRadius: 3, background: C.card, cursor: "pointer" }}>{copied ? "copied" : "copy"}</button>
    </div>
    <div style={{ fontFamily: serif, fontSize: 15, color: C.ink, lineHeight: 1.5 }}>{text}</div>
  </div>);
}
function Note({ children }) { return <Card style={{ marginTop: 16, color: C.warn, fontFamily: mono, fontSize: 13 }}>{children}</Card>; }

/* visualizations */
function Hist({ a }) {
  const m = mean(a), s = sd(a);
  return (<Card style={{ marginTop: 16 }}><Eyebrow>Distribution (dots = values, bar = ±1 SD, line = mean)</Eyebrow><Track data={a} color={C.slate} name="data" /></Card>);
}
function Hist2({ a, b }) {
  return (<Card style={{ marginTop: 16 }}><Eyebrow>Distributions</Eyebrow><Track data={a} color={C.slate} name="Group A" all={[...a, ...b]} /><Track data={b} color={C.amber} name="Group B" all={[...a, ...b]} /></Card>);
}
function Track({ data, color, name, all }) {
  const pool = all || data;
  const min = Math.min(...pool), max = Math.max(...pool), range = max - min || 1;
  const pos = (x) => ((x - min) / range) * 100;
  const m = mean(data), s = sd(data);
  return (<div style={{ marginBottom: 16 }}>
    <div style={{ fontFamily: mono, fontSize: 11, color, marginBottom: 4 }}>{name} (x̄={fmt(m, 1)})</div>
    <div style={{ position: "relative", height: 34, background: C.faint, borderRadius: 3 }}>
      <div style={{ position: "absolute", left: `${pos(m - s)}%`, width: `${(s * 2 / range) * 100}%`, top: 12, height: 10, background: color, opacity: 0.22, borderRadius: 2 }} />
      {data.map((x, i) => <div key={i} title={x} style={{ position: "absolute", left: `${pos(x)}%`, top: 11, width: 7, height: 7, marginLeft: -3.5, borderRadius: "50%", background: color }} />)}
      <div style={{ position: "absolute", left: `${pos(m)}%`, top: 4, width: 2, height: 26, marginLeft: -1, background: color }} />
    </div>
  </div>);
}
function DiffNumberLine({ lo, hi, diff }) {
  const span = Math.max(Math.abs(lo), Math.abs(hi)) * 1.3 || 1;
  const pos = (x) => 50 + (x / span) * 50;
  return (<div style={{ margin: "12px 0 4px" }}>
    <div style={{ fontFamily: mono, fontSize: 10, color: C.sage, marginBottom: 6 }}>95% CI FOR THE DIFFERENCE (vertical line = no effect)</div>
    <div style={{ position: "relative", height: 40 }}>
      <div style={{ position: "absolute", top: 20, left: 0, right: 0, height: 1, background: C.line }} />
      <div style={{ position: "absolute", top: 6, left: "50%", width: 2, height: 28, marginLeft: -1, background: C.ink }} />
      <div style={{ position: "absolute", top: 18, left: `${pos(lo)}%`, width: `${pos(hi) - pos(lo)}%`, height: 5, background: C.amber, borderRadius: 3, opacity: 0.6 }} />
      <div style={{ position: "absolute", top: 14, left: `${pos(diff)}%`, width: 11, height: 11, marginLeft: -5.5, borderRadius: "50%", background: C.amber }} />
    </div>
    <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 11, color: "#667" }}><span>{fmt(lo)}</span><span>0</span><span>{fmt(hi)}</span></div>
  </div>);
}
function Scatter({ x, y, m }) {
  const W = 300, H = 160, pad = 28;
  const xmin = Math.min(...x), xmax = Math.max(...x), ymin = Math.min(...y), ymax = Math.max(...y);
  const sx = (v) => pad + ((v - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const sy = (v) => H - pad - ((v - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);
  return (<svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 340, margin: "10px 0" }}>
    <line x1={sx(xmin)} y1={sy(m.slope * xmin + m.intercept)} x2={sx(xmax)} y2={sy(m.slope * xmax + m.intercept)} stroke={C.amber} strokeWidth="2" />
    {x.map((xi, i) => <circle key={i} cx={sx(xi)} cy={sy(y[i])} r="3.5" fill={C.slate} />)}
  </svg>);
}

/* ============================================================
   POWER / SAMPLE-SIZE
   ============================================================ */
function PowerLab() {
  const [d, setD] = useState(0.8), [alpha, setAlpha] = useState(0.05), [nTarget, setNTarget] = useState(0.8);
  const pow = (n) => powerTwoSample(d, n, alpha);
  // n needed for target power
  let nNeed = 2; while (pow(nNeed) < nTarget && nNeed < 5000) nNeed++;
  const pts = []; for (let n = 2; n <= 120; n += 2) pts.push([n, pow(n)]);
  const W = 560, H = 220, pad = 36;
  const sx = (n) => pad + (n / 120) * (W - 2 * pad);
  const sy = (p) => H - pad - p * (H - 2 * pad);
  return (<div>
    <Eyebrow>Power & sample size</Eyebrow><h2 style={h2}>How many subjects do I need?</h2>
    <p style={lead}>For a two-sample comparison. Drag the inputs; the curve and required n update live. Built on the Cochran/power logic from Module 1.</p>
    <Card>
      <Slider label={`Effect size (Cohen's d) = ${d.toFixed(2)}`} min={0.1} max={2} step={0.05} value={d} onChange={setD} />
      <Slider label={`α = ${alpha.toFixed(3)}`} min={0.01} max={0.1} step={0.005} value={alpha} onChange={setAlpha} />
      <Slider label={`Target power = ${(nTarget * 100).toFixed(0)}%`} min={0.5} max={0.99} step={0.01} value={nTarget} onChange={setNTarget} />
      <div style={{ ...grid, marginTop: 14 }}><Stat label="n per group" v={nNeed} color={C.amber} /><Stat label="total N" v={nNeed * 2} /><Stat label="power @ that n" v={`${(pow(nNeed) * 100).toFixed(0)}%`} /></div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", marginTop: 14 }}>
        <line x1={pad} y1={sy(nTarget)} x2={W - pad} y2={sy(nTarget)} stroke={C.line} strokeDasharray="4 4" />
        <line x1={sx(nNeed)} y1={pad} x2={sx(nNeed)} y2={H - pad} stroke={C.amber} strokeDasharray="3 3" />
        <polyline fill="none" stroke={C.slate} strokeWidth="2.5" points={pts.map(([n, p]) => `${sx(n)},${sy(p)}`).join(" ")} />
        <text x={pad} y={H - 8} fontFamily={mono} fontSize="10" fill={C.sage}>n per group →</text>
        <text x={pad} y={sy(nTarget) - 5} fontFamily={mono} fontSize="10" fill={C.sage}>target</text>
      </svg>
      <Interpret>With d={d.toFixed(2)} and α={alpha.toFixed(3)}, you need <b>{nNeed} per group</b> ({nNeed * 2} total) for {(nTarget * 100).toFixed(0)}% power. Smaller expected effects demand sharply larger samples — the curve flattens, so chasing tiny effects gets expensive fast.</Interpret>
    </Card>
  </div>);
}
function Slider({ label, min, max, step, value, onChange }) {
  return (<div style={{ marginBottom: 12 }}>
    <label style={lbl}>{label}</label>
    <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ width: "100%", accentColor: C.amber }} />
  </div>);
}

/* ============================================================
   QUIZ — spot the error
   ============================================================ */
const QUIZ = [
  { q: "A paper reports 'mean ± SEM' error bars to show how spread out the patients are.", bad: true, why: "SEM measures precision of the mean, not spread of patients. Describing variability needs SD. SEM is smaller, so this misleadingly tightens the bars." },
  { q: "Authors tested 18 outcomes at α=0.05 and highlight the 2 that reached p<0.05.", bad: true, why: "With 18 tests you expect ≈1 false positive by chance. Without correction (Bonferroni/BH) and a pre-registered plan, this is p-hacking." },
  { q: "A 95% CI for a difference is [2.1, 8.4] mmHg, reported as a significant effect.", bad: false, why: "Correct: the CI excludes 0, so it's significant at α=0.05 — and it also communicates the effect size and precision." },
  { q: "p=0.07, so the authors conclude the treatment has no effect.", bad: true, why: "p>0.05 is not proof of no effect — it may be underpowered. 'Absence of evidence ≠ evidence of absence.' Report the CI." },
  { q: "Titres (2, 4, 8, 16, 256) summarized with a geometric mean.", bad: false, why: "Correct: titres are multiplicative/log-normal, so the geometric mean is the honest center; an arithmetic mean would be dragged up by 256." },
  { q: "n=4 per group, normality untested, analyzed with a Student t-test assuming equal variance.", bad: true, why: "Tiny n with no assumption check. Prefer Welch's t (no equal-variance assumption) and consider a non-parametric test or stating the limitation." },
];
function Quiz() {
  const [i, setI] = useState(0), [picked, setPicked] = useState(null), [score, setScore] = useState(0), [seen, setSeen] = useState(0);
  const item = QUIZ[i];
  const answer = (guessBad) => { if (picked !== null) return; setPicked(guessBad); if (guessBad === item.bad) setScore(score + 1); setSeen(seen + 1); };
  const next = () => { setPicked(null); setI((i + 1) % QUIZ.length); };
  const correct = picked !== null && picked === item.bad;
  return (<div>
    <Eyebrow>Quiz — spot the error</Eyebrow><h2 style={h2}>Is the statistics sound?</h2>
    <p style={lead}>Real reporting habits. Decide whether each is legitimate or a misuse. Score: {score}/{seen}.</p>
    <Card>
      <div style={{ fontFamily: serif, fontSize: 19, lineHeight: 1.5, marginBottom: 18 }}>{item.q}</div>
      <div style={{ display: "flex", gap: 10 }}>
        <button onClick={() => answer(false)} disabled={picked !== null} style={quizBtn(picked !== null && !item.bad)}>Sound ✓</button>
        <button onClick={() => answer(true)} disabled={picked !== null} style={quizBtn(picked !== null && item.bad)}>Misuse ✗</button>
      </div>
      {picked !== null && (<div style={{ marginTop: 16 }}>
        <Badge ok={correct}>{correct ? "correct" : "not quite"}</Badge>
        <p style={{ ...pS, marginTop: 10 }}>{item.why}</p>
        <button onClick={next} style={{ ...chipBtn(true), marginTop: 4 }}>Next →</button>
      </div>)}
    </Card>
  </div>);
}
function quizBtn(highlight) {
  return { flex: 1, fontFamily: mono, fontSize: 14, padding: "14px", borderRadius: 4, border: `1px solid ${highlight ? C.amber : C.line}`, background: highlight ? "#F6E2D6" : C.card, cursor: "pointer" };
}

/* ============================================================
   WORKED EXAMPLE + REFERENCE
   ============================================================ */
function Worked() {
  const t = [128, 122, 130, 119, 124, 127, 121, 133, 118, 125, 123, 129];
  const c = [138, 142, 135, 145, 140, 137, 144, 139, 141, 136, 143, 140];
  const w = welch(t, c), tc = tCrit(w.df), lo = w.diff - tc * w.se, hi = w.diff + tc * w.se;
  return (<Card style={{ marginTop: 20 }}>
    <Eyebrow>Integrated worked example</Eyebrow>
    <h3 style={{ fontFamily: serif, fontSize: 22, margin: "0 0 6px" }}>A two-arm RCT — all six concepts at once</h3>
    <p style={pS}>A drug's effect on systolic BP, 12 per arm.</p>
    <Formula>Treatment: {t.join(", ")}</Formula><Formula>Control:&nbsp;&nbsp; {c.join(", ")}</Formula>
    <h4 style={h4}>1 · Means</h4><p style={pS}>Treatment {fmt(w.ma)}, Control {fmt(w.mb)}; difference <b>{fmt(w.diff)} mmHg</b>.</p>
    <h4 style={h4}>2–3 · Variance & SD</h4><p style={pS}>s(treat)={fmt(Math.sqrt(w.va))}, s(control)={fmt(Math.sqrt(w.vb))}, both n−1. SEM(treat)={fmt(Math.sqrt(w.va) / Math.sqrt(w.na))}.</p>
    <h4 style={h4}>4 · CI for the difference</h4><Formula>= {fmt(w.diff)} ± {fmt(tc, 3)}·{fmt(w.se, 3)} = [{fmt(lo)}, {fmt(hi)}] mmHg</Formula>
    <h4 style={h4}>5 · P-value</h4><p style={pS}>Welch t={fmt(w.t)}, p={fmtP(w.p)} — consistent with the CI excluding 0.</p>
    <h4 style={h4}>6 · Cochran sanity check</h4><p style={pS}>To estimate each arm within E=2 mmHg at 95%: n=(1.96·σ/E)² ≈ <b>{Math.ceil(((1.96 * Math.sqrt((w.va + w.vb) / 2)) / 2) ** 2)}</b> per arm.</p>
    <div style={{ ...proofBox, borderColor: C.sage }}><Eyebrow>Together</Eyebrow><p style={{ ...pS, margin: 0 }}>The drug lowered SBP by ≈{fmt(Math.abs(w.diff), 0)} mmHg (95% CI {fmt(lo, 1)} to {fmt(hi, 1)}), p={fmtP(w.p)}, Cohen's d={fmt(w.d)}. The CI gives the <i>size</i> of the benefit the p-value alone never could.</p></div>
  </Card>);
}
function SummaryTable() {
  const rows = [["Arithmetic mean", "(1/n)Σx", "Symmetric data"], ["Geometric mean", "(Πx)^(1/n)", "Multiplicative / log-normal"], ["Sample variance", "Σ(x−x̄)²/(n−1)", "Spread; n−1 unbiased"], ["SD", "√s²", "Spread, original units"], ["SEM", "s/√n", "Precision of the mean"], ["CV", "s/x̄", "Relative variability"], ["CI (mean)", "x̄ ± t*·s/√n", "Plausible parameter range"], ["Cochran n₀", "Z²p(1−p)/E²", "Sample size, proportion"], ["Cohen's d", "(x̄₁−x̄₂)/s_p", "Standardized effect size"], ["Power", "1 − β", "Detecting a real effect"]];
  return (<Card style={{ marginTop: 20 }}><Eyebrow>Quick-reference summary</Eyebrow><div style={{ overflowX: "auto" }}><table style={tbl}><thead><tr><th style={th}>Quantity</th><th style={th}>Formula</th><th style={th}>Use for</th></tr></thead><tbody>{rows.map((r) => <tr key={r[0]}><td style={{ ...td, fontWeight: 600 }}>{r[0]}</td><td style={{ ...td, fontFamily: mono, color: C.slate }}>{r[1]}</td><td style={td}>{r[2]}</td></tr>)}</tbody></table></div></Card>);
}
function RefTables() {
  const z = [["80%", "1.282"], ["90%", "1.645"], ["95%", "1.960"], ["98%", "2.326"], ["99%", "2.576"], ["99.9%", "3.291"]];
  const tt = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042 };
  return (<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(260px,1fr))", gap: 16, marginTop: 20 }}>
    <Card><Eyebrow>z* critical values</Eyebrow><table style={tbl}><thead><tr><th style={th}>Confidence</th><th style={th}>z*</th></tr></thead><tbody>{z.map((r) => <tr key={r[0]}><td style={td}>{r[0]}</td><td style={{ ...td, fontFamily: mono, color: C.amber }}>{r[1]}</td></tr>)}</tbody></table></Card>
    <Card><Eyebrow>t* — 95% two-sided</Eyebrow><div style={{ maxHeight: 280, overflowY: "auto" }}><table style={tbl}><thead><tr><th style={th}>df</th><th style={th}>t*</th></tr></thead><tbody>{Object.entries(tt).map(([df, v]) => <tr key={df}><td style={td}>{df}</td><td style={{ ...td, fontFamily: mono, color: C.amber }}>{v.toFixed(3)}</td></tr>)}<tr><td style={td}>∞</td><td style={{ ...td, fontFamily: mono, color: C.amber }}>1.960</td></tr></tbody></table></div></Card>
  </div>);
}

/* ============================================================
   ROOT
   ============================================================ */
export default function App() {
  const [tab, setTab] = useState("learn");
  const [open, setOpen] = useState("mean");
  const [lang, setLang] = useState("r");

  useEffect(() => {
    if (!document.getElementById("cs-fonts")) {
      const l = document.createElement("link"); l.id = "cs-fonts"; l.rel = "stylesheet";
      l.href = "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap";
      document.head.appendChild(l);
    }
  }, []);
  const tabs = [["learn", "Learn"], ["question", "Ask"], ["analyze", "Analyze"], ["power", "Power"], ["quiz", "Quiz"], ["reference", "Reference"]];

  return (<div style={{ background: C.paper, minHeight: "100vh", fontFamily: sans, color: C.ink }}>
    <style>{`*{box-sizing:border-box}input:focus,textarea:focus{outline:2px solid ${C.amber};outline-offset:1px}::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-thumb{background:${C.line};border-radius:5px}`}</style>
    <header style={{ background: C.ink, color: C.paper }}>
      <div style={{ maxWidth: 940, margin: "0 auto", padding: "26px 24px 0" }}>
        <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: C.amber }}>Biostatistics, for people who do the experiments</div>
        <h1 style={{ fontFamily: serif, fontSize: 40, fontWeight: 600, margin: "6px 0 18px", lineHeight: 1.05 }}>ClariStat</h1>
        <nav style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>{tabs.map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={{ fontFamily: mono, fontSize: 13, letterSpacing: 1, padding: "10px 16px", border: "none", borderRadius: "4px 4px 0 0", cursor: "pointer", background: tab === id ? C.paper : "transparent", color: tab === id ? C.ink : C.paper, fontWeight: tab === id ? 600 : 400, borderBottom: tab === id ? `2px solid ${C.amber}` : "2px solid transparent" }}>{label}</button>
        ))}</nav>
      </div>
    </header>

    <main style={{ maxWidth: 940, margin: "0 auto", padding: "28px 24px 80px" }}>
      {tab === "learn" && (<div>
        <Eyebrow>Module 1 — Six concepts, properly</Eyebrow><h2 style={h2}>Learn the maths behind your results</h2>
        <p style={lead}>Click any concept to expand. Each has the formula, the intuition, and a proof or property.</p>
        {LESSONS.map((L) => (<div key={L.id} style={{ marginBottom: 12 }}>
          <button onClick={() => setOpen(open === L.id ? "" : L.id)} style={{ width: "100%", textAlign: "left", background: open === L.id ? C.ink : C.card, color: open === L.id ? C.paper : C.ink, border: `1px solid ${C.line}`, borderRadius: open === L.id ? "4px 4px 0 0" : 4, padding: "16px 20px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
            <div><div style={{ fontFamily: serif, fontSize: 20, fontWeight: 600 }}>{L.title}</div><div style={{ fontFamily: sans, fontSize: 13, opacity: 0.75, marginTop: 2 }}>{L.blurb}</div></div>
            <span style={{ fontFamily: mono, fontSize: 20, color: open === L.id ? C.amber : C.sage }}>{open === L.id ? "−" : "+"}</span>
          </button>
          {open === L.id && <div style={{ border: `1px solid ${C.line}`, borderTop: "none", borderRadius: "0 0 4px 4px", background: C.card, padding: "20px 22px" }}>{L.render()}</div>}
        </div>))}
        <Worked /><SummaryTable />
      </div>)}
      {tab === "question" && <QuestionBuilder />}
      {tab === "analyze" && <Analyzer lang={lang} setLang={setLang} />}
      {tab === "power" && <PowerLab />}
      {tab === "quiz" && <Quiz />}
      {tab === "reference" && (<div><Eyebrow>Statistical tables</Eyebrow><h2 style={h2}>Reference</h2><p style={lead}>Critical values for intervals and tests.</p><RefTables /><SummaryTable /></div>)}
    </main>

    <footer style={{ borderTop: `1px solid ${C.line}`, padding: "20px 24px", textAlign: "center" }}>
      <p style={{ fontFamily: mono, fontSize: 11, color: C.sage, margin: 0, letterSpacing: 1 }}>A teaching tool — verify analyses in validated software before publication.</p>
    </footer>
  </div>);
}

/* shared styles */
const h2 = { fontFamily: serif, fontSize: 30, fontWeight: 600, margin: "4px 0 10px", lineHeight: 1.1 };
const h4 = { fontFamily: sans, fontSize: 15, fontWeight: 600, color: C.sage, margin: "18px 0 4px" };
const pS = { fontFamily: sans, fontSize: 15, lineHeight: 1.65, color: "#33403c", margin: "0 0 10px" };
const lead = { ...pS, marginBottom: 18 };
const tbl = { width: "100%", borderCollapse: "collapse", fontFamily: sans, fontSize: 13.5 };
const th = { textAlign: "left", padding: "9px 11px", borderBottom: `2px solid ${C.sage}`, fontWeight: 600, fontSize: 12.5 };
const td = { padding: "8px 11px", borderBottom: `1px solid ${C.faint}`, color: "#3a4742", verticalAlign: "top" };
const lbl = { display: "block", fontFamily: mono, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: C.sage, marginBottom: 5 };
const inp = { width: "100%", padding: "10px 12px", border: `1px solid ${C.line}`, borderRadius: 3, fontFamily: sans, fontSize: 15, color: C.ink, background: C.paper, marginBottom: 12 };
const ta = { width: "100%", padding: "10px 12px", border: `1px solid ${C.line}`, borderRadius: 3, fontFamily: mono, fontSize: 14, color: C.ink, background: C.paper, marginBottom: 12, resize: "vertical" };
const grid = { display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 8 };
