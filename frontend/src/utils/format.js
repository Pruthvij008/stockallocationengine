// Shorthand multipliers the amount field accepts.
//  k = thousand, l = lakh, m = million, c = crore, b = billion
const SUFFIX = { k: 1e3, l: 1e5, m: 1e6, c: 1e7, b: 1e9 };

// Keep only digits, a single decimal point, and one trailing shorthand letter.
// Anything else the user types is dropped, so the field can never hold junk.
export function sanitizeAmountInput(raw) {
  let s = String(raw).toLowerCase().replace(/[^0-9.klmcb]/g, "");

  // collapse multiple dots to the first one
  const firstDot = s.indexOf(".");
  if (firstDot !== -1) {
    s = s.slice(0, firstDot + 1) + s.slice(firstDot + 1).replace(/\./g, "");
  }

  // keep digits/dot in order, then a single (last) shorthand letter at the end
  const digits = s.replace(/[klmcb]/g, "");
  const letters = s.replace(/[^klmcb]/g, "");
  const suffix = letters ? letters[letters.length - 1] : "";
  return digits + suffix;
}

// Turn "2.5cr" / "500k" / "1m" into an actual number.
export function parseAmount(raw) {
  const s = sanitizeAmountInput(raw);
  if (!s) return 0;
  const suffix = /[klmcb]$/.test(s) ? s[s.length - 1] : "";
  const num = parseFloat(suffix ? s.slice(0, -1) : s);
  if (Number.isNaN(num)) return NaN;
  return num * (suffix ? SUFFIX[suffix] : 1);
}

// ₹ with Indian digit grouping (e.g. ₹1,50,00,000).
export function formatINR(value, decimals = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return (
    "₹" +
    Number(value).toLocaleString("en-IN", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
  );
}

// Compact ₹ (e.g. ₹1.67 Cr, ₹2.50 L, ₹50.00 K).
export function formatCompactINR(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const v = Number(value);
  const abs = Math.abs(v);
  if (abs >= 1e7) return "₹" + (v / 1e7).toFixed(2) + " Cr";
  if (abs >= 1e5) return "₹" + (v / 1e5).toFixed(2) + " L";
  if (abs >= 1e3) return "₹" + (v / 1e3).toFixed(2) + " K";
  return "₹" + v.toFixed(2);
}

// Plain number formatting with fixed decimals; returns "—" for null/NaN.
export function formatNum(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(decimals);
}
