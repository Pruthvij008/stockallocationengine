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

// ---------------------------------------------------------------------------
// Dates. The UI shows dd-MMM-yyyy (e.g. 06-Aug-2026) everywhere — unambiguous
// no matter which day/month ordering the reader expects.
// ---------------------------------------------------------------------------

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

// Accepts "YYYY-MM-DD" (what the API returns), an ISO timestamp, or a Date.
// Plain YYYY-MM-DD is parsed by hand so a UTC/local shift can't move the day.
export function formatDate(value) {
  if (value === null || value === undefined || value === "") return "—";

  if (typeof value === "string") {
    const m = value.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) {
      const month = MONTHS[Number(m[2]) - 1];
      if (month) return `${m[3]}-${month}-${m[1]}`;
    }
  }

  const dt = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(dt.getTime())) return String(value);
  const dd = String(dt.getDate()).padStart(2, "0");
  return `${dd}-${MONTHS[dt.getMonth()]}-${dt.getFullYear()}`;
}

// Compact variant for dense chart axes: MMM-yy (e.g. Aug-26).
export function formatDateShort(value) {
  const full = formatDate(value);
  const parts = full.split("-");
  if (parts.length !== 3) return full;
  return `${parts[1]}-${parts[2].slice(2)}`;
}

// "01-Jan-2020 → 06-Aug-2026" — pair with the .iv-nowrap class to keep it on
// a single line.
export function formatDateRange(start, end) {
  return `${formatDate(start)} → ${formatDate(end)}`;
}
