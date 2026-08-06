// Central place for the backend base URL.
// Override in development/production via a .env file: VITE_API_URL=...
//
// Must be a full host (e.g. https://investify-api-xxxx.onrender.com), not a
// bare service name. A scheme is added when missing, but a name without a
// domain cannot resolve — so warn loudly rather than failing silently on
// every request.
let base = (import.meta.env.VITE_API_URL || "").trim().replace(/\/+$/, "");

if (base && !/^https?:\/\//i.test(base)) {
  base = `https://${base}`;
}

const host = base.replace(/^https?:\/\//i, "");
if (base && !host.includes(".") && host !== "localhost" && !host.startsWith("localhost:")) {
  console.error(
    `VITE_API_URL looks like a service name, not a host: "${import.meta.env.VITE_API_URL}". ` +
      `Every API call will fail. Set it to the full URL, e.g. https://${host}.onrender.com`
  );
}

export const API_URL = base || "http://localhost:8080";
