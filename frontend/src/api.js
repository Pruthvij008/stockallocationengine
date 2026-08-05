// Central place for the backend base URL.
// Override in development/production via a .env file: VITE_API_URL=...
//
// Hosts like Render expose a service's address as a bare hostname
// (e.g. "investify-api.onrender.com"), so prepend https:// when no scheme is
// present. Falls back to the local Express dev server.
let base = (import.meta.env.VITE_API_URL || "").trim();
if (base && !/^https?:\/\//i.test(base)) {
  base = `https://${base}`;
}

export const API_URL = base || "http://localhost:8080";
