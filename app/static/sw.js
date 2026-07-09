/* DETECTOR service worker — deliberately minimal.
   Network-first for everything; the cache exists only as an offline fallback
   for the app shell. /api is never cached (we have scar tissue about caches
   serving stale "processing" states). Bump VERSION on asset changes. */
const VERSION = "aidet-v4";
const SHELL = ["/", "/styles.css?v=5", "/app.js?v=5", "/manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin
      || url.pathname.startsWith("/api/")) {
    return; // straight to network, no caching, ever
  }
  e.respondWith(
    fetch(e.request)
      .then((resp) => {
        if (resp.ok) {
          const copy = resp.clone();
          caches.open(VERSION).then((c) => c.put(e.request, copy));
        }
        return resp;
      })
      .catch(() => caches.match(e.request, { ignoreSearch: url.pathname === "/" }))
  );
});
