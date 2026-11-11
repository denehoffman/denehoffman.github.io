// Monochrome event layers: preserve sampled positions, rasterize each time slice
// once, and tint the final chamber separately so theme changes need no rebuild.
(() => {
  "use strict";
  const SLICE = 64; // ms; fade approximation error <= 0.8% with a 4 s fade.
  const TILE = 192; // avoid caching empty space between unrelated tracks
  const CACHE_LIMIT = 96 * 1024 * 1024;
  let cacheBytes = 0;
  const bucket = (bubble) => Math.min(2, Math.floor((bubble.strength - 0.55) / 0.15));
  function pathsFor(bubbles, until = Infinity) {
    const paths = [new Path2D(), new Path2D(), new Path2D()];
    for (const bubble of bubbles) {
      if (bubble.time > until) continue;
      const path = paths[bucket(bubble)];
      path.moveTo(bubble.x + bubble.radius, bubble.y);
      path.arc(bubble.x, bubble.y, bubble.radius, 0, 2 * Math.PI);
    }
    return paths;
  }
  function fillPaths(context, paths, opacity) {
    for (let index = 0; index < paths.length; index += 1) {
      context.globalAlpha = opacity * (0.625 + 0.15 * index);
      context.fill(paths[index]);
    }
  }
  window.BubbleRenderer = {
    createLayer(width, height, ratio, { photograph = false } = {}) {
      const slices = new Map();
      const images = [];
      let complete = null;
      let end = 0;
      function rasterize(bounds, paint) {
        const x = Math.max(0, Math.floor(bounds.minX * ratio));
        const y = Math.max(0, Math.floor(bounds.minY * ratio));
        const w = Math.max(0, Math.min(Math.ceil(bounds.maxX * ratio), Math.ceil(width * ratio)) - x);
        const h = Math.max(0, Math.min(Math.ceil(bounds.maxY * ratio), Math.ceil(height * ratio)) - y);
        const bytes = w * h * 4;
        if (!w || !h || cacheBytes + bytes > CACHE_LIMIT) return null;
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const context = canvas.getContext("2d");
        if (!context) return null;
        context.setTransform(ratio, 0, 0, ratio, -x, -y);
        context.fillStyle = "black";
        paint(context);
        const image = { canvas, x: x / ratio, y: y / ratio, w: w / ratio, h: h / ratio, bytes };
        images.push(image);
        cacheBytes += bytes;
        return image;
      }
      function releaseImage(image) {
        cacheBytes -= image.bytes;
        image.bytes = 0;
        image.canvas.width = image.canvas.height = 0;
      }
      function paintSlice(context, slice, opacity) {
        if (slice.image) {
          const image = slice.image;
          context.globalAlpha = opacity;
          context.drawImage(image.canvas, image.x, image.y, image.w, image.h);
        } else fillPaths(context, slice.paths, opacity);
      }
      return {
        addTrack(track) {
          end = Math.max(end, track.delay + track.times[track.times.length - 1]);
          for (const bubble of track.bubbles) {
            const time = photograph ? 0 : Math.floor(bubble.time / SLICE) * SLICE;
            const key = `${time}:${Math.floor(bubble.x / TILE)}:${Math.floor(bubble.y / TILE)}`;
            if (!slices.has(key)) slices.set(key, { time, bubbles: [],
              minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity });
            const slice = slices.get(key);
            slice.bubbles.push(bubble);
            slice.minX = Math.min(slice.minX, bubble.x - bubble.radius - 1);
            slice.minY = Math.min(slice.minY, bubble.y - bubble.radius - 1);
            slice.maxX = Math.max(slice.maxX, bubble.x + bubble.radius + 1);
            slice.maxY = Math.max(slice.maxY, bubble.y + bubble.radius + 1);
          }
        },
        *prepare() {
          const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
          for (const slice of slices.values()) {
            slice.paths = pathsFor(slice.bubbles);
            slice.image = rasterize(slice, (context) => fillPaths(context, slice.paths, 1));
            if (slice.image) slice.paths = null;
            bounds.minX = Math.min(bounds.minX, slice.minX);
            bounds.minY = Math.min(bounds.minY, slice.minY);
            bounds.maxX = Math.max(bounds.maxX, slice.maxX);
            bounds.maxY = Math.max(bounds.maxY, slice.maxY);
            yield;
          }
          if (slices.size) complete = rasterize(bounds, (context) => {
            for (const slice of slices.values()) paintSlice(context, slice, 1);
          });
          if (photograph && complete) {
            // A photograph never needs a moving head or per-bubble fading.
            // Keep only its combined image, releasing temporary tile storage
            // and sample arrays immediately instead of retaining both copies.
            for (const image of images) if (image !== complete) releaseImage(image);
            slices.clear();
          }
        },
        draw(context, { age, opacity, photo, growth, scale, hold, fade, keep }) {
          // Subpixel bubble growth/compression is represented by covered area.
          // Sample locations and the fully developed bubble sizes stay exact.
          if (photo) opacity *= growth * (0.2 + 0.8 * Math.sqrt(growth)) ** 2 * scale ** 2;
          if (opacity <= 0) return;
          if (complete && (photo || (age >= end && (keep || age <= hold)))) {
            context.globalAlpha = opacity;
            context.drawImage(complete.canvas, complete.x, complete.y, complete.w, complete.h);
            return;
          }
          for (const slice of slices.values()) {
            if (!photo && age < slice.time) continue;
            const visibility = photo || keep ? 1 : Math.max(0, Math.min(1, 1 - (age - slice.time - SLICE / 2 - hold) / fade));
            if (!visibility) continue;
            if (!photo && age < slice.time + SLICE) {
              fillPaths(context, pathsFor(slice.bubbles, age), opacity * visibility);
            } else paintSlice(context, slice, opacity * visibility);
          }
        },
        dispose() {
          for (const image of images) releaseImage(image);
          images.length = 0;
          slices.clear();
          complete = null;
        },
      };
    },
  };
})();
