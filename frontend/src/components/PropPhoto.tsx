import { useState } from "react";
import { Building2, ChevronLeft, ChevronRight } from "lucide-react";

/**
 * A property image that degrades gracefully: if the URL is missing OR the
 * remote image fails to load (offline / blocked CDN), it swaps to the same
 * aurora-gradient + Building2 placeholder used across the app.
 */
export function PropThumb({
  src,
  alt = "",
  className = "prop-img",
  phClassName = "prop-img-ph",
}: {
  src?: string | null;
  alt?: string;
  className?: string;
  phClassName?: string;
}) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div className={phClassName} aria-hidden>
        <Building2 />
      </div>
    );
  }
  return (
    <img
      className={className}
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

/**
 * A multi-photo gallery: one large hero with a filmstrip of thumbnails.
 * Falls back to a single placeholder when there are no usable photos.
 * De-dupes the incoming list and hides any image that fails to load.
 */
export function PropGallery({
  photos,
  alt = "",
  heroClassName = "gallery-hero",
}: {
  photos?: (string | undefined)[];
  alt?: string;
  heroClassName?: string;
}) {
  const list = Array.from(new Set((photos ?? []).filter(Boolean) as string[]));
  const [idx, setIdx] = useState(0);
  const [broken, setBroken] = useState<Record<number, boolean>>({});

  const usable = list.filter((_, i) => !broken[i]);
  if (usable.length === 0) {
    return (
      <div className={`${heroClassName} prop-img-ph`} aria-hidden>
        <Building2 />
      </div>
    );
  }

  const safeIdx = Math.min(idx, list.length - 1);
  const step = (d: number) => {
    // advance to the next non-broken photo in direction d
    const n = list.length;
    for (let k = 1; k <= n; k++) {
      const j = (safeIdx + d * k + n * k) % n;
      if (!broken[j]) {
        setIdx(j);
        return;
      }
    }
  };

  return (
    <div className="gallery">
      <div className="gallery-stage">
        <img
          className={heroClassName}
          src={list[safeIdx]}
          alt={alt}
          onError={() => setBroken((b) => ({ ...b, [safeIdx]: true }))}
        />
        {usable.length > 1 && (
          <>
            <button
              className="gallery-nav prev"
              onClick={() => step(-1)}
              aria-label="Previous photo"
            >
              <ChevronLeft size={18} />
            </button>
            <button
              className="gallery-nav next"
              onClick={() => step(1)}
              aria-label="Next photo"
            >
              <ChevronRight size={18} />
            </button>
            <span className="gallery-count">
              {usable.findIndex((u) => u === list[safeIdx]) + 1} / {usable.length}
            </span>
          </>
        )}
      </div>
      {usable.length > 1 && (
        <div className="gallery-strip">
          {list.map((u, i) =>
            broken[i] ? null : (
              <img
                key={u}
                src={u}
                alt=""
                loading="lazy"
                className={`gallery-thumb${i === safeIdx ? " active" : ""}`}
                onClick={() => setIdx(i)}
                onError={() => setBroken((b) => ({ ...b, [i]: true }))}
              />
            )
          )}
        </div>
      )}
    </div>
  );
}
