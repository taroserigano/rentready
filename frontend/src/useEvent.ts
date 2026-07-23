import { useCallback, useLayoutEffect, useRef } from "react";

/**
 * A permanently-stable function identity that always invokes the latest
 * render's closure. Lets a callback be passed to a React.memo'd child (e.g. a
 * chat transcript row) without the child re-rendering just because the
 * parent re-rendered and recreated the callback — while never going stale,
 * unlike a bare useCallback with a hand-picked dependency array.
 */
export function useEvent<Args extends unknown[], R>(
  fn: (...args: Args) => R,
): (...args: Args) => R {
  const ref = useRef(fn);
  useLayoutEffect(() => {
    ref.current = fn;
  });
  return useCallback((...args: Args) => ref.current(...args), []);
}
