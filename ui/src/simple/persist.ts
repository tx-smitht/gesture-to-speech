// A useState that this browser remembers across reloads (where you are in the intro, simple vs advanced view).
import { useCallback, useState } from "react";

export function usePersistent<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const saved = localStorage.getItem(key);
      return saved === null ? initial : (JSON.parse(saved) as T);
    } catch {
      return initial; // private window or blocked storage: just don't remember
    }
  });
  const set = useCallback((v: T) => {
    setValue(v);
    try {
      localStorage.setItem(key, JSON.stringify(v));
    } catch {
      // as above
    }
  }, [key]);
  return [value, set] as const;
}
