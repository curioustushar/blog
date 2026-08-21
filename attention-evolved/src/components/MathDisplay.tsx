"use client";

import { useEffect, useRef } from "react";
import katex from "katex";

interface MathDisplayProps {
  math: string;
  display?: boolean;
  className?: string;
}

export function MathDisplay({ math, display = false, className = "" }: MathDisplayProps) {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        katex.render(math, containerRef.current, {
          displayMode: display,
          throwOnError: false,
          strict: false,
        });
      } catch (e) {
        console.error("KaTeX error:", e);
        if (containerRef.current) {
          containerRef.current.textContent = math;
        }
      }
    }
  }, [math, display]);

  return <span ref={containerRef} className={className} />;
}
