"use client";

import { useState, useMemo } from "react";

type AttentionPattern = "full" | "causal" | "sliding" | "sparse" | "sink";

interface AttentionMatrixVizProps {
  className?: string;
}

export function AttentionMatrixViz({ className = "" }: AttentionMatrixVizProps) {
  const [pattern, setPattern] = useState<AttentionPattern>("full");
  const [seqLength, setSeqLength] = useState(8);
  const [windowSize, setWindowSize] = useState(3);

  const cellSize = Math.min(32, Math.floor(280 / seqLength));

  const matrix = useMemo(() => {
    const m: number[][] = [];
    for (let i = 0; i < seqLength; i++) {
      const row: number[] = [];
      for (let j = 0; j < seqLength; j++) {
        let value = 0;
        switch (pattern) {
          case "full":
            value = 1;
            break;
          case "causal":
            value = j <= i ? 1 : 0;
            break;
          case "sliding":
            value = Math.abs(i - j) <= windowSize ? 1 : 0;
            break;
          case "sparse":
            value = j <= i && (j === 0 || j === i || j % 2 === 0) ? 1 : 0;
            break;
          case "sink":
            value = (j <= i && j <= 1) || (j <= i && i - j <= windowSize) ? 1 : 0;
            break;
        }
        row.push(value);
      }
      m.push(row);
    }
    return m;
  }, [pattern, seqLength, windowSize]);

  const patternDescriptions: Record<AttentionPattern, string> = {
    full: "Every token attends to every other token. O(n²) compute and memory.",
    causal: "Each token only attends to itself and previous tokens. Used in decoder-only models.",
    sliding: `Each token attends to tokens within a window of ${windowSize}. O(n·w) complexity.`,
    sparse: "Structured sparsity: first token, self, and strided positions. Reduces compute.",
    sink: `First 2 tokens are 'sinks' (always attended) + sliding window of ${windowSize}. Used in StreamingLLM.`,
  };

  return (
    <div className={`bg-white dark:bg-zinc-900 rounded-xl border border-border p-6 ${className}`}>
      <h3 className="text-lg font-semibold mb-4">Attention Pattern Explorer</h3>
      
      <div className="flex flex-wrap gap-2 mb-4">
        {(["full", "causal", "sliding", "sparse", "sink"] as AttentionPattern[]).map((p) => (
          <button
            key={p}
            onClick={() => setPattern(p)}
            className={`px-3 py-1.5 text-sm rounded-lg capitalize transition-colors ${
              pattern === p
                ? "bg-accent text-white"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {p === "sink" ? "Attention Sink" : p}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-6 mb-4">
        <div>
          <label className="block text-sm font-medium text-muted-foreground mb-1">
            Sequence Length: {seqLength}
          </label>
          <input
            type="range"
            min={4}
            max={16}
            value={seqLength}
            onChange={(e) => setSeqLength(parseInt(e.target.value))}
            className="w-32"
          />
        </div>
        {(pattern === "sliding" || pattern === "sink") && (
          <div>
            <label className="block text-sm font-medium text-muted-foreground mb-1">
              Window Size: {windowSize}
            </label>
            <input
              type="range"
              min={1}
              max={Math.floor(seqLength / 2)}
              value={windowSize}
              onChange={(e) => setWindowSize(parseInt(e.target.value))}
              className="w-32"
            />
          </div>
        )}
      </div>

      <div className="flex justify-center mb-4">
        <div className="inline-block">
          <div className="flex items-end mb-1">
            <div style={{ width: cellSize + 4 }} />
            <div className="flex">
              {Array.from({ length: seqLength }).map((_, j) => (
                <div
                  key={j}
                  className="text-xs text-muted-foreground text-center"
                  style={{ width: cellSize }}
                >
                  {j}
                </div>
              ))}
            </div>
          </div>
          <div className="flex">
            <div className="flex flex-col justify-start" style={{ width: cellSize + 4 }}>
              {Array.from({ length: seqLength }).map((_, i) => (
                <div
                  key={i}
                  className="text-xs text-muted-foreground flex items-center justify-end pr-1"
                  style={{ height: cellSize }}
                >
                  {i}
                </div>
              ))}
            </div>
            <svg
              width={seqLength * cellSize}
              height={seqLength * cellSize}
              className="border border-border rounded"
            >
              {matrix.map((row, i) =>
                row.map((val, j) => (
                  <rect
                    key={`${i}-${j}`}
                    x={j * cellSize}
                    y={i * cellSize}
                    width={cellSize - 1}
                    height={cellSize - 1}
                    fill={val ? "#3b82f6" : "#f5f5f5"}
                    className="transition-colors"
                  />
                ))
              )}
              {matrix.map((row, i) =>
                row.map((val, j) =>
                  val ? (
                    <text
                      key={`t-${i}-${j}`}
                      x={j * cellSize + cellSize / 2}
                      y={i * cellSize + cellSize / 2}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      className="text-[10px] fill-white font-medium"
                    >
                      {i === j ? "Q" : "•"}
                    </text>
                  ) : null
                )
              )}
            </svg>
          </div>
          <div className="text-xs text-muted-foreground text-center mt-2">
            Keys (columns) → attended by Query (rows)
          </div>
        </div>
      </div>

      <p className="text-sm text-muted-foreground bg-muted p-3 rounded-lg">
        {patternDescriptions[pattern]}
      </p>

      <div className="mt-4 flex items-center gap-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-accent rounded" />
          <span>Can attend</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-muted rounded border border-border" />
          <span>Masked</span>
        </div>
      </div>
    </div>
  );
}
