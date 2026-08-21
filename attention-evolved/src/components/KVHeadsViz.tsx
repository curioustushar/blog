"use client";

import { useState } from "react";

type HeadConfig = "mha" | "mqa" | "gqa";

interface KVHeadsVizProps {
  className?: string;
}

export function KVHeadsViz({ className = "" }: KVHeadsVizProps) {
  const [config, setConfig] = useState<HeadConfig>("mha");
  const numQueryHeads = 8;

  const getKVHeads = () => {
    switch (config) {
      case "mha":
        return numQueryHeads;
      case "mqa":
        return 1;
      case "gqa":
        return 2;
    }
  };

  const kvHeads = getKVHeads();
  const kvCacheReduction = numQueryHeads / kvHeads;

  const descriptions: Record<HeadConfig, string> = {
    mha: `Multi-Head Attention: Each query head has its own K and V head. Maximum expressiveness, maximum KV cache.`,
    mqa: `Multi-Query Attention: All query heads share ONE K and V head. ${kvCacheReduction}× smaller KV cache.`,
    gqa: `Grouped Query Attention: Query heads share K/V in groups. ${kvCacheReduction}× smaller KV cache. Balances quality and efficiency.`,
  };

  const headColors = [
    "#3b82f6", "#8b5cf6", "#ec4899", "#ef4444",
    "#f59e0b", "#10b981", "#06b6d4", "#6366f1"
  ];

  const getKVColor = (queryIdx: number) => {
    switch (config) {
      case "mha":
        return headColors[queryIdx];
      case "mqa":
        return headColors[0];
      case "gqa":
        return headColors[Math.floor(queryIdx / (numQueryHeads / kvHeads))];
    }
  };

  return (
    <div className={`bg-white dark:bg-zinc-900 rounded-xl border border-border p-6 ${className}`}>
      <h3 className="text-lg font-semibold mb-4">KV Head Configurations</h3>
      
      <div className="flex flex-wrap gap-2 mb-6">
        {(["mha", "gqa", "mqa"] as HeadConfig[]).map((c) => (
          <button
            key={c}
            onClick={() => setConfig(c)}
            className={`px-3 py-1.5 text-sm rounded-lg uppercase transition-colors ${
              config === c
                ? "bg-accent text-white"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="mb-6">
        <div className="flex items-start gap-8 justify-center">
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase mb-2 text-center">
              Query Heads ({numQueryHeads})
            </p>
            <div className="flex gap-1">
              {Array.from({ length: numQueryHeads }).map((_, i) => (
                <div
                  key={i}
                  className="w-8 h-16 rounded flex items-center justify-center text-white text-xs font-bold"
                  style={{ backgroundColor: headColors[i] }}
                >
                  Q{i}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-center my-4">
          <svg width="280" height="40">
            {Array.from({ length: numQueryHeads }).map((_, i) => {
              const startX = 16 + i * 36;
              let endX: number;
              switch (config) {
                case "mha":
                  endX = 16 + i * 36;
                  break;
                case "mqa":
                  endX = 140;
                  break;
                case "gqa":
                  const groupIdx = Math.floor(i / (numQueryHeads / kvHeads));
                  endX = 52 + groupIdx * 92;
                  break;
              }
              return (
                <path
                  key={i}
                  d={`M${startX},0 Q${startX},20 ${endX},40`}
                  stroke={getKVColor(i)}
                  strokeWidth="2"
                  fill="none"
                  className="transition-all duration-300"
                />
              );
            })}
          </svg>
        </div>

        <div className="flex justify-center">
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase mb-2 text-center">
              K/V Heads ({kvHeads})
            </p>
            <div className="flex gap-1 justify-center">
              {Array.from({ length: kvHeads }).map((_, i) => (
                <div
                  key={i}
                  className="rounded flex flex-col gap-1"
                >
                  <div
                    className="w-8 h-8 rounded flex items-center justify-center text-white text-xs font-bold"
                    style={{ backgroundColor: headColors[config === "mha" ? i : config === "mqa" ? 0 : i] }}
                  >
                    K{i}
                  </div>
                  <div
                    className="w-8 h-8 rounded flex items-center justify-center text-white text-xs font-bold"
                    style={{ backgroundColor: headColors[config === "mha" ? i : config === "mqa" ? 0 : i] }}
                  >
                    V{i}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-muted p-3 rounded-lg">
          <p className="text-xs font-semibold text-muted-foreground uppercase mb-1">KV Cache Size</p>
          <p className="text-lg font-bold">
            {kvCacheReduction > 1 ? `1/${kvCacheReduction}` : "Full"} of MHA
          </p>
        </div>
        <div className="bg-muted p-3 rounded-lg">
          <p className="text-xs font-semibold text-muted-foreground uppercase mb-1">Query Expressiveness</p>
          <p className="text-lg font-bold">
            {config === "mha" ? "Maximum" : config === "gqa" ? "High" : "Reduced"}
          </p>
        </div>
      </div>

      <p className="text-sm text-muted-foreground bg-muted p-3 rounded-lg">
        {descriptions[config]}
      </p>
    </div>
  );
}
