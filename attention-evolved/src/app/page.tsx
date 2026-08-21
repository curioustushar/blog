"use client";

import { useState } from "react";
import { Timeline } from "@/components/Timeline";
import { CompareMode } from "@/components/CompareMode";
import { AttentionMatrixViz } from "@/components/AttentionMatrixViz";
import { KVHeadsViz } from "@/components/KVHeadsViz";
import { ThemeToggle } from "@/components/ThemeToggle";
import { SynthesisView } from "@/components/SynthesisView";
import { attentionMechanisms, pre2017Mechanisms } from "@/data/attentionMechanisms";

export default function Home() {
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [showCompare, setShowCompare] = useState(false);
  const [activeSection, setActiveSection] = useState<"timeline" | "lab" | "synthesis">("timeline");

  const handleCompareSelect = (id: string) => {
    setCompareIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((i) => i !== id);
      }
      if (prev.length >= 2) {
        return [prev[1], id];
      }
      return [...prev, id];
    });
  };

  const handleRemoveFromCompare = (id: string) => {
    setCompareIds((prev) => prev.filter((i) => i !== id));
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 bg-background/80 backdrop-blur-sm border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-xl font-bold">
              Attention, <span className="text-accent">Evolved</span>
            </h1>
            <div className="flex items-center gap-2">
              <nav className="flex items-center gap-1">
                <button
                  onClick={() => setActiveSection("timeline")}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  activeSection === "timeline"
                    ? "bg-accent text-white"
                    : "text-muted-foreground hover:bg-muted"
                }`}
              >
                Timeline
              </button>
              <button
                onClick={() => setActiveSection("lab")}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  activeSection === "lab"
                    ? "bg-accent text-white"
                    : "text-muted-foreground hover:bg-muted"
                }`}
              >
                Visual Lab
              </button>
              <button
                onClick={() => setActiveSection("synthesis")}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  activeSection === "synthesis"
                    ? "bg-accent text-white"
                    : "text-muted-foreground hover:bg-muted"
                }`}
              >
                What Changed?
              </button>
              </nav>
              {compareIds.length > 0 && (
                <button
                  onClick={() => setShowCompare(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent/90 transition-colors"
                >
                  Compare ({compareIds.length}/2)
                </button>
              )}
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      <main>
        <section className="py-16 md:py-24 border-b border-border">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              The Evolution of Attention
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
              From <strong>{pre2017Mechanisms.length} pre-Transformer techniques</strong> through{" "}
              <strong>{attentionMechanisms.length} post-2017 mechanisms</strong> — each a response
              to a different bottleneck.
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-sm">
              <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <span className="font-semibold">{pre2017Mechanisms.length}</span>
                <span className="text-muted-foreground">Before 2017</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg">
                <span className="font-semibold">{attentionMechanisms.filter(m => m.category === "attention-mechanism").length}</span>
                <span className="text-muted-foreground">Core Mechanisms</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg">
                <span className="font-semibold">{attentionMechanisms.filter(m => m.category === "positional-encoding" || m.category === "positional-scaling").length}</span>
                <span className="text-muted-foreground">Position Techniques</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg">
                <span className="font-semibold">{attentionMechanisms.filter(m => m.category === "kv-cache-optimization").length}</span>
                <span className="text-muted-foreground">KV Optimizations</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg">
                <span className="font-semibold">{attentionMechanisms.filter(m => m.category === "sparse-attention" || m.category === "linear-attention" || m.category === "recurrent-attention").length}</span>
                <span className="text-muted-foreground">Efficiency Methods</span>
              </div>
            </div>
            <p className="mt-8 text-muted-foreground">
              <strong>The pattern:</strong> compute → memory → context length → positional extrapolation → memory bandwidth → compressed representations
            </p>
          </div>
        </section>

        {activeSection === "timeline" && (
          <section className="py-12 md:py-16">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="mb-10">
                <h3 className="text-2xl font-bold mb-2">Before 2017</h3>
                <p className="text-muted-foreground max-w-3xl">
                  Attention did not start with the Transformer. Seq2seq RNNs had an information
                  bottleneck; these two papers introduced learned alignment and dot-product scoring —
                  the direct precursors to Q, K, V.
                </p>
              </div>
              <Timeline
                mechanisms={pre2017Mechanisms}
                onCompareSelect={handleCompareSelect}
                compareSelected={compareIds}
                showFilters={false}
                variant="pre2017"
              />

              <div className="my-16 flex flex-col items-center text-center px-4">
                <div className="w-0.5 h-12 bg-gradient-to-b from-amber-600 to-accent mb-4" />
                <p className="text-sm font-semibold text-accent uppercase tracking-wide mb-2">
                  June 2017 — the pivot
                </p>
                <p className="text-muted-foreground max-w-2xl">
                  Vaswani et al. removed the RNN entirely: self-attention with scaled dot-product
                  and multi-head attention. The timeline below starts here — Session 2&apos;s baseline —
                  and tracks every cost-cutting response that followed.
                </p>
              </div>

              <div className="mb-10">
                <h3 className="text-2xl font-bold mb-2">Chronological Timeline (2017–2026)</h3>
                <p className="text-muted-foreground">
                  Each mechanism in verified publication order. Click to expand. Select two to compare
                  (including entries from Before 2017).
                </p>
              </div>
              <Timeline
                mechanisms={attentionMechanisms}
                onCompareSelect={handleCompareSelect}
                compareSelected={compareIds}
              />
            </div>
          </section>
        )}

        {activeSection === "lab" && (
          <section className="py-12 md:py-16">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="mb-8">
                <h3 className="text-2xl font-bold mb-2">Interactive Visual Lab</h3>
                <p className="text-muted-foreground">
                  Explore how different attention patterns and head configurations work.
                </p>
              </div>
              <div className="grid md:grid-cols-2 gap-6">
                <AttentionMatrixViz />
                <KVHeadsViz />
              </div>
            </div>
          </section>
        )}

        {activeSection === "synthesis" && (
          <section className="py-12 md:py-16">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
              <h3 className="text-2xl font-bold mb-2">What Changed?</h3>
              <p className="text-muted-foreground mb-8">
                The timeline in date order — plus context from before 2017 and systems milestones not on the timeline.
              </p>
              <SynthesisView />
            </div>
          </section>
        )}
      </main>

      <footer className="border-t border-border py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-sm text-muted-foreground">
              <p>
                All dates verified against primary sources (arXiv, conference proceedings, technical reports).
              </p>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <a
                href="https://github.com/curioustushar/curioustushar.github.io/tree/main/attention-evolved"
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent hover:underline"
              >
                GitHub Repo
              </a>
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
                className="text-muted-foreground hover:text-foreground"
              >
                Back to top
              </a>
            </div>
          </div>
        </div>
      </footer>

      {showCompare && (
        <CompareMode
          selectedIds={compareIds}
          onClose={() => setShowCompare(false)}
          onRemove={handleRemoveFromCompare}
        />
      )}
    </div>
  );
}
