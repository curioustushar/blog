"use client";

import { useState } from "react";
import { AttentionMechanism } from "@/data/attentionMechanisms";
import { CategoryBadge } from "./CategoryBadge";
import { MathDisplay } from "./MathDisplay";

interface MechanismCardProps {
  mechanism: AttentionMechanism;
  isExpanded?: boolean;
  onToggle?: () => void;
  onCompare?: () => void;
  isCompareSelected?: boolean;
}

export function MechanismCard({
  mechanism,
  isExpanded = false,
  onToggle,
  onCompare,
  isCompareSelected = false,
}: MechanismCardProps) {
  const [showMath, setShowMath] = useState(false);

  const dateTypeLabels: Record<string, string> = {
    arxiv: "arXiv",
    conference: "Conference",
    "technical-report": "Technical Report",
    release: "Release",
    forum: "Forum",
  };

  const formatDate = (dateStr: string) => {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const [year, month, day] = dateStr.split("-").map(Number);
    return `${months[month - 1]} ${day}, ${year}`;
  };

  return (
    <article
      className={`bg-white dark:bg-zinc-900 rounded-xl border transition-all duration-200 ${
        isExpanded
          ? "border-accent shadow-lg"
          : "border-border hover:border-accent/50 hover:shadow-md"
      }`}
    >
      <header
        className="p-5 cursor-pointer select-none"
        onClick={onToggle}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 flex-wrap mb-2">
              <h3 className="text-lg font-semibold text-foreground">
                {mechanism.name}
              </h3>
              <CategoryBadge category={mechanism.category} size="sm" />
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <time dateTime={mechanism.date} className="font-medium">
                {formatDate(mechanism.date)}
              </time>
              <span className="text-xs px-1.5 py-0.5 bg-muted rounded">
                {dateTypeLabels[mechanism.dateType]}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {onCompare && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onCompare();
                }}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  isCompareSelected
                    ? "bg-accent text-white"
                    : "bg-muted text-muted-foreground hover:bg-accent/10 hover:text-accent"
                }`}
              >
                {isCompareSelected ? "Selected" : "Compare"}
              </button>
            )}
            <button
              className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
              aria-label={isExpanded ? "Collapse" : "Expand"}
            >
              <svg
                className={`w-5 h-5 transition-transform ${
                  isExpanded ? "rotate-180" : ""
                }`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>
          </div>
        </div>

        <p className="mt-3 text-sm text-muted-foreground line-clamp-2">
          {mechanism.intuition}
        </p>
      </header>

      {isExpanded && (
        <div className="px-5 pb-5 border-t border-border animate-fadeIn">
          <div className="pt-5 space-y-6">
            <section>
              <h4 className="text-sm font-semibold text-accent uppercase tracking-wide mb-2">
                The Problem
              </h4>
              <p className="text-sm text-foreground prose-technical">
                {mechanism.problem}
              </p>
            </section>

            <section>
              <h4 className="text-sm font-semibold text-accent uppercase tracking-wide mb-2">
                The Idea
              </h4>
              <p className="text-sm text-foreground prose-technical">
                {mechanism.technicalExplanation}
              </p>
            </section>

            {mechanism.equation && (
              <section>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-accent uppercase tracking-wide">
                    The Math
                  </h4>
                  <button
                    onClick={() => setShowMath(!showMath)}
                    className="text-xs text-muted-foreground hover:text-accent transition-colors"
                  >
                    {showMath ? "Hide equation" : "Show equation"}
                  </button>
                </div>
                {showMath && (
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <MathDisplay math={mechanism.equation} display />
                    {mechanism.equationExplanation && (
                      <p className="mt-3 text-xs text-muted-foreground">
                        {mechanism.equationExplanation}
                      </p>
                    )}
                  </div>
                )}
              </section>
            )}

            <div className="grid md:grid-cols-2 gap-6">
              <section>
                <h4 className="text-sm font-semibold text-green-600 dark:text-green-400 uppercase tracking-wide mb-2 flex items-center gap-1">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  What It Buys
                </h4>
                <ul className="space-y-1.5">
                  {mechanism.advantages.map((adv, i) => (
                    <li key={i} className="text-sm text-foreground flex items-start gap-2">
                      <span className="text-green-500 mt-1">+</span>
                      {adv}
                    </li>
                  ))}
                </ul>
              </section>

              <section>
                <h4 className="text-sm font-semibold text-red-600 dark:text-red-400 uppercase tracking-wide mb-2 flex items-center gap-1">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  What It Costs
                </h4>
                <ul className="space-y-1.5">
                  {mechanism.disadvantages.map((dis, i) => (
                    <li key={i} className="text-sm text-foreground flex items-start gap-2">
                      <span className="text-red-500 mt-1">-</span>
                      {dis}
                    </li>
                  ))}
                </ul>
              </section>
            </div>

            <section className="grid md:grid-cols-3 gap-4">
              <div className="bg-muted p-3 rounded-lg">
                <h5 className="text-xs font-semibold text-muted-foreground uppercase mb-1">
                  Time Complexity
                </h5>
                <p className="text-sm font-mono font-medium">
                  {mechanism.complexity.time}
                </p>
              </div>
              <div className="bg-muted p-3 rounded-lg">
                <h5 className="text-xs font-semibold text-muted-foreground uppercase mb-1">
                  Memory Complexity
                </h5>
                <p className="text-sm font-mono font-medium">
                  {mechanism.complexity.memory}
                </p>
              </div>
              <div className="bg-muted p-3 rounded-lg">
                <h5 className="text-xs font-semibold text-muted-foreground uppercase mb-1">
                  KV Cache
                </h5>
                <p className="text-sm font-medium capitalize">
                  {mechanism.kvCache.size}
                </p>
              </div>
            </section>

            <section>
              <h4 className="text-sm font-semibold text-accent uppercase tracking-wide mb-2">
                When Would I Use This?
              </h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs font-medium text-green-600 dark:text-green-400 mb-1">
                    Good choice for:
                  </p>
                  <ul className="space-y-1">
                    {mechanism.useCases.good.map((use, i) => (
                      <li key={i} className="text-sm text-foreground">
                        • {use}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-xs font-medium text-red-600 dark:text-red-400 mb-1">
                    Poor choice for:
                  </p>
                  <ul className="space-y-1">
                    {mechanism.useCases.poor.map((use, i) => (
                      <li key={i} className="text-sm text-foreground">
                        • {use}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </section>

            <section className="pt-4 border-t border-border">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                Primary Source
              </h4>
              <a
                href={mechanism.paper.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-accent hover:underline inline-flex items-center gap-1"
              >
                {mechanism.paper.title}
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
              <p className="text-xs text-muted-foreground mt-1">
                {mechanism.paper.authors.join(", ")}
              </p>
            </section>

            {mechanism.transitionNote && (
              <section className="bg-accent/5 p-4 rounded-lg border-l-4 border-accent">
                <p className="text-sm text-foreground italic">
                  {mechanism.transitionNote}
                </p>
              </section>
            )}
          </div>
        </div>
      )}
    </article>
  );
}
