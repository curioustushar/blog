"use client";

import { useState, useMemo } from "react";
import {
  getAllYears,
  getAllCategories,
  categoryLabels,
  type AttentionMechanism,
  type Category,
} from "@/data/attentionMechanisms";
import { MechanismCard } from "./MechanismCard";

interface TimelineProps {
  mechanisms: AttentionMechanism[];
  onCompareSelect: (id: string) => void;
  compareSelected: string[];
  showFilters?: boolean;
  variant?: "default" | "pre2017";
}

export function Timeline({
  mechanisms,
  onCompareSelect,
  compareSelected,
  showFilters = true,
  variant = "default",
}: TimelineProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filterCategory, setFilterCategory] = useState<Category | "all">("all");

  const years = getAllYears(mechanisms);
  const categories = getAllCategories(mechanisms);

  const filteredMechanisms = useMemo(() => {
    if (filterCategory === "all") return mechanisms;
    return mechanisms.filter((m) => m.category === filterCategory);
  }, [filterCategory, mechanisms]);

  const mechanismsByYear = useMemo(() => {
    const grouped: Record<number, AttentionMechanism[]> = {};
    for (const year of years) {
      const yearMechanisms = filteredMechanisms.filter((m) => m.year === year);
      if (yearMechanisms.length > 0) {
        grouped[year] = yearMechanisms;
      }
    }
    return grouped;
  }, [filteredMechanisms, years]);

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const spineClass =
    variant === "pre2017"
      ? "bg-gradient-to-b from-amber-500/20 via-amber-600 to-amber-500/20"
      : "bg-gradient-to-b from-accent/20 via-accent to-accent/20";

  const yearBadgeClass =
    variant === "pre2017"
      ? "bg-amber-600 text-white"
      : "bg-accent text-white";

  const yearDotClass =
    variant === "pre2017"
      ? "bg-amber-600 border-background"
      : "bg-accent border-background";

  return (
    <div className="relative">
      {showFilters && categories.length > 1 && (
        <div className="mb-8 flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-muted-foreground mr-2">Filter:</span>
          <button
            onClick={() => setFilterCategory("all")}
            className={`px-3 py-1.5 text-sm rounded-full transition-colors ${
              filterCategory === "all"
                ? "bg-accent text-white"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            All ({mechanisms.length})
          </button>
          {categories.map((cat) => {
            const count = mechanisms.filter((m) => m.category === cat).length;
            return (
              <button
                key={cat}
                onClick={() => setFilterCategory(cat)}
                className={`px-3 py-1.5 text-sm rounded-full transition-colors ${
                  filterCategory === cat
                    ? "bg-accent text-white"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                {categoryLabels[cat]} ({count})
              </button>
            );
          })}
        </div>
      )}

      <div className="relative">
        <div className={`absolute left-6 md:left-1/2 top-0 bottom-0 w-0.5 ${spineClass}`} />

        {Object.entries(mechanismsByYear).map(([year, yearMechanisms]) => (
          <div key={year} className="relative">
            <div className="sticky top-20 z-10 flex items-center mb-6 pt-4">
              <div
                className={`absolute left-6 md:left-1/2 w-4 h-4 -ml-2 rounded-full border-4 shadow-lg ${yearDotClass}`}
              />
              <div className="ml-14 md:ml-0 md:absolute md:left-1/2 md:-translate-x-1/2">
                <span
                  className={`inline-block px-4 py-1.5 text-lg font-bold rounded-full shadow-md ${yearBadgeClass}`}
                >
                  {year}
                </span>
              </div>
            </div>

            <div className="space-y-4 pl-14 md:pl-0 pb-8">
              {yearMechanisms.map((mechanism, mechIndex) => {
                const isEven = mechIndex % 2 === 0;
                return (
                  <div
                    key={mechanism.id}
                    className={`relative md:w-[calc(50%-2rem)] ${
                      isEven ? "md:mr-auto md:pr-8" : "md:ml-auto md:pl-8"
                    }`}
                  >
                    <div
                      className={`hidden md:block absolute top-6 w-6 h-0.5 bg-border ${
                        isEven ? "right-0" : "left-0"
                      }`}
                    />
                    <div
                      className={`hidden md:block absolute top-5 w-3 h-3 rounded-full bg-muted border-2 border-border ${
                        isEven ? "-right-1.5" : "-left-1.5"
                      }`}
                    />
                    <MechanismCard
                      mechanism={mechanism}
                      isExpanded={expandedId === mechanism.id}
                      onToggle={() => toggleExpand(mechanism.id)}
                      onCompare={() => onCompareSelect(mechanism.id)}
                      isCompareSelected={compareSelected.includes(mechanism.id)}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {filteredMechanisms.length === 0 && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No mechanisms match the selected filter.</p>
        </div>
      )}
    </div>
  );
}
