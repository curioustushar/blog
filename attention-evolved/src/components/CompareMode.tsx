"use client";

import { AttentionMechanism, getById } from "@/data/attentionMechanisms";
import { CategoryBadge } from "./CategoryBadge";
import { MathDisplay } from "./MathDisplay";

interface CompareModeProps {
  selectedIds: string[];
  onClose: () => void;
  onRemove: (id: string) => void;
}

export function CompareMode({ selectedIds, onClose, onRemove }: CompareModeProps) {
  const mechanisms = selectedIds
    .map((id) => getById(id))
    .filter((m): m is AttentionMechanism => m !== undefined);

  if (mechanisms.length < 2) {
    return (
      <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-8 max-w-md text-center">
          <p className="text-lg font-medium mb-4">
            Select 2 mechanisms to compare
          </p>
          <p className="text-muted-foreground mb-6">
            Click the &quot;Compare&quot; button on any mechanism card to add it to comparison.
          </p>
          <button
            onClick={onClose}
            className="px-6 py-2 bg-accent text-white rounded-lg hover:bg-accent/90"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  const [mech1, mech2] = mechanisms;

  const formatDate = (dateStr: string) => {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const [year, month] = dateStr.split("-").map(Number);
    return `${months[month - 1]} ${year}`;
  };

  const compareFields = [
    { label: "Category", render: (m: AttentionMechanism) => <CategoryBadge category={m.category} size="sm" /> },
    { label: "Date", render: (m: AttentionMechanism) => formatDate(m.date) },
    { label: "Time Complexity", render: (m: AttentionMechanism) => <span className="font-mono">{m.complexity.time}</span> },
    { label: "Memory Complexity", render: (m: AttentionMechanism) => <span className="font-mono">{m.complexity.memory}</span> },
    { label: "KV Cache Size", render: (m: AttentionMechanism) => <span className="capitalize">{m.kvCache.size}</span> },
    { label: "Extrapolation", render: (m: AttentionMechanism) => <span className="capitalize">{m.contextBehavior.extrapolation}</span> },
  ];

  return (
    <div className="fixed inset-0 z-50 bg-black/50 overflow-auto">
      <div className="min-h-full flex items-start justify-center p-4 py-8">
        <div className="bg-white dark:bg-zinc-900 rounded-xl w-full max-w-6xl shadow-2xl">
          <header className="sticky top-0 bg-white dark:bg-zinc-900 border-b border-border p-4 flex items-center justify-between rounded-t-xl z-10">
            <h2 className="text-xl font-bold">Compare Mechanisms</h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-muted rounded-lg transition-colors"
              aria-label="Close"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </header>

          <div className="p-6">
            <div className="grid grid-cols-2 gap-6 mb-8">
              {[mech1, mech2].map((mech, idx) => (
                <div key={mech.id} className="relative">
                  <button
                    onClick={() => onRemove(mech.id)}
                    className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs flex items-center justify-center hover:bg-red-600 z-10"
                    aria-label="Remove from comparison"
                  >
                    ×
                  </button>
                  <div className={`p-4 rounded-lg border-2 ${idx === 0 ? "border-blue-500 bg-blue-50 dark:bg-blue-950" : "border-purple-500 bg-purple-50 dark:bg-purple-950"}`}>
                    <h3 className="font-bold text-lg">{mech.name}</h3>
                    <p className="text-sm text-muted-foreground mt-1">{mech.intuition}</p>
                  </div>
                </div>
              ))}
            </div>

            <table className="w-full border-collapse mb-8">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 text-sm font-semibold text-muted-foreground">Property</th>
                  <th className="text-left p-3 text-sm font-semibold text-blue-600">{mech1.shortName || mech1.name}</th>
                  <th className="text-left p-3 text-sm font-semibold text-purple-600">{mech2.shortName || mech2.name}</th>
                </tr>
              </thead>
              <tbody>
                {compareFields.map((field) => (
                  <tr key={field.label} className="border-b border-border">
                    <td className="p-3 text-sm font-medium">{field.label}</td>
                    <td className="p-3 text-sm">{field.render(mech1)}</td>
                    <td className="p-3 text-sm">{field.render(mech2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="grid grid-cols-2 gap-6 mb-8">
              <div>
                <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">Advantages</h4>
                <div className="grid grid-cols-1 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-950/50 p-3 rounded-lg">
                    <p className="text-xs font-semibold text-blue-600 mb-2">{mech1.shortName || mech1.name}</p>
                    <ul className="space-y-1">
                      {mech1.advantages.map((adv, i) => (
                        <li key={i} className="text-sm flex items-start gap-1">
                          <span className="text-green-500">+</span>
                          {adv}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="bg-purple-50 dark:bg-purple-950/50 p-3 rounded-lg">
                    <p className="text-xs font-semibold text-purple-600 mb-2">{mech2.shortName || mech2.name}</p>
                    <ul className="space-y-1">
                      {mech2.advantages.map((adv, i) => (
                        <li key={i} className="text-sm flex items-start gap-1">
                          <span className="text-green-500">+</span>
                          {adv}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">Disadvantages</h4>
                <div className="grid grid-cols-1 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-950/50 p-3 rounded-lg">
                    <p className="text-xs font-semibold text-blue-600 mb-2">{mech1.shortName || mech1.name}</p>
                    <ul className="space-y-1">
                      {mech1.disadvantages.map((dis, i) => (
                        <li key={i} className="text-sm flex items-start gap-1">
                          <span className="text-red-500">-</span>
                          {dis}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="bg-purple-50 dark:bg-purple-950/50 p-3 rounded-lg">
                    <p className="text-xs font-semibold text-purple-600 mb-2">{mech2.shortName || mech2.name}</p>
                    <ul className="space-y-1">
                      {mech2.disadvantages.map((dis, i) => (
                        <li key={i} className="text-sm flex items-start gap-1">
                          <span className="text-red-500">-</span>
                          {dis}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {(mech1.equation || mech2.equation) && (
              <div className="mb-8">
                <h4 className="font-semibold mb-4">Equations</h4>
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-muted p-4 rounded-lg">
                    <p className="text-xs font-semibold text-blue-600 mb-2">{mech1.shortName || mech1.name}</p>
                    {mech1.equation ? (
                      <MathDisplay math={mech1.equation} display />
                    ) : (
                      <p className="text-sm text-muted-foreground italic">No equation specified</p>
                    )}
                  </div>
                  <div className="bg-muted p-4 rounded-lg">
                    <p className="text-xs font-semibold text-purple-600 mb-2">{mech2.shortName || mech2.name}</p>
                    {mech2.equation ? (
                      <MathDisplay math={mech2.equation} display />
                    ) : (
                      <p className="text-sm text-muted-foreground italic">No equation specified</p>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div>
              <h4 className="font-semibold mb-4">Use Cases</h4>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <div className="bg-blue-50 dark:bg-blue-950/50 p-4 rounded-lg mb-4">
                    <p className="text-xs font-semibold text-blue-600 mb-2">{mech1.shortName || mech1.name}</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs font-medium text-green-600 mb-1">Good for:</p>
                        <ul className="text-sm space-y-1">
                          {mech1.useCases.good.slice(0, 3).map((u, i) => (
                            <li key={i}>• {u}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-xs font-medium text-red-600 mb-1">Poor for:</p>
                        <ul className="text-sm space-y-1">
                          {mech1.useCases.poor.slice(0, 3).map((u, i) => (
                            <li key={i}>• {u}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="bg-purple-50 dark:bg-purple-950/50 p-4 rounded-lg mb-4">
                    <p className="text-xs font-semibold text-purple-600 mb-2">{mech2.shortName || mech2.name}</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs font-medium text-green-600 mb-1">Good for:</p>
                        <ul className="text-sm space-y-1">
                          {mech2.useCases.good.slice(0, 3).map((u, i) => (
                            <li key={i}>• {u}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-xs font-medium text-red-600 mb-1">Poor for:</p>
                        <ul className="text-sm space-y-1">
                          {mech2.useCases.poor.slice(0, 3).map((u, i) => (
                            <li key={i}>• {u}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
