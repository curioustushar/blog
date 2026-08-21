import { Category, categoryLabels, categoryColors } from "@/data/attentionMechanisms";

interface CategoryBadgeProps {
  category: Category;
  size?: "sm" | "md";
}

export function CategoryBadge({ category, size = "md" }: CategoryBadgeProps) {
  const color = categoryColors[category];
  const label = categoryLabels[category];
  
  const sizeClasses = size === "sm" 
    ? "text-xs px-2 py-0.5" 
    : "text-sm px-3 py-1";

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium ${sizeClasses}`}
      style={{
        backgroundColor: `${color}20`,
        color: color,
        border: `1px solid ${color}40`,
      }}
    >
      {label}
    </span>
  );
}
