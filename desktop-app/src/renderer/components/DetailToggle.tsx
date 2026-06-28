export function DetailToggle({
  expanded,
  collapsedLabel,
  expandedLabel,
  collapsedAriaLabel,
  expandedAriaLabel,
  className,
  onToggle
}: {
  readonly expanded: boolean;
  readonly collapsedLabel: string;
  readonly expandedLabel: string;
  readonly collapsedAriaLabel: string;
  readonly expandedAriaLabel: string;
  readonly className?: string;
  readonly onToggle: () => void;
}) {
  return (
    <button
      aria-label={expanded ? expandedAriaLabel : collapsedAriaLabel}
      className={className === undefined ? "detail-toggle" : `detail-toggle ${className}`}
      type="button"
      onClick={onToggle}
    >
      <span>{expanded ? "−" : "+"}</span>
      {expanded ? expandedLabel : collapsedLabel}
    </button>
  );
}
