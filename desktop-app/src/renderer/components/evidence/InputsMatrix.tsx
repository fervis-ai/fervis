import { useState } from "react";

import type { InputResult } from "../../../fervis-api/contracts";
import { DetailToggle } from "../DetailToggle";
import { inputSummary } from "./inputSummary";
import { formatInputValue } from "./inputValues";

const DEFAULT_VISIBLE_INPUT_VALUES = 5;

export function InputsMatrix({ inputs }: { readonly inputs: readonly InputResult[] }) {
  if (inputs.length === 0) {
    return <p className="quiet">no inputs available</p>;
  }

  return (
    <div className="input-list">
      {inputs.map((input) => (
        <InputRow input={input} key={input.factResultId} />
      ))}
    </div>
  );
}

function InputRow({ input }: { readonly input: InputResult }) {
  return (
    <details className="input-card">
      <summary>
        <span>{input.factDescription}</span>
        <span>{inputSummary(input)}</span>
      </summary>
      <div className="input-card-body">
        <InputGroup label="Explicit" values={input.explicit} />
        <InputGroup label="Derived" values={input.derived} />
        <InputGroup label="Contextual" values={input.contextual} />
      </div>
    </details>
  );
}

function InputGroup({
  label,
  values,
  visibleCount = DEFAULT_VISIBLE_INPUT_VALUES
}: {
  readonly label: string;
  readonly values: readonly string[];
  readonly visibleCount?: number;
}) {
  const formattedValues = values.map(formatInputValue);
  if (formattedValues.length === 0) {
    return null;
  }

  return (
    <div className="input-group">
      <div className="input-label">{label}</div>
      <ExpandableValueList values={formattedValues} visibleCount={visibleCount} />
    </div>
  );
}

function ExpandableValueList({
  values,
  visibleCount
}: {
  readonly values: readonly string[];
  readonly visibleCount: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const visibleValues = expanded ? values : values.slice(0, visibleCount);
  const hiddenCount = values.length - visibleValues.length;

  return (
    <div className="input-values">
      <ul>
        {visibleValues.map((value, index) => (
          <li key={`${index}-${value}`}>{value}</li>
        ))}
      </ul>
      {hiddenCount > 0 ? (
        <DetailToggle
          collapsedAriaLabel="Show input details"
          collapsedLabel={`${hiddenCount} more`}
          expanded={expanded}
          expandedAriaLabel="Hide input details"
          expandedLabel="Hide details"
          onToggle={() => setExpanded((current) => !current)}
        />
      ) : null}
    </div>
  );
}
