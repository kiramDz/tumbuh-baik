import React from "react";
import { InlineMath, BlockMath } from "react-katex";
// import "katex/dist/katex.min.css";

interface MathFormulaProps {
  title?: string;
  formula: string;
  inline?: boolean;
}

const MathFormula: React.FC<MathFormulaProps> = ({ title, formula, inline = false }) => {
  return (
    <div className="math-formula my-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
      {title && <h4 className="text-lg font-medium mb-2 text-gray-800">{title}</h4>}
      <div className="formula-container py-2">{inline ? <InlineMath math={formula} /> : <BlockMath math={formula} />}</div>
    </div>
  );
};

export default MathFormula;
