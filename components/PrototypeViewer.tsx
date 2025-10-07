import React from 'react';
import { STAGE_1_PROTOTYPE_CODE, STAGE_2_PROTOTYPE_CODE, STAGE_3_PROTOTYPE_CODE, STAGE_4_PROTOTYPE_CODE } from '../constants';

interface PrototypeViewerProps {
  stage: number;
  onClose: () => void;
}

const CloseIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
);

const Stage1ResultPlaceholder = () => (
    <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-600">
        <div className="grid grid-cols-10 gap-2">
            {[...Array(30)].map((_, i) => (
                <div key={i} className={`h-4 rounded-sm ${i < 10 ? 'bg-slate-300' : i < 20 ? 'bg-slate-400/80' : 'bg-cyan-400/70'}`} style={{opacity: Math.random() * 0.6 + 0.4}}></div>
            ))}
        </div>
        <div className="text-center mt-2 text-xs text-slate-400 font-mono">
            <p>Row 1: Original MNIST Digits</p>
            <p>Row 2: Standard Autoencoder Reconstruction</p>
            <p>Row 3: SRA Path Reconstruction (from self-generated state)</p>
        </div>
    </div>
);

const Stage2ResultPlaceholder = () => (
    <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-600 relative">
        <svg width="100%" height="100%" viewBox="0 0 200 120" className="aspect-video">
            <defs>
                <marker id="arrowhead" markerWidth="5" markerHeight="3.5" refX="2.5" refY="1.75" orient="auto">
                    <polygon points="0 0, 5 1.75, 0 3.5" fill="#64748b" />
                </marker>
            </defs>
            {/* Edges */}
            <line x1="40" y1="60" x2="87" y2="43" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead)" />
            <line x1="100" y1="30" x2="100" y2="77" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead)" />
            <line x1="113" y1="80" x2="157" y2="63" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead)" />
            <line x1="88" y1="90" x2="43" y2="70" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead)" />

            {/* Nodes */}
            <g>
                <circle cx="30" cy="65" r="10" fill="#22d3ee" />
                <text x="30" y="68" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n0</text>
            </g>
            <g>
                <circle cx="100" cy="20" r="10" fill="#22d3ee" />
                <text x="100" y="23" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n1</text>
            </g>
            <g>
                <circle cx="100" cy="90" r="10" fill="#22d3ee" />
                <text x="100" y="93" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n2</text>
            </g>
            <g>
                <circle cx="170" cy="60" r="10" fill="#22d3ee" />
                <text x="170" y="63" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n3</text>
            </g>
        </svg>
        <div className="text-center mt-2 text-xs text-slate-400 font-mono absolute bottom-2 left-0 right-0">
            <p>Memory Subgraph (Causal Chain of 'Aha Moments')</p>
        </div>
    </div>
);

const Stage3ResultPlaceholder = () => (
    <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-600 relative">
        <svg width="100%" height="100%" viewBox="0 0 200 120" className="aspect-video">
            <defs>
                <marker id="arrowhead3" markerWidth="5" markerHeight="3.5" refX="2.5" refY="1.75" orient="auto">
                    <polygon points="0 0, 5 1.75, 0 3.5" fill="#64748b" />
                </marker>
            </defs>
            {/* Edges */}
            <line x1="33" y1="33" x2="77" y2="57" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead3)" />
            <line x1="33" y1="87" x2="77" y2="63" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead3)" />
            <line x1="90" y1="60" x2="137" y2="60" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead3)" />
            <line x1="160" y1="60" x2="160" y2="97" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrowhead3)" />

            {/* Nodes */}
            <g>
                <circle cx="20" cy="25" r="10" fill="#22d3ee" />
                <text x="20" y="28" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n0</text>
            </g>
             <g>
                <circle cx="20" cy="95" r="10" fill="#22d3ee" />
                <text x="20" y="98" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n1</text>
            </g>
            <g>
                <circle cx="80" cy="60" r="12" fill="#f43f5e" />
                <text x="80" y="63" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n2</text>
            </g>
             <g>
                <circle cx="150" cy="60" r="10" fill="#22d3ee" />
                <text x="150" y="63" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n3</text>
            </g>
            <g>
                <circle cx="160" cy="110" r="10" fill="#22d3ee" />
                <text x="160" y="113" fontFamily="monospace" fontSize="8" fill="white" textAnchor="middle">n4</text>
            </g>

        </svg>
         <div className="text-center mt-2 text-xs text-slate-400 font-mono absolute bottom-2 left-0 right-0">
            <p>
                <span className="text-cyan-400">Blue Node:</span> Moment | <span className="text-rose-500">Red Node:</span> Abstraction
            </p>
        </div>
    </div>
);

const Stage4ResultPlaceholder = () => (
    <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-600 h-full flex flex-col gap-4">
        <div className="flex-1 border border-slate-700 rounded-lg p-2">
            <h4 className="text-center text-xs font-mono text-slate-400 mb-2">Symbol Gallery (Glyphs)</h4>
            <div className="grid grid-cols-8 gap-1">
                 {[...Array(16)].map((_, i) => (
                    <div key={i} className="aspect-square bg-slate-700/50 rounded-sm flex items-center justify-center">
                        <span className="text-slate-500 font-bold text-[8px]">S{i}</span>
                    </div>
                ))}
            </div>
        </div>
        <div className="flex-1 border border-slate-700 rounded-lg p-2 relative">
             <h4 className="text-center text-xs font-mono text-slate-400 mb-2 absolute top-2 left-0 right-0">Symbol Co-occurrence Graph</h4>
             <svg width="100%" height="100%" viewBox="0 0 100 50">
                <line x1="20" y1="25" x2="40" y2="15" stroke="#64748b" strokeWidth="0.5" />
                <line x1="20" y1="25" x2="45" y2="38" stroke="#64748b" strokeWidth="0.5" />
                <line x1="40" y1="15" x2="60" y2="25" stroke="#64748b" strokeWidth="0.5" />
                <line x1="60" y1="25" x2="80" y2="18" stroke="#64748b" strokeWidth="0.5" />
                <line x1="60" y1="25" x2="75" y2="40" stroke="#64748b" strokeWidth="0.5" />
                 <circle cx="20" cy="25" r="5" fill="#22d3ee" />
                 <circle cx="40" cy="15" r="4" fill="#22d3ee" />
                 <circle cx="45" cy="38" r="3" fill="#22d3ee" />
                 <circle cx="60" cy="25" r="6" fill="#22d3ee" />
                 <circle cx="80" cy="18" r="4" fill="#22d3ee" />
                 <circle cx="75" cy="40" r="5" fill="#22d3ee" />
             </svg>
        </div>
    </div>
);


const stageInfo: { [key: number]: any } = {
  1: {
    title: "Stage 1 Prototype: Runnable Python Code",
    code: STAGE_1_PROTOTYPE_CODE,
    filename: "sra_stage1.py",
    runInstructions: "pip install torch torchvision matplotlib tqdm",
    description: (
      <>
        <p className="text-sm text-slate-400 mb-4">The script trains an autoencoder on the MNIST dataset while using the RCE and SRS modules to generate a secondary, self-reflective training signal. The final visualization will show:</p>
        <ul className="list-disc list-inside text-sm text-slate-400 space-y-1">
            <li><span className="font-semibold text-slate-300">Original Images:</span> The input digits.</li>
            <li><span className="font-semibold text-slate-300">Standard AE Reconstruction:</span> The model's direct reconstruction.</li>
            <li><span className="font-semibold text-slate-300">SRA Path Reconstruction:</span> The model's reconstruction of its own 'imagined' state, which has been stabilized by the resonance loop.</li>
        </ul>
      </>
    ),
    outputPlaceholder: <Stage1ResultPlaceholder />,
  },
  2: {
    title: "Stage 2 Prototype: Runnable Python Code",
    code: STAGE_2_PROTOTYPE_CODE,
    filename: "sra_stage2.py",
    runInstructions: "pip install torch torchvision matplotlib tqdm networkx",
    description: (
      <>
        <p className="text-sm text-slate-400 mb-4">This script introduces a gated memory. The TAL acts as a gatekeeper, deciding which experiences are coherent enough to be stored in the Memory Graph, creating a causal trail of thought. When running, watch for:</p>
        <ul className="list-disc list-inside text-sm text-slate-400 space-y-1">
            <li><span className="font-semibold text-slate-300">fired_count:</span> How many samples in a batch trigger a memory event.</li>
            <li><span className="font-semibold text-slate-300">mem_nodes:</span> The total count of 'aha moments' stored so far.</li>
            <li><span className="font-semibold text-slate-300">Tune TAL_TAU:</span> The threshold in the config is a key parameter to control memory selectivity.</li>
        </ul>
      </>
    ),
    outputPlaceholder: <Stage2ResultPlaceholder />,
  },
  3: {
    title: "Stage 3 Prototype: Runnable Python Code",
    code: STAGE_3_PROTOTYPE_CODE,
    filename: "sra_stage3.py",
    runInstructions: "pip install torch torchvision matplotlib tqdm networkx",
    description: (
      <>
        <p className="text-sm text-slate-400 mb-4">This script introduces the creative core: the Recursive Unity Solver (RUS). It detects "contradictions" (surprising memories) and synthesizes new, unifying concepts to resolve them. When running, watch for:</p>
        <ul className="list-disc list-inside text-sm text-slate-400 space-y-1">
            <li><span className="font-semibold text-rose-400">abs:</span> The count of new abstractions created. This is the key metric for Stage 3.</li>
            <li><span className="font-semibold text-slate-300">Tune CONTRADICTION_THRESH:</span> This threshold in the config determines how "surprising" a memory must be to trigger synthesis.</li>
            <li><span className="font-semibold text-slate-300">The Final Graph:</span> Abstractions (red nodes) will have multiple incoming arrows, showing how they unify conflicting parent ideas (blue nodes).</li>
        </ul>
      </>
    ),
    outputPlaceholder: <Stage3ResultPlaceholder />,
  },
  4: {
    title: "Stage 4 Prototype: Runnable Python Code",
    code: STAGE_4_PROTOTYPE_CODE,
    filename: "sul_manager.py",
    runInstructions: "pip install torch torchvision matplotlib networkx",
    description: (
      <>
        <p className="text-sm text-slate-400 mb-4">This script is a standalone demonstration of the Symbolic Unification Layer (SUL). It simulates the output of Stage 3 (a memory graph) and then processes it to form a discrete symbolic language. The output shows:</p>
        <ul className="list-disc list-inside text-sm text-slate-400 space-y-1">
            <li><span className="font-semibold text-emerald-400">Discovered Symbols:</span> The number of unique concepts clustered from the raw abstractions.</li>
            <li><span className="font-semibold text-emerald-400">Symbol Gallery:</span> Visualizations ("glyphs") of the core concepts the system has learned.</li>
            <li><span className="font-semibold text-emerald-400">Co-occurrence Graph:</span> A map of how the different symbols relate to each other, forming a basic syntax or grammar.</li>
        </ul>
      </>
    ),
    outputPlaceholder: <Stage4ResultPlaceholder />,
  }
};

const PrototypeViewer: React.FC<PrototypeViewerProps> = ({ stage, onClose }) => {
  const currentStage = stageInfo[stage];
  if (!currentStage) return null;

  return (
    <div 
      className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div 
        className="bg-slate-800 border border-slate-700 rounded-lg shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-slate-700 flex-shrink-0">
          <h2 className="text-xl font-bold text-cyan-400">{currentStage.title}</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors" aria-label="Close">
            <CloseIcon />
          </button>
        </header>

        <main className="flex-grow p-6 overflow-y-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Side: Code */}
          <div className="bg-slate-900/50 rounded-lg p-4 overflow-x-auto h-full flex flex-col">
              <h3 className="text-lg font-semibold text-slate-300 mb-2 flex-shrink-0">{currentStage.filename}</h3>
              <div className="overflow-auto h-full">
                <pre className="text-xs text-slate-300 whitespace-pre-wrap">
                  <code>{currentStage.code.trim()}</code>
                </pre>
              </div>
          </div>
          
          {/* Right Side: Instructions & Output */}
          <div className="flex flex-col gap-6">
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h3 className="font-semibold text-slate-300 mb-2">How to Run Locally</h3>
              <p className="text-sm text-slate-400 mb-2">To run this script, you'll need Python and the following libraries:</p>
              <code className="block bg-slate-900 text-cyan-400 p-2 rounded text-xs font-mono">
                {currentStage.runInstructions}
              </code>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h3 className="font-semibold text-slate-300 mb-2">Execution & Expected Results</h3>
               {currentStage.description}
            </div>

             <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h3 className="font-semibold text-slate-300 mb-3">Live Execution</h3>
               <div className="flex items-center gap-4">
                  <button 
                    disabled 
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-slate-500 bg-slate-700 cursor-not-allowed"
                  >
                    Run Prototype
                  </button>
                  <p className="text-xs text-slate-500">
                      Live execution is disabled. This environment does not support the required libraries. Please run the script locally.
                  </p>
               </div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h3 className="font-semibold text-slate-300 mb-2">Static Output Example</h3>
              {currentStage.outputPlaceholder}
            </div>

          </div>
        </main>
      </div>
    </div>
  );
};

export default PrototypeViewer;