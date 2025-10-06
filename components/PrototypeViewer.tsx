import React from 'react';
import { STAGE_1_PROTOTYPE_CODE } from '../constants';

interface PrototypeViewerProps {
  onClose: () => void;
}

const CloseIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
);

const PrototypeResultPlaceholder = () => (
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


const PrototypeViewer: React.FC<PrototypeViewerProps> = ({ onClose }) => {
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
          <h2 className="text-xl font-bold text-cyan-400">Stage 1 Prototype: Runnable Python Code</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors" aria-label="Close">
            <CloseIcon />
          </button>
        </header>

        <main className="flex-grow p-6 overflow-y-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Side: Code */}
          <div className="bg-slate-900/50 rounded-lg p-4 overflow-x-auto h-full">
              <h3 className="text-lg font-semibold text-slate-300 mb-2">sra_prototype.py</h3>
              <pre className="text-xs text-slate-300 whitespace-pre-wrap">
                <code>{STAGE_1_PROTOTYPE_CODE.trim()}</code>
              </pre>
          </div>
          
          {/* Right Side: Instructions & Output */}
          <div className="flex flex-col gap-6">
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h3 className="font-semibold text-slate-300 mb-2">How to Run Locally</h3>
              <p className="text-sm text-slate-400 mb-2">To run this script, you'll need Python and the following libraries:</p>
              <code className="block bg-slate-900 text-cyan-400 p-2 rounded text-xs font-mono">
                pip install torch torchvision matplotlib tqdm
              </code>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h3 className="font-semibold text-slate-300 mb-2">Execution & Expected Results</h3>
               <p className="text-sm text-slate-400 mb-4">The script trains an autoencoder on the MNIST dataset while using the RCE and SRS modules to generate a secondary, self-reflective training signal. The final visualization will show:</p>
                <ul className="list-disc list-inside text-sm text-slate-400 space-y-1">
                    <li><span className="font-semibold text-slate-300">Original Images:</span> The input digits.</li>
                    <li><span className="font-semibold text-slate-300">Standard AE Reconstruction:</span> The model's direct reconstruction.</li>
                    <li><span className="font-semibold text-slate-300">SRA Path Reconstruction:</span> The model's reconstruction of its own 'imagined' state, which has been stabilized by the resonance loop.</li>
                </ul>
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
                      Live execution is disabled. This environment does not support the required PyTorch library. Please run the script locally to see it in action.
                  </p>
               </div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h3 className="font-semibold text-slate-300 mb-2">Static Output Example</h3>
              <PrototypeResultPlaceholder />
            </div>

          </div>
        </main>
      </div>
    </div>
  );
};

export default PrototypeViewer;
