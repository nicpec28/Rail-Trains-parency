import React, { useState } from 'react';

interface InspectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCompleteRepair: () => void;
  isRepaired: boolean;
}

export const InspectModal: React.FC<InspectModalProps> = ({
  isOpen,
  onClose,
  onCompleteRepair,
  isRepaired,
}) => {
  const [step1Done, setStep1Done] = useState(isRepaired);
  const [step2Done, setStep2Done] = useState(isRepaired);
  const [step3Done, setStep3Done] = useState(isRepaired);
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [testCycle, setTestCycle] = useState(0);

  if (!isOpen) return null;

  const handleRun5CycleTest = () => {
    setIsRunningTest(true);
    setTestCycle(1);

    const interval = setInterval(() => {
      setTestCycle((prev) => {
        if (prev >= 5) {
          clearInterval(interval);
          setIsRunningTest(false);
          setStep1Done(true);
          setStep2Done(true);
          setStep3Done(true);
          onCompleteRepair();
          return 5;
        }
        return prev + 1;
      });
    }, 600);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-lg bg-surface-container rounded-2xl border-2 border-primary-container/40 shadow-2xl flex flex-col overflow-hidden max-h-[90vh]">
        {/* Header */}
        <div className="p-4 bg-surface-container-high border-b border-outline-variant/30 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-primary-container/20 text-primary-container flex items-center justify-center border border-primary-container/30">
              <span className="material-symbols-outlined text-[20px]">build</span>
            </div>
            <div>
              <h3 className="font-headline-sm text-sm sm:text-base font-bold text-on-surface">
                Guided Track Inspection &amp; Door 4 Rectification
              </h3>
              <p className="font-body-sm text-[11px] text-outline">
                Depot Standard Operating Procedure • EN 50126 Safety Verification
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-surface-container-lowest hover:bg-surface-variant flex items-center justify-center text-outline hover:text-on-surface transition-colors cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">close</span>
          </button>
        </div>

        {/* Body */}
        <div className="p-4 overflow-y-auto flex flex-col gap-3.5">
          {/* Step 1 */}
          <div className="p-3 rounded-xl bg-surface-container-low border border-outline-variant/30 flex items-start gap-3">
            <input
              type="checkbox"
              id="step1"
              checked={step1Done}
              onChange={(e) => setStep1Done(e.target.checked)}
              className="mt-0.5 rounded border-outline-variant text-primary-container focus:ring-primary-container w-4 h-4 cursor-pointer"
            />
            <label htmlFor="step1" className="min-w-0 flex-1 cursor-pointer">
              <div className="flex items-center justify-between">
                <span className="font-headline-sm text-xs font-bold text-on-surface">
                  1. Track Clearance &amp; Debris Removal
                </span>
                <span className="font-label-sm text-[9px] px-2 py-0.5 rounded bg-primary-container/15 text-primary-container font-semibold">
                  Required
                </span>
              </div>
              <p className="font-body-sm text-[11px] text-outline mt-0.5">
                Inspected Door 4 lower guide rail runner. Found 1x ballast aggregate pebble wedged between slide seal lip and guide track. Cleared and vacuumed.
              </p>
            </label>
          </div>

          {/* Step 2 */}
          <div className="p-3 rounded-xl bg-surface-container-low border border-outline-variant/30 flex items-start gap-3">
            <input
              type="checkbox"
              id="step2"
              checked={step2Done}
              onChange={(e) => setStep2Done(e.target.checked)}
              className="mt-0.5 rounded border-outline-variant text-primary-container focus:ring-primary-container w-4 h-4 cursor-pointer"
            />
            <label htmlFor="step2" className="min-w-0 flex-1 cursor-pointer">
              <div className="flex items-center justify-between">
                <span className="font-headline-sm text-xs font-bold text-on-surface">
                  2. Drive Belt Tension &amp; Motor Actuator Linkage
                </span>
                <span className="font-label-sm text-[9px] px-2 py-0.5 rounded bg-primary-container/15 text-primary-container font-semibold">
                  Checked
                </span>
              </div>
              <p className="font-body-sm text-[11px] text-outline mt-0.5">
                Tested toothed timing belt deflection. Tension verified at 46.5 N (tolerance: 40–50 N). Motor pinion gear mesh nominal with zero mechanical binding.
              </p>
            </label>
          </div>

          {/* Step 3 */}
          <div className="p-3 rounded-xl bg-surface-container-low border border-outline-variant/30 flex items-start gap-3">
            <input
              type="checkbox"
              id="step3"
              checked={step3Done}
              onChange={(e) => setStep3Done(e.target.checked)}
              className="mt-0.5 rounded border-outline-variant text-primary-container focus:ring-primary-container w-4 h-4 cursor-pointer"
            />
            <label htmlFor="step3" className="min-w-0 flex-1 cursor-pointer">
              <div className="flex items-center justify-between">
                <span className="font-headline-sm text-xs font-bold text-on-surface">
                  3. Optical Clearance Calibration
                </span>
                <span className="font-label-sm text-[9px] px-2 py-0.5 rounded bg-primary-container/15 text-primary-container font-semibold">
                  Laser Verified
                </span>
              </div>
              <p className="font-body-sm text-[11px] text-outline mt-0.5">
                Laser micrometer clearance aligned to 0.35 mm across leaf overlap. Micro-switch limit sensors verified for sensitive edge safety trigger.
              </p>
            </label>
          </div>

          {/* Automated 5-Stroke Test Simulator */}
          <div className="p-3.5 rounded-xl bg-surface-container-lowest border-2 border-primary-container/30 flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
              <span className="font-headline-sm text-xs font-bold text-primary">
                4. Automated 5-Stroke Cycle Test
              </span>
              <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-primary-container/20 text-primary-container font-bold">
                {testCycle > 0 ? `Cycle ${testCycle} / 5` : isRepaired ? 'Verified 5/5' : 'Ready'}
              </span>
            </div>

            {/* Visual Progress Bar */}
            <div className="w-full bg-surface-container h-2.5 rounded-full overflow-hidden border border-outline-variant/30">
              <div
                className="h-full bg-gradient-to-r from-primary-container to-secondary-fixed-dim transition-all duration-300"
                style={{ width: `${(testCycle / 5) * 100 || (isRepaired ? 100 : 0)}%` }}
              ></div>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center text-[10px] font-mono py-1">
              <div className="p-1 rounded bg-surface-container border border-outline-variant/20">
                <div className="text-outline">Thrust</div>
                <div className="font-bold text-secondary-fixed-dim">
                  {testCycle > 0 || isRepaired ? '185 N' : '340 N'}
                </div>
              </div>
              <div className="p-1 rounded bg-surface-container border border-outline-variant/20">
                <div className="text-outline">Cycle Time</div>
                <div className="font-bold text-secondary-fixed-dim">
                  {testCycle > 0 || isRepaired ? '2.8 s' : '4.8 s'}
                </div>
              </div>
              <div className="p-1 rounded bg-surface-container border border-outline-variant/20">
                <div className="text-outline">Power Draw</div>
                <div className="font-bold text-secondary-fixed-dim">
                  {testCycle > 0 || isRepaired ? '5.4 A' : '14.2 A'}
                </div>
              </div>
            </div>

            <button
              type="button"
              disabled={isRunningTest}
              onClick={handleRun5CycleTest}
              className={`w-full py-2 px-3 rounded-lg font-headline-sm text-xs font-bold flex items-center justify-center gap-2 transition-all cursor-pointer ${
                isRunningTest
                  ? 'bg-surface-container text-outline'
                  : 'bg-primary-container hover:bg-primary-fixed-dim text-on-primary-container shadow-md'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">
                {isRunningTest ? 'sync' : 'play_arrow'}
              </span>
              <span>
                {isRunningTest
                  ? `Cycling Door 4 (Stroke ${testCycle} of 5)...`
                  : isRepaired
                  ? 'Re-Run Automated 5-Stroke Test'
                  : 'Run Automated 5-Stroke Cycle Test'}
              </span>
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 bg-surface-container-high border-t border-outline-variant/30 flex items-center justify-between">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg bg-surface-container hover:bg-surface-variant text-on-surface font-headline-sm text-xs font-semibold"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              onCompleteRepair();
              onClose();
            }}
            className="px-4 py-1.5 rounded-lg bg-secondary-fixed-dim hover:bg-secondary-fixed text-on-secondary font-headline-sm text-xs font-bold flex items-center gap-1.5 shadow-md"
          >
            <span className="material-symbols-outlined text-[16px]">check_circle</span>
            <span>Sign Off &amp; Clear Fault</span>
          </button>
        </div>
      </div>
    </div>
  );
};
