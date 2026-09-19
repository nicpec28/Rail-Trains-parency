import React, { useState } from 'react';
import { SubsystemDetail, DepotInfo } from '../types';

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  subsystem: SubsystemDetail;
  depot: DepotInfo;
  isRepaired: boolean;
}

export const ReportModal: React.FC<ReportModalProps> = ({
  isOpen,
  onClose,
  subsystem,
  depot,
  isRepaired,
}) => {
  const [downloading, setDownloading] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  if (!isOpen) return null;

  const handlePrintOrDownload = () => {
    setDownloading(true);
    setTimeout(() => {
      setDownloading(false);
      setDownloaded(true);
      setTimeout(() => {
        window.print();
      }, 300);
    }, 800);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/85 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-2xl bg-surface-container rounded-2xl border-2 border-primary-container/50 shadow-2xl flex flex-col overflow-hidden max-h-[92vh]">
        {/* Header */}
        <div className="p-4 bg-surface-container-high border-b border-outline-variant/30 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-primary-container/20 text-primary-container flex items-center justify-center border border-primary-container/30">
              <span className="material-symbols-outlined text-[20px]">description</span>
            </div>
            <div>
              <h3 className="font-headline-sm text-sm sm:text-base font-bold text-on-surface">
                Prediction &amp; Diagnostic Report
              </h3>
              <p className="font-body-sm text-[11px] text-outline">
                Doc Ref: EN-50126-SMRT-2025-08429 • Certified Official Document
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

        {/* Printable Report Document Sheet */}
        <div className="p-4 sm:p-6 overflow-y-auto bg-surface-container-lowest text-on-surface font-body-sm flex flex-col gap-4 text-xs">
          {/* Document Header */}
          <div className="border-b border-outline-variant/30 pb-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-headline-md text-sm font-bold text-primary">
                  SMRT TRAINS • FLEET ENGINEERING DIVISION
                </span>
              </div>
              <div className="text-[11px] text-outline font-label-sm mt-0.5">
                Depot Facility: {depot.name} ({depot.line})
              </div>
            </div>
            <div className="text-right font-mono text-[11px] text-outline">
              <div>Date: 24 Oct 2025 • 09:41 SGT</div>
              <div className="text-secondary-fixed-dim font-bold">Standard: CENELEC EN 50126 / RAMS</div>
            </div>
          </div>

          {/* Asset & Subsystem Information Table */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 bg-surface-container p-3 rounded-xl border border-outline-variant/30 font-mono text-[11px]">
            <div>
              <div className="text-outline text-[10px]">ROLLING STOCK</div>
              <div className="font-bold text-on-surface">Kawasaki C151B (#541)</div>
            </div>
            <div>
              <div className="text-outline text-[10px]">SUBSYSTEM</div>
              <div className="font-bold text-primary">{subsystem.name} Unit</div>
            </div>
            <div>
              <div className="text-outline text-[10px]">INSPECTOR</div>
              <div className="font-bold text-on-surface">Alex Tan (Lead Tech)</div>
            </div>
            <div>
              <div className="text-outline text-[10px]">STATUS</div>
              <div className={`font-bold ${isRepaired ? 'text-secondary-fixed-dim' : 'text-error'}`}>
                {isRepaired ? 'CLEARED & NOMINAL' : 'ANOMALY DETECTED'}
              </div>
            </div>
          </div>

          {/* AI Telemetry Analysis & Findings */}
          <div className="flex flex-col gap-2">
            <h4 className="font-headline-sm text-xs font-bold text-primary uppercase tracking-wider">
              1. Telemetry Inference &amp; Anomaly Detection Results
            </h4>
            <div className="p-3 rounded-lg bg-surface-container/60 border border-outline-variant/20 flex flex-col gap-1.5 leading-relaxed text-[11px]">
              <p>
                <strong className="text-on-surface">Input Data Feed:</strong> {subsystem.defaultFile.name} ({subsystem.defaultFile.sizeMb} MB) processed across 10,000 sampling points at 50 Hz.
              </p>
              <p>
                <strong className="text-on-surface">Inference Summary:</strong>{' '}
                {isRepaired
                  ? 'All mechanical resistance curves and current draw signatures returned to nominal envelope.'
                  : subsystem.alertDescription}
              </p>
            </div>
          </div>

          {/* Quantitative Metrics Matrix */}
          <div className="flex flex-col gap-2">
            <h4 className="font-headline-sm text-xs font-bold text-primary uppercase tracking-wider">
              2. Parameter Threshold Matrix
            </h4>
            <table className="w-full border-collapse text-left font-mono text-[11px]">
              <thead>
                <tr className="border-b border-outline-variant/40 text-outline text-[10px]">
                  <th className="py-1.5">PARAMETER</th>
                  <th className="py-1.5">OBSERVED VALUE</th>
                  <th className="py-1.5">SAFETY LIMIT</th>
                  <th className="py-1.5">VARIANCE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/20">
                <tr>
                  <td className="py-1.5 font-bold">Closing Thrust Peak</td>
                  <td className={`py-1.5 ${isRepaired ? 'text-secondary-fixed-dim' : 'text-error'}`}>
                    {isRepaired ? '185 N' : '340 N'}
                  </td>
                  <td className="py-1.5 text-outline">≤ 200 N</td>
                  <td className="py-1.5">{isRepaired ? '-7.5%' : '+70.0% [FAIL]'}</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-bold">Closing Stroke Time</td>
                  <td className={`py-1.5 ${isRepaired ? 'text-secondary-fixed-dim' : 'text-error'}`}>
                    {isRepaired ? '2.8 s' : '4.8 s'}
                  </td>
                  <td className="py-1.5 text-outline">2.7 – 3.2 s</td>
                  <td className="py-1.5">{isRepaired ? 'Nominal' : '+50.0% [DELAY]'}</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-bold">Motor Current Draw</td>
                  <td className={`py-1.5 ${isRepaired ? 'text-secondary-fixed-dim' : 'text-error'}`}>
                    {isRepaired ? '5.4 A' : '14.2 A'}
                  </td>
                  <td className="py-1.5 text-outline">4.5 – 7.0 A</td>
                  <td className="py-1.5">{isRepaired ? 'Nominal' : '+102.8% [OVERCURRENT]'}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Action Log / Recommendations */}
          <div className="flex flex-col gap-2">
            <h4 className="font-headline-sm text-xs font-bold text-primary uppercase tracking-wider">
              3. Maintenance Action Items &amp; Sign-off
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {subsystem.recommendations.map((rec) => (
                <div key={rec.id} className="p-2.5 rounded bg-surface-container border border-outline-variant/20">
                  <div className="font-headline-sm text-[11px] font-bold text-on-surface">
                    {rec.id}. {rec.title}
                  </div>
                  <div className="text-[10px] text-outline mt-0.5 leading-snug">{rec.description}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Signatures & Certification */}
          <div className="mt-2 pt-3 border-t border-outline-variant/30 flex flex-col sm:flex-row items-center justify-between gap-3 text-[10px] font-mono text-outline">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px] text-secondary-fixed-dim">verified</span>
              <div>
                <div className="font-bold text-on-surface">CENELEC EN 50126 COMPLIANCE SEAL</div>
                <div>Hash: SHA256: 8f72c914e91845bb02e7039a82f3</div>
              </div>
            </div>
            <div className="text-right">
              <div className="font-bold text-on-surface underline decoration-dotted">Alex Tan, Lead Tech</div>
              <div>Digital Certificate #TW-TECH-9041</div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-3 bg-surface-container-high border-t border-outline-variant/30 flex items-center justify-between">
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg bg-surface-container hover:bg-surface-variant text-on-surface font-headline-sm text-xs font-semibold"
          >
            Close Preview
          </button>
          <button
            type="button"
            onClick={handlePrintOrDownload}
            disabled={downloading}
            className="px-4 py-2 rounded-xl bg-primary-container hover:bg-primary-fixed-dim text-on-primary-container font-headline-sm text-xs font-bold flex items-center gap-2 shadow-lg shadow-primary-container/20 cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">
              {downloading ? 'hourglass_top' : downloaded ? 'check_circle' : 'print'}
            </span>
            <span>
              {downloading
                ? 'Preparing Document...'
                : downloaded
                ? 'Print Dialog Triggered'
                : 'Print / Save Official PDF'}
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};
