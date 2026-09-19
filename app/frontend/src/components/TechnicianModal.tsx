import React from 'react';
import { DepotInfo } from '../types';

interface TechnicianModalProps {
  isOpen: boolean;
  onClose: () => void;
  depot: DepotInfo;
}

export const TechnicianModal: React.FC<TechnicianModalProps> = ({
  isOpen,
  onClose,
  depot,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-md bg-surface-container rounded-2xl border-2 border-outline-variant/50 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 bg-surface-container-high border-b border-outline-variant/30 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-10 h-10 rounded-full bg-primary-container/20 text-primary-container flex items-center justify-center border border-primary-container/30 font-headline-md font-bold text-sm">
              AT
            </div>
            <div>
              <h3 className="font-headline-sm text-sm font-bold text-on-surface">
                Alex Tan
              </h3>
              <p className="font-body-sm text-[11px] text-secondary-fixed-dim font-medium">
                Lead Depot Technician • Shift 1 (Day Revenue)
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

        {/* Content */}
        <div className="p-4 flex flex-col gap-3 text-xs">
          <div className="p-3 rounded-xl bg-surface-container-low border border-outline-variant/30 flex flex-col gap-1.5">
            <span className="font-label-sm text-[10px] text-outline uppercase tracking-wider">
              Assigned Facility &amp; Stationing
            </span>
            <div className="font-headline-sm text-sm font-bold text-on-surface">
              {depot.name}
            </div>
            <p className="font-body-sm text-[11px] text-outline">
              Primary maintenance bay for SMRT East-West Line 6-car and 8-car Kawasaki rolling stock.
            </p>
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="font-label-sm text-[10px] text-outline uppercase tracking-wider">
              Active Certifications
            </span>
            <div className="grid grid-cols-2 gap-2">
              <div className="p-2 rounded-lg bg-surface-container border border-outline-variant/20 flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-secondary-fixed-dim">verified</span>
                <div>
                  <div className="font-bold text-[11px]">CENELEC EN 50126</div>
                  <div className="text-[9px] text-outline">RAMS Safety Sign-off</div>
                </div>
              </div>
              <div className="p-2 rounded-lg bg-surface-container border border-outline-variant/20 flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-secondary-fixed-dim">verified</span>
                <div>
                  <div className="font-bold text-[11px]">C151B Doors &amp; Pneumatics</div>
                  <div className="text-[9px] text-outline">Specialist Level III</div>
                </div>
              </div>
              <div className="p-2 rounded-lg bg-surface-container border border-outline-variant/20 flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-secondary-fixed-dim">verified</span>
                <div>
                  <div className="font-bold text-[11px]">HVAC / ACV Systems</div>
                  <div className="text-[9px] text-outline">Refrigerant Level II</div>
                </div>
              </div>
              <div className="p-2 rounded-lg bg-surface-container border border-outline-variant/20 flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-secondary-fixed-dim">verified</span>
                <div>
                  <div className="font-bold text-[11px]">Optical Geometry</div>
                  <div className="text-[9px] text-outline">Rail &amp; Bogie Laser QA</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 bg-surface-container-high border-t border-outline-variant/30 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-primary-container hover:bg-primary-fixed-dim text-on-primary-container font-headline-sm text-xs font-bold"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
