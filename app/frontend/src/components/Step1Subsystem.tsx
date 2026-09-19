import React from 'react';
import { SubsystemId } from '../types';
import { SUBSYSTEM_DATA } from '../data/telemetryData';

interface Step1SubsystemProps {
  selectedSubsystem: SubsystemId;
  onSelectSubsystem: (id: SubsystemId) => void;
  isMobileLayout: boolean;
}

export const Step1Subsystem: React.FC<Step1SubsystemProps> = ({
  selectedSubsystem,
  onSelectSubsystem,
  isMobileLayout,
}) => {
  const subsystems: Array<{ id: SubsystemId; label: string; icon: string }> = [
    { id: 'acv', label: 'ACV', icon: 'mode_fan' },
    { id: 'door', label: 'Door', icon: 'sensor_door' },
    { id: 'corrugation', label: 'Rail Corrugation', icon: 'waves' },
    { id: 'shm', label: 'SHM', icon: 'monitor_heart' },
  ];

  const currentDetail = SUBSYSTEM_DATA[selectedSubsystem];

  return (
    <section className="flex flex-col gap-2 shrink-0">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-primary-container/20 text-primary-container font-bold tracking-wider">
            STEP 01
          </span>
          <h2 className="font-headline-sm text-xs sm:text-sm text-on-surface font-semibold tracking-wide">
            Select Subsystem
          </h2>
        </div>

        {isMobileLayout ? (
          <span className="font-label-sm text-[10px] text-primary-container font-bold px-2 py-0.5 rounded bg-primary-container/20 border border-primary-container/30">
            {currentDetail.activePillText}
          </span>
        ) : (
          <span className="text-[11px] text-outline font-body-sm hidden md:inline">
            Click any subsystem to view instant telemetry predictions
          </span>
        )}
      </div>

      {/* Grid: 2x2 on mobile, 4 columns on desktop */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {subsystems.map((sub) => {
          const isActive = selectedSubsystem === sub.id;
          return (
            <button
              key={sub.id}
              type="button"
              onClick={() => onSelectSubsystem(sub.id)}
              className={`h-12 px-3 rounded-xl transition-all cursor-pointer flex items-center justify-between ${
                isActive
                  ? 'bg-gradient-to-r from-primary-container/20 to-surface-container-high border-2 border-primary-container shadow-md shadow-primary-container/10 ring-2 ring-primary-container/30 text-primary font-bold'
                  : 'bg-surface-container-low hover:bg-surface-container rounded-xl border border-outline-variant/30 text-on-surface'
              }`}
            >
              <div className="flex items-center gap-2 truncate">
                <span
                  className={`material-symbols-outlined text-[20px] ${
                    isActive ? 'text-primary font-bold' : 'text-outline'
                  }`}
                >
                  {sub.icon}
                </span>
                <span
                  className={`font-headline-md text-xs truncate ${
                    isActive ? 'font-bold text-primary' : 'font-semibold'
                  }`}
                >
                  {sub.label}
                </span>
              </div>
              {isActive && (
                <span
                  className="material-symbols-outlined text-[17px] text-primary font-bold shrink-0 ml-1"
                  style={{ fontVariationSettings: '"FILL" 1' }}
                >
                  check_circle
                </span>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
};
