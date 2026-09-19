import React, { useState } from 'react';
import { DEPOTS } from '../data/telemetryData';
import { DepotId } from '../types';

interface HeaderProps {
  selectedDepot: DepotId;
  onSelectDepot: (id: DepotId) => void;
  viewMode: 'responsive' | 'mobile' | 'desktop';
  onViewModeChange: (mode: 'responsive' | 'mobile' | 'desktop') => void;
  onOpenTechnicianModal: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  selectedDepot,
  onSelectDepot,
  viewMode,
  onViewModeChange,
  onOpenTechnicianModal,
}) => {
  const currentDepot = DEPOTS.find((d) => d.id === selectedDepot) || DEPOTS[0];
  const [showDepotMenu, setShowDepotMenu] = useState(false);

  return (
    <header className="w-full">
      {/* Top View Mode Switcher Toolbar */}
      <div className="flex items-center justify-between py-1 px-2 mb-2 text-[11px] font-mono text-outline border-b border-outline-variant/20">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 rounded-full bg-secondary-fixed-dim animate-pulse"></span>
          <span className="hidden sm:inline">SYSTEM ONLINE: EN 50126 TELEMETRY ENGINE</span>
          <span className="sm:hidden">SYSTEM ONLINE</span>
        </div>
        
        {/* Device View Mode Switcher */}
        <div className="flex items-center gap-1 bg-surface-container rounded-lg p-0.5 border border-outline-variant/30">
          <button
            type="button"
            onClick={() => onViewModeChange('responsive')}
            title="Fluid responsive layout"
            className={`px-2 py-0.5 rounded text-[10px] font-bold transition-colors ${
              viewMode === 'responsive'
                ? 'bg-primary-container text-on-primary-container'
                : 'text-outline hover:text-on-surface'
            }`}
          >
            Auto
          </button>
          <button
            type="button"
            onClick={() => onViewModeChange('desktop')}
            title="Force desktop wide dashboard (Image 3)"
            className={`px-2 py-0.5 rounded text-[10px] font-bold transition-colors ${
              viewMode === 'desktop'
                ? 'bg-primary-container text-on-primary-container'
                : 'text-outline hover:text-on-surface'
            }`}
          >
            Desktop
          </button>
          <button
            type="button"
            onClick={() => onViewModeChange('mobile')}
            title="Force handheld mobile layout (Image 1)"
            className={`px-2 py-0.5 rounded text-[10px] font-bold transition-colors ${
              viewMode === 'mobile'
                ? 'bg-primary-container text-on-primary-container'
                : 'text-outline hover:text-on-surface'
            }`}
          >
            Mobile
          </button>
        </div>
      </div>

      {/* Main App Bar */}
      <div className="py-2.5 px-3 rounded-xl bg-surface-container-low border border-outline-variant/30 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        {/* Left Section: Branding & Depot / Tech */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Logo Badge */}
          <div className="w-10 h-10 rounded-lg bg-primary-container/15 flex items-center justify-center text-primary-container shrink-0 border border-primary-container/20">
            <span className="material-symbols-outlined text-[24px]">directions_railway</span>
          </div>

          {/* Title & Anomaly Subtext (matches desktop Image 3) */}
          <div className="hidden lg:block border-r border-outline-variant/30 pr-4">
            <div className="flex items-center gap-2">
              <span className="font-headline-sm text-sm font-bold text-on-surface tracking-tight">RailGuard</span>
              <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-primary-container/15 text-primary-container font-semibold">
                Fleet Diagnostics
              </span>
            </div>
            <p className="font-body-sm text-[11px] text-outline mt-0.5">
              Predictive Subsystem AI &amp; Anomaly Detection
            </p>
          </div>

          {/* Depot Selector & Technician */}
          <div className="flex flex-col gap-0.5">
            <div className="flex items-center gap-1.5 relative">
              <span className="material-symbols-outlined text-[15px] text-outline hidden sm:inline">domain</span>
              <span className="font-headline-sm text-xs font-bold text-on-surface">Depot:</span>
              <div className="relative inline-block">
                <button
                  type="button"
                  onClick={() => setShowDepotMenu(!showDepotMenu)}
                  className="bg-surface-container border border-outline-variant/40 hover:border-primary text-primary font-headline-sm text-xs font-bold rounded px-2 py-0.5 pr-6 flex items-center gap-1 transition-colors cursor-pointer"
                >
                  <span>{currentDepot.name}</span>
                  <span className="material-symbols-outlined text-[14px] text-primary absolute right-1.5 top-1/2 -translate-y-1/2 pointer-events-none">
                    expand_more
                  </span>
                </button>

                {/* Dropdown Menu */}
                {showDepotMenu && (
                  <div className="absolute left-0 mt-1 w-52 rounded-xl bg-surface-container-high border border-outline-variant/40 shadow-xl z-50 py-1 overflow-hidden">
                    <div className="px-2.5 py-1 text-[10px] font-mono text-outline uppercase border-b border-outline-variant/30">
                      SMRT Depot Facilities
                    </div>
                    {DEPOTS.map((depot) => (
                      <button
                        key={depot.id}
                        type="button"
                        onClick={() => {
                          onSelectDepot(depot.id);
                          setShowDepotMenu(false);
                        }}
                        className={`w-full text-left px-3 py-1.5 text-xs flex items-center justify-between hover:bg-primary-container/20 transition-colors ${
                          selectedDepot === depot.id ? 'text-primary font-bold bg-primary-container/10' : 'text-on-surface'
                        }`}
                      >
                        <div>
                          <div className="font-headline-sm">{depot.name}</div>
                          <div className="text-[10px] text-outline font-body-sm">{depot.line} • {depot.activeTrains} trains</div>
                        </div>
                        {selectedDepot === depot.id && (
                          <span className="material-symbols-outlined text-[16px] text-primary">check</span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Technician Profile with Clickable Certifications Modal */}
            <button
              type="button"
              onClick={onOpenTechnicianModal}
              className="flex items-center gap-1.5 text-left group hover:opacity-90 transition-opacity"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-secondary-fixed-dim"></span>
              <p className="font-body-sm text-[11px] text-outline group-hover:text-primary transition-colors">
                Alex Tan • Lead Depot Technician
              </p>
              <span className="material-symbols-outlined text-[14px] text-outline group-hover:text-primary">
                expand_more
              </span>
            </button>
          </div>
        </div>

        {/* Right Section (Matches Image 3 Header Status Bar) */}
        <div className="flex items-center gap-3 self-end md:self-auto text-outline font-label-sm text-[11px] bg-surface-container/60 px-3 py-1.5 rounded-lg border border-outline-variant/20">
          <div className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[14px]">calendar_today</span>
            <span>Fri, 24 Oct 2025 • 09:41 AM</span>
          </div>
          <div className="h-3 w-px bg-outline-variant/40 hidden sm:block"></div>
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[16px] text-secondary-fixed-dim" title="5G Depot Mesh Active">
              wifi
            </span>
            <div className="flex items-center gap-0.5 text-on-surface font-semibold" title="Handheld Rugged Tablet Battery">
              <span className="material-symbols-outlined text-[16px] text-secondary-fixed-dim">battery_charging_full</span>
              <span>98%</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
