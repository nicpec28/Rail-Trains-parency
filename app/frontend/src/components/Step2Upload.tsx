import React, { useRef, useState } from 'react';
import { TelemetryFile, SubsystemId } from '../types';

interface Step2UploadProps {
  currentFile: TelemetryFile;
  onFileChange: (file: TelemetryFile) => void;
  onResetFile: () => void;
  subsystemId: SubsystemId;
  onRunPrediction: () => void;
  isAnalyzing: boolean;
}

export const Step2Upload: React.FC<Step2UploadProps> = ({
  currentFile,
  onFileChange,
  onResetFile,
  subsystemId,
  onRunPrediction,
  isAnalyzing,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const processFile = (file: File) => {
    const ext =
      file.name.split('.').pop()?.toLowerCase() || 'dat';

    const sizeMb = Number(
      (file.size / (1024 * 1024)).toFixed(1)
    );

    onFileChange({
      name: file.name,

      sizeMb: sizeMb > 0 ? sizeMb : 0.1,

      format: ext,

      fleetId:
        'SMRT East-West Line Fleet #541 (Custom Upload)',

      status: 'active',

      uploadTime: 'Just now',

      // IMPORTANT:
      // Preserve the actual browser File object.
      file,
    });
  };

  const handleNativeFileChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file =
      e.target.files && e.target.files[0];

    if (file) {
      processFile(file);
    }

    // Allow selecting the same file again.
    e.target.value = '';
  };

  const handleDragOver = (
    e: React.DragEvent
  ) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (
    e: React.DragEvent
  ) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (
    e: React.DragEvent
  ) => {
    e.preventDefault();
    setIsDragging(false);

    const file =
      e.dataTransfer.files &&
      e.dataTransfer.files[0];

    if (file) {
      processFile(file);
    }
  };

  const getSubsystemIcon = () => {
    switch (subsystemId) {
      case 'acv':
        return 'mode_fan';

      case 'door':
        return 'sensor_door';

      case 'corrugation':
        return 'waves';

      case 'shm':
        return 'monitor_heart';

      default:
        return 'description';
    }
  };

  return (
    <section className="flex flex-col gap-1.5 shrink-0">

      {/* HEADER */}
      <div className="flex items-center justify-between">

        <div className="flex items-center gap-2">

          <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-surface-container-high text-on-surface-variant font-bold">
            STEP 02
          </span>

          <h2 className="font-headline-sm text-xs sm:text-sm text-on-surface font-semibold">
            Upload Data File
          </h2>

        </div>

        <span className="font-body-sm text-[10px] sm:text-[11px] text-outline">
          Supports .csv, .parquet, .las, .h5 fleet files
        </span>

      </div>

      {/* UPLOAD AREA */}
      <div className="p-2 rounded-xl bg-surface-container-low border border-outline-variant/30">

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.parquet,.las,.h5,.dat,.bin"
          className="hidden"
          onChange={handleNativeFileChange}
        />

        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() =>
            fileInputRef.current?.click()
          }
          className={`flex items-center justify-between gap-3 p-2.5 sm:p-3 rounded-lg border-2 border-dashed transition-all cursor-pointer bg-surface-container-lowest/60 ${
            isDragging
              ? 'border-primary-container bg-primary-container/10'
              : 'border-outline-variant/40 hover:border-primary-container'
          }`}
        >

          {/* FILE INFORMATION */}
          <div className="flex items-center gap-3 min-w-0 flex-1">

            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-lg bg-primary-container/15 text-primary-container flex items-center justify-center shrink-0 border border-primary-container/30">

              <span className="material-symbols-outlined text-[22px] sm:text-[24px]">
                {getSubsystemIcon()}
              </span>

            </div>

            <div className="min-w-0 flex-1">

              <div className="flex items-center gap-2 flex-wrap">

                <span className="font-headline-sm text-xs sm:text-sm font-bold text-on-surface truncate">
                  {currentFile.name}
                </span>

                <span className="font-label-sm text-[9px] sm:text-[10px] px-2 py-0.5 rounded-full bg-secondary-fixed-dim/20 text-secondary-fixed-dim font-bold flex items-center gap-1 border border-secondary-fixed-dim/30">

                  <span className="w-1.5 h-1.5 rounded-full bg-secondary-fixed-dim animate-pulse" />

                  {currentFile.status === 'analyzing'
                    ? 'Analyzing'
                    : currentFile.status === 'error'
                    ? 'Error'
                    : 'Active & Verified'}

                </span>

              </div>

              <p className="font-body-sm text-[10px] sm:text-[11px] text-secondary-fixed-dim/90 font-medium mt-0.5 truncate flex items-center gap-1">

                <span>
                  {currentFile.sizeMb} MB
                </span>

                <span>•</span>

                <span>
                  {currentFile.status === 'analyzing'
                    ? 'Model analysis running'
                    : currentFile.status === 'error'
                    ? 'Analysis failed'
                    : 'Active File Loaded'}
                </span>

                <span className="hidden sm:inline">
                  •
                </span>

                <span className="hidden sm:inline truncate">
                  {currentFile.fleetId}
                </span>

              </p>

            </div>
          </div>

          {/* BUTTONS */}
          <div className="flex items-center gap-2 shrink-0">

            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
              className="px-2.5 sm:px-3 py-1.5 rounded-lg bg-primary-container hover:bg-primary-fixed-dim text-on-primary-container font-headline-sm text-[11px] font-bold flex items-center gap-1.5 shadow-sm transition-all cursor-pointer"
            >

              <span className="material-symbols-outlined text-[15px]">
                swap_horiz
              </span>

              <span className="hidden sm:inline">
                Replace File
              </span>

              <span className="sm:hidden">
                Replace
              </span>

            </button>

            <button
              type="button"
              title="Reset to default fleet file"
              onClick={(e) => {
                e.stopPropagation();
                onResetFile();
              }}
              className="p-1.5 rounded-lg bg-surface-container-high hover:bg-surface-bright text-outline hover:text-error transition-colors border border-outline-variant/40"
            >

              <span className="material-symbols-outlined text-[16px]">
                delete
              </span>

            </button>

          </div>

        </div>

        {/* RUN MODEL BUTTON */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRunPrediction();
          }}
          disabled={!currentFile.file || isAnalyzing}
          className="w-full mt-2.5 py-2.5 px-4 rounded-xl bg-primary-container hover:bg-primary-fixed-dim disabled:opacity-50 disabled:cursor-not-allowed text-on-primary-container font-headline-sm text-xs font-bold flex items-center justify-center gap-2 shadow-md transition-all"
        >

          <span
            className={`material-symbols-outlined text-[18px] ${
              isAnalyzing ? 'animate-spin' : ''
            }`}
          >
            {isAnalyzing
              ? 'sync'
              : 'analytics'}
          </span>

          <span>
            {isAnalyzing
              ? 'Running Diagnostic Model...'
              : 'Run Diagnostic Model'}
          </span>

        </button>

      </div>

    </section>
  );
};