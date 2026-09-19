import React from 'react';
import {
  PredictionResult,
  SubsystemId,
} from '../types';

interface Step3DiagnosticsProps {
  subsystemId: SubsystemId;
  onInspectRail: () => void;
  isDoorRepaired: boolean;
  predictionResult?: PredictionResult;
  isAnalyzing?: boolean;
}

export const Step3Diagnostics: React.FC<
  Step3DiagnosticsProps
> = ({
  subsystemId,
  onInspectRail,
  isDoorRepaired,
  predictionResult,
  isAnalyzing = false,
}) => {

  const hasResult =
    predictionResult?.status === 'success';

  const hasError =
    predictionResult?.status === 'error';

  const prediction =
    predictionResult?.prediction;

  const confidence =
    predictionResult?.confidence !== undefined
      ? `${(
          predictionResult.confidence * 100
        ).toFixed(1)}%`
      : null;

  const getSubsystemName = () => {
    switch (subsystemId) {
      case 'door':
        return 'Door';

      case 'acv':
        return 'ACV';

      case 'corrugation':
        return 'Rail Corrugation';

      case 'shm':
        return 'Structural Health Monitor';

      default:
        return 'Subsystem';
    }
  };

  const getIcon = () => {
    switch (subsystemId) {
      case 'door':
        return 'sensor_door';

      case 'acv':
        return 'mode_fan';

      case 'corrugation':
        return 'waves';

      case 'shm':
        return 'monitor_heart';

      default:
        return 'analytics';
    }
  };

  return (
    <section className="flex flex-col gap-2.5">

      {/* =====================================================
          HEADER
      ===================================================== */}

      <div className="flex items-center justify-between shrink-0">

        <div className="flex items-center gap-2">

          <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-surface-container-high text-on-surface-variant font-bold">
            STEP 03
          </span>

          <h2 className="font-headline-sm text-xs sm:text-sm text-on-surface font-semibold">
            Subsystem Diagnostics
          </h2>

        </div>

        <div
          className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-label-md text-[10px] sm:text-[11px] font-bold border ${
            isAnalyzing
              ? 'bg-primary-container/15 text-primary-container border-primary-container/30'
              : hasResult
              ? 'bg-secondary-fixed-dim/15 text-secondary-fixed-dim border-secondary-fixed-dim/30'
              : hasError
              ? 'bg-error/15 text-error border-error/30'
              : 'bg-surface-container-high text-outline border-outline-variant/30'
          }`}
        >

          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isAnalyzing
                ? 'bg-primary-container animate-ping'
                : hasResult
                ? 'bg-secondary-fixed-dim'
                : hasError
                ? 'bg-error'
                : 'bg-outline'
            }`}
          />

          <span>
            {isAnalyzing
              ? 'Analysis Running'
              : hasResult
              ? 'Analysis Complete'
              : hasError
              ? 'Analysis Error'
              : 'Awaiting Analysis'}
          </span>

        </div>

      </div>

      {/* =====================================================
          MAIN MODEL RESULT
      ===================================================== */}

      <div
        className={`p-4 rounded-xl border-2 shadow-sm ${
          isAnalyzing
            ? 'bg-primary-container/5 border-primary-container/30'
            : hasResult
            ? 'bg-secondary-fixed-dim/5 border-secondary-fixed-dim/30'
            : hasError
            ? 'bg-error/5 border-error/30'
            : 'bg-surface-container border-outline-variant/40'
        }`}
      >

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">

          <div className="flex items-center gap-3">

            <div
              className={`w-11 h-11 rounded-xl flex items-center justify-center border ${
                isAnalyzing
                  ? 'bg-primary-container/15 text-primary-container border-primary-container/30'
                  : hasResult
                  ? 'bg-secondary-fixed-dim/15 text-secondary-fixed-dim border-secondary-fixed-dim/30'
                  : hasError
                  ? 'bg-error/15 text-error border-error/30'
                  : 'bg-surface-container-high text-outline border-outline-variant/40'
              }`}
            >

              <span
                className={`material-symbols-outlined text-[24px] ${
                  isAnalyzing
                    ? 'animate-spin'
                    : ''
                }`}
              >
                {isAnalyzing
                  ? 'sync'
                  : hasResult
                  ? 'verified'
                  : hasError
                  ? 'error'
                  : getIcon()}
              </span>

            </div>

            <div>

              <div className="font-label-sm text-[10px] text-outline uppercase tracking-wider font-bold">
                {getSubsystemName()} • ML Diagnostic
              </div>

              <div
                className={`font-headline-md text-base sm:text-lg font-black tracking-tight ${
                  isAnalyzing
                    ? 'text-primary-container'
                    : hasResult
                    ? 'text-secondary-fixed-dim'
                    : hasError
                    ? 'text-error'
                    : 'text-on-surface'
                }`}
              >

                {isAnalyzing
                  ? 'ANALYZING TELEMETRY'
                  : hasResult
                  ? `PREDICTION: ${String(
                      prediction
                    )}`
                  : hasError
                  ? 'MODEL EXECUTION FAILED'
                  : 'AWAITING MODEL ANALYSIS'}

              </div>

              <p className="font-body-sm text-[11px] text-outline mt-0.5">

                {isAnalyzing
                  ? 'The uploaded telemetry file is being processed by the diagnostic model.'
                  : hasResult
                  ? 'Prediction returned directly from the deployed machine-learning model.'
                  : hasError
                  ? predictionResult?.message ||
                    'The backend returned an error while running the model.'
                  : 'Upload a telemetry file and run the diagnostic model.'}

              </p>

            </div>

          </div>

          {confidence && (
            <div className="flex flex-col items-start sm:items-end">

              <span className="font-label-sm text-[9px] text-outline uppercase font-bold">
                Model Confidence
              </span>

              <span className="font-headline-md text-xl font-black text-secondary-fixed-dim">
                {confidence}
              </span>

            </div>
          )}

        </div>

        {/* MODEL DETAILS */}

        {hasResult && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-4 pt-3 border-t border-outline-variant/30">

            <div className="p-2.5 rounded-lg bg-surface-container">

              <div className="font-label-sm text-[9px] text-outline uppercase font-bold">
                Model
              </div>

              <div className="font-headline-sm text-xs font-bold text-on-surface mt-1 truncate">
                {predictionResult?.modelName || '—'}
              </div>

            </div>

            <div className="p-2.5 rounded-lg bg-surface-container">

              <div className="font-label-sm text-[9px] text-outline uppercase font-bold">
                Prediction
              </div>

              <div className="font-headline-sm text-xs font-bold text-on-surface mt-1">
                {String(prediction)}
              </div>

            </div>

            <div className="p-2.5 rounded-lg bg-surface-container">

              <div className="font-label-sm text-[9px] text-outline uppercase font-bold">
                Rows Analysed
              </div>

              <div className="font-headline-sm text-xs font-bold text-on-surface mt-1">
                {predictionResult?.rowCount ?? '—'}
              </div>

            </div>

            <div className="p-2.5 rounded-lg bg-surface-container">

              <div className="font-label-sm text-[9px] text-outline uppercase font-bold">
                Features
              </div>

              <div className="font-headline-sm text-xs font-bold text-on-surface mt-1">
                {predictionResult?.featureCount ?? '—'}
              </div>

            </div>

          </div>
        )}

      </div>

      {/* =====================================================
          DOOR
      ===================================================== */}

      {subsystemId === 'door' && (
        <div className="flex flex-col gap-3">

          {/* DOOR STATUS */}

          <div
            className={`p-3.5 rounded-xl border-2 shadow-md flex items-start justify-between gap-3 ${
              isDoorRepaired
                ? 'border-secondary-fixed-dim bg-secondary-fixed-dim/15'
                : hasResult
                ? 'border-primary-container/40 bg-primary-container/5'
                : 'border-outline-variant/40 bg-surface-container'
            }`}
          >

            <div className="flex items-center gap-3 min-w-0">

              <div
                className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 border ${
                  isDoorRepaired
                    ? 'bg-secondary-fixed-dim/30 text-secondary-fixed-dim border-secondary-fixed-dim/40'
                    : hasResult
                    ? 'bg-primary-container/15 text-primary-container border-primary-container/30'
                    : 'bg-surface-container-high text-outline border-outline-variant/40'
                }`}
              >

                <span className="material-symbols-outlined text-[24px]">
                  {isDoorRepaired
                    ? 'verified'
                    : hasResult
                    ? 'analytics'
                    : 'pending'}
                </span>

              </div>

              <div className="min-w-0">

                <div
                  className={`font-headline-md text-sm font-extrabold tracking-tight ${
                    isDoorRepaired
                      ? 'text-secondary-fixed-dim'
                      : hasResult
                      ? 'text-primary-container'
                      : 'text-on-surface'
                  }`}
                >
                  {isDoorRepaired
                    ? 'Door Subsystem Nominal & Recalibrated'
                    : hasResult
                    ? `Model Prediction: ${String(
                        prediction
                      )}`
                    : 'Awaiting Door Analysis'}
                </div>

                <span className="font-body-sm text-[11px] text-on-surface font-medium block leading-tight mt-0.5">
                  {isDoorRepaired
                    ? 'Obstruction cleared and optical 5-cycle stroke test verified.'
                    : hasResult
                    ? 'Prediction generated from the uploaded Door telemetry file.'
                    : 'Upload Door telemetry and run the Door diagnostic model.'}
                </span>

              </div>

            </div>

            <span
              className={`font-label-sm text-[10px] font-black px-2.5 py-1 rounded-full uppercase shrink-0 shadow-sm ${
                isDoorRepaired
                  ? 'bg-secondary-fixed-dim text-on-secondary'
                  : hasResult
                  ? 'bg-primary-container text-on-primary-container'
                  : 'bg-surface-container-high text-outline'
              }`}
            >
              {isDoorRepaired
                ? 'VERIFIED READY'
                : hasResult
                ? 'MODEL RESULT'
                : 'AWAITING'}
            </span>

          </div>

          {/* DOOR MODEL METRICS */}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">

            <div className="p-3 rounded-xl bg-surface-container border-2 border-outline-variant/40 flex flex-col justify-between gap-2 shadow-sm">

              <div className="flex items-center justify-between">

                <span className="font-label-sm text-[10px] text-outline font-bold uppercase tracking-wider">
                  MODEL STATUS
                </span>

                <span className="material-symbols-outlined text-[18px] text-primary">
                  memory
                </span>

              </div>

              <div>

                <div className="font-headline-md text-base sm:text-lg font-black text-on-surface uppercase tracking-tight">
                  {hasResult
                    ? 'MODEL EXECUTED'
                    : isAnalyzing
                    ? 'RUNNING'
                    : 'NOT RUN'}
                </div>

                <p className="font-body-sm text-[11px] text-on-surface font-medium leading-relaxed mt-1">
                  {hasResult
                    ? 'The Door model successfully processed the uploaded telemetry.'
                    : 'No completed model prediction is available yet.'}
                </p>

              </div>

            </div>

            <div className="p-3 rounded-xl bg-surface-container border-2 border-outline-variant/40 flex flex-col justify-between gap-2 shadow-sm">

              <div className="flex items-center justify-between">

                <span className="font-label-sm text-[10px] text-outline font-bold uppercase tracking-wider">
                  PREDICTION
                </span>

                <span className="material-symbols-outlined text-[18px] text-primary">
                  analytics
                </span>

              </div>

              <div>

                <div className="font-headline-md text-base sm:text-lg font-black text-on-surface uppercase tracking-tight">
                  {hasResult
                    ? String(prediction)
                    : '—'}
                </div>

                <p className="font-body-sm text-[11px] text-on-surface font-medium leading-relaxed mt-1">
                  Raw prediction returned by the Door model.
                </p>

              </div>

            </div>

            <div className="p-3 rounded-xl bg-surface-container border-2 border-outline-variant/40 flex flex-col justify-between gap-2 shadow-sm">

              <div className="flex items-center justify-between">

                <span className="font-label-sm text-[10px] text-outline font-bold uppercase tracking-wider">
                  CONFIDENCE
                </span>

                <span className="material-symbols-outlined text-[18px] text-primary">
                  verified
                </span>

              </div>

              <div>

                <div className="font-headline-md text-base sm:text-lg font-black text-on-surface uppercase tracking-tight">
                  {confidence ?? '—'}
                </div>

                <p className="font-body-sm text-[11px] text-on-surface font-medium leading-relaxed mt-1">
                  Confidence supplied by the backend model.
                </p>

              </div>

            </div>

          </div>

          {/* ACTION */}

          <div className="p-2.5 sm:p-3 rounded-xl bg-surface-container-high border-2 border-outline-variant/40 flex items-center justify-between gap-3 shadow-sm">

            <div className="flex items-center gap-2.5 min-w-0 flex-1">

              <div className="w-8 h-8 rounded-lg bg-primary-container/20 flex items-center justify-center text-primary shrink-0 border border-primary-container/30">

                <span className="material-symbols-outlined text-[18px]">
                  build
                </span>

              </div>

              <div className="min-w-0">

                <div className="font-headline-sm text-xs sm:text-[13px] font-bold text-on-surface leading-tight">
                  What to do right now:
                </div>

                <p className="font-body-sm text-[10px] sm:text-[11px] text-outline truncate leading-tight mt-0.5">
                  {hasResult
                    ? 'Review the model prediction and perform the recommended inspection procedure.'
                    : 'Upload telemetry and run the Door diagnostic model.'}
                </p>

              </div>

            </div>

            <button
              type="button"
              onClick={onInspectRail}
              className="px-3 py-1.5 rounded-lg bg-primary-container hover:bg-primary-fixed-dim text-on-primary-container font-headline-sm text-[11px] sm:text-xs font-bold shrink-0 flex items-center gap-1.5 shadow-sm transition-all cursor-pointer"
            >

              <span className="material-symbols-outlined text-[16px]">
                check
              </span>

              <span>
                Inspect Rail Now
              </span>

            </button>

          </div>

        </div>
      )}

      {/* =====================================================
          ACV
      ===================================================== */}

      {subsystemId === 'acv' && (
        <div className="p-4 rounded-xl bg-surface-container border-2 border-outline-variant/40">

          <div className="flex items-center gap-2 mb-4">

            <span className="material-symbols-outlined text-primary">
              mode_fan
            </span>

            <span className="font-headline-sm text-sm font-bold text-on-surface">
              ACV Diagnostic Result
            </span>

          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">

            <DiagnosticValue
              label="MODEL STATUS"
              value={
                hasResult
                  ? 'MODEL EXECUTED'
                  : isAnalyzing
                  ? 'RUNNING'
                  : 'NOT RUN'
              }
            />

            <DiagnosticValue
              label="PREDICTION"
              value={
                hasResult
                  ? String(prediction)
                  : '—'
              }
            />

            <DiagnosticValue
              label="CONFIDENCE"
              value={confidence ?? '—'}
            />

          </div>

        </div>
      )}

      {/* =====================================================
          RAIL
      ===================================================== */}

      {subsystemId === 'corrugation' && (
        <div className="p-4 rounded-xl bg-surface-container border-2 border-outline-variant/40">

          <div className="flex items-center gap-2 mb-4">

            <span className="material-symbols-outlined text-primary">
              waves
            </span>

            <span className="font-headline-sm text-sm font-bold">
              Rail Corrugation Diagnostic Result
            </span>

          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">

            <DiagnosticValue
              label="MODEL"
              value={
                hasResult
                  ? predictionResult?.modelName || '—'
                  : 'Awaiting Analysis'
              }
            />

            <DiagnosticValue
              label="PREDICTION"
              value={
                hasResult
                  ? String(prediction)
                  : '—'
              }
            />

            <DiagnosticValue
              label="CONFIDENCE"
              value={confidence ?? '—'}
            />

          </div>

        </div>
      )}

      {/* =====================================================
          SHM
      ===================================================== */}

      {subsystemId === 'shm' && (
        <div className="p-4 rounded-xl bg-surface-container border-2 border-outline-variant/40">

          <div className="flex items-center justify-between mb-4">

            <div className="flex items-center gap-2">

              <span className="material-symbols-outlined text-primary">
                monitor_heart
              </span>

              <h3 className="font-headline-sm text-sm font-bold uppercase">
                Structural Health Monitor
              </h3>

            </div>

            <span
              className={`px-2.5 py-0.5 rounded-full border font-headline-sm text-[10px] font-bold ${
                hasResult
                  ? 'bg-secondary-fixed-dim/20 border-secondary-fixed-dim text-secondary-fixed-dim'
                  : 'bg-surface-container-high border-outline-variant text-outline'
              }`}
            >
              {hasResult
                ? 'MODEL COMPLETE'
                : isAnalyzing
                ? 'ANALYZING'
                : 'AWAITING'}
            </span>

          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">

            <DiagnosticValue
              label="MODEL"
              value={
                hasResult
                  ? predictionResult?.modelName || '—'
                  : 'Awaiting Analysis'
              }
            />

            <DiagnosticValue
              label="PREDICTION"
              value={
                hasResult
                  ? String(prediction)
                  : '—'
              }
            />

            <DiagnosticValue
              label="CONFIDENCE"
              value={confidence ?? '—'}
            />

          </div>

        </div>
      )}

    </section>
  );
};


/* =========================================================
   SMALL REUSABLE DIAGNOSTIC VALUE CARD
   ========================================================= */

interface DiagnosticValueProps {
  label: string;
  value: string;
}

const DiagnosticValue: React.FC<
  DiagnosticValueProps
> = ({
  label,
  value,
}) => {
  return (
    <div className="p-3.5 rounded-lg bg-surface-container-lowest border border-outline-variant/30">

      <div className="font-label-sm text-[9px] text-outline uppercase font-bold">
        {label}
      </div>

      <div className="font-headline-md text-lg font-black mt-1 break-words">
        {value}
      </div>

    </div>
  );
};