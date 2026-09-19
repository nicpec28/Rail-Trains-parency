import { useState } from 'react';

import {
  SubsystemId,
  DepotId,
  TelemetryFile,
  PredictionResult,
} from './types';

import { DEPOTS, SUBSYSTEM_DATA } from './data/telemetryData';

import { Header } from './components/Header';
import { Step1Subsystem } from './components/Step1Subsystem';
import { Step2Upload } from './components/Step2Upload';
import { Step3Diagnostics } from './components/Step3Diagnostics';
import { InspectModal } from './components/InspectModal';
import { ReportModal } from './components/ReportModal';
import { TechnicianModal } from './components/TechnicianModal';


/* ============================================================
   BACKEND CONFIGURATION
   ============================================================ */

const API_URL =
  import.meta.env.VITE_API_URL ||
  'http://localhost:8000';


/*
 * Maps the dashboard subsystem to the corresponding
 * FastAPI endpoint.
 *
 * Door:
 *   /door/predict
 *
 * ACV:
 *   /acv/predict
 *
 * SHM:
 *   /shm/predict
 *
 * Rail Corrugation:
 *   /rail/predict
 */

const MODEL_ENDPOINTS: Record<
  SubsystemId,
  string
> = {
  door: '/door/predict',
  acv: '/acv/predict',
  shm: '/shm/predict',
  corrugation: '/rail/predict',
};


export default function App() {

  /* ==========================================================
     EXISTING DASHBOARD STATE
     ========================================================== */

  const [
    selectedDepot,
    setSelectedDepot,
  ] = useState<DepotId>('tuas-west');


  const [
    selectedSubsystem,
    setSelectedSubsystem,
  ] = useState<SubsystemId>('door');


  const [
    viewMode,
    setViewMode,
  ] = useState<
    'responsive' | 'mobile' | 'desktop'
  >('responsive');


  const [
    files,
    setFiles,
  ] = useState<
    Record<SubsystemId, TelemetryFile>
  >({
    door: {
      ...SUBSYSTEM_DATA.door.defaultFile,
    },

    acv: {
      ...SUBSYSTEM_DATA.acv.defaultFile,
    },

    corrugation: {
      ...SUBSYSTEM_DATA.corrugation.defaultFile,
    },

    shm: {
      ...SUBSYSTEM_DATA.shm.defaultFile,
    },
  });


  /* ==========================================================
     DIAGNOSTIC STATE
     ========================================================== */

  const [
    predictionResults,
    setPredictionResults,
  ] = useState<
    Partial<
      Record<
        SubsystemId,
        PredictionResult
      >
    >
  >({});


  const [
    isAnalyzing,
    setIsAnalyzing,
  ] = useState(false);


  /* ==========================================================
     EXISTING MODAL STATE
     ========================================================== */

  const [
    isDoorRepaired,
    setIsDoorRepaired,
  ] = useState(false);


  const [
    isInspectModalOpen,
    setIsInspectModalOpen,
  ] = useState(false);


  const [
    isReportModalOpen,
    setIsReportModalOpen,
  ] = useState(false);


  const [
    isTechnicianModalOpen,
    setIsTechnicianModalOpen,
  ] = useState(false);


  const [
    downloadingReport,
    setDownloadingReport,
  ] = useState(false);


  /* ==========================================================
     CURRENT DASHBOARD DATA
     ========================================================== */

  const currentDepot =
    DEPOTS.find(
      (d) => d.id === selectedDepot
    ) || DEPOTS[0];


  const currentSubsystemDetail =
    SUBSYSTEM_DATA[selectedSubsystem];


  const currentFile =
    files[selectedSubsystem];


  const currentPrediction =
    predictionResults[selectedSubsystem];


  /* ==========================================================
     FILE CHANGE
     ========================================================== */

  const handleFileChange = (
    newFile: TelemetryFile
  ) => {

    setFiles((prev) => ({
      ...prev,

      [selectedSubsystem]: newFile,
    }));


    /*
     * Clear the previous model result when
     * a new file is uploaded.
     */

    setPredictionResults((prev) => {

      const updated = {
        ...prev,
      };

      delete updated[selectedSubsystem];

      return updated;
    });


    /*
     * If Door gets a new file, reset the
     * repair state because the new file
     * needs to be analysed independently.
     */

    if (selectedSubsystem === 'door') {
      setIsDoorRepaired(false);
    }
  };


  /* ==========================================================
     RESET FILE
     ========================================================== */

  const handleResetFile = () => {

    setFiles((prev) => ({
      ...prev,

      [selectedSubsystem]: {
        ...SUBSYSTEM_DATA[
          selectedSubsystem
        ].defaultFile,
      },
    }));


    setPredictionResults((prev) => {

      const updated = {
        ...prev,
      };

      delete updated[selectedSubsystem];

      return updated;
    });


    if (selectedSubsystem === 'door') {
      setIsDoorRepaired(false);
    }
  };


  /* ==========================================================
     RUN ML MODEL
     ========================================================== */

  const runPrediction = async () => {

    /*
     * IMPORTANT:
     *
     * The actual File object is stored inside
     * currentFile.file by Step2Upload.
     */

    const uploadedFile =
      currentFile.file;


    if (!uploadedFile) {

      alert(
        'Please upload a telemetry file first.'
      );

      return;
    }


    setIsAnalyzing(true);


    /*
     * Update Step 2 status.
     */

    setFiles((prev) => ({
      ...prev,

      [selectedSubsystem]: {
        ...prev[selectedSubsystem],

        status: 'analyzing',
      },
    }));


    /*
     * Clear previous result while the new
     * prediction is running.
     */

    setPredictionResults((prev) => {

      const updated = {
        ...prev,
      };

      delete updated[selectedSubsystem];

      return updated;
    });


    try {

      const formData =
        new FormData();


      /*
       * FastAPI expects:
       *
       * file=<uploaded CSV>
       */

      formData.append(
        'file',
        uploadedFile
      );


      const endpoint =
        MODEL_ENDPOINTS[
          selectedSubsystem
        ];


      console.log(
        'Running model:',
        selectedSubsystem
      );

      console.log(
        'Endpoint:',
        `${API_URL}${endpoint}`
      );

      console.log(
        'File:',
        uploadedFile.name
      );


      const response =
        await fetch(
          `${API_URL}${endpoint}`,
          {
            method: 'POST',

            body: formData,
          }
        );


      /*
       * Handle HTTP errors.
       */

      if (!response.ok) {

        let errorMessage =
          `Prediction failed with HTTP ${response.status}.`;


        try {

          const errorData =
            await response.json();


          if (
            errorData?.detail
          ) {
            errorMessage =
              typeof errorData.detail ===
              'string'
                ? errorData.detail
                : JSON.stringify(
                    errorData.detail
                  );
          }

        } catch {
          /*
           * Response wasn't JSON.
           */
        }


        throw new Error(
          errorMessage
        );
      }


      /*
       * Read model response.
       */

      const result =
        await response.json();


      console.log(
        'Model response:',
        result
      );


      /*
       * Store result for the currently
       * selected subsystem.
       */

      const predictionResult:
        PredictionResult = {

        ...result,

        subsystemId:
          selectedSubsystem,

        status: 'success',

        filename:
          result.filename ||
          uploadedFile.name,
      };


      setPredictionResults(
        (prev) => ({
          ...prev,

          [selectedSubsystem]:
            predictionResult,
        })
      );


      /*
       * Mark file as active again.
       */

      setFiles((prev) => ({
        ...prev,

        [selectedSubsystem]: {
          ...prev[selectedSubsystem],

          status: 'active',

          uploadTime: 'Just now',
        },
      }));


    } catch (error) {

      console.error(
        'Prediction error:',
        error
      );


      const message =
        error instanceof Error
          ? error.message
          : 'Unknown prediction error';


      /*
       * Store error in Step 3.
       */

      setPredictionResults(
        (prev) => ({
          ...prev,

          [selectedSubsystem]: {

            subsystemId:
              selectedSubsystem,

            modelName:
              selectedSubsystem,

            prediction:
              'ERROR',

            filename:
              uploadedFile.name,

            status:
              'error',

            message,
          },
        })
      );


      /*
       * Mark file as error.
       */

      setFiles((prev) => ({
        ...prev,

        [selectedSubsystem]: {
          ...prev[selectedSubsystem],

          status: 'error',
        },
      }));


    } finally {

      setIsAnalyzing(false);
    }
  };


  /* ==========================================================
     REPORT DOWNLOAD
     ========================================================== */

  const triggerDownloadAction = () => {

    setDownloadingReport(true);


    setTimeout(() => {

      setDownloadingReport(false);

      setIsReportModalOpen(true);

    }, 600);
  };


  /* ==========================================================
     VIEW MODE
     ========================================================== */

  const isStrictMobile =
    viewMode === 'mobile';


  const isStrictDesktop =
    viewMode === 'desktop';


  /* ==========================================================
     RENDER
     ========================================================== */

  return (

    <div className="min-h-screen bg-background text-on-surface font-body-md selection:bg-primary-container selection:text-on-primary-container">

      <div
        className={`min-h-screen flex flex-col justify-between mx-auto p-3 sm:p-5 box-border transition-all ${
          isStrictMobile
            ? 'max-w-[420px] pb-28'
            : isStrictDesktop
            ? 'max-w-[1240px] pb-10'
            : 'max-w-[420px] md:max-w-[1240px] pb-28 md:pb-10'
        }`}
      >

        <main className="w-full flex flex-col gap-4">

          {/* ==================================================
              HEADER
              ================================================== */}

          <Header
            selectedDepot={
              selectedDepot
            }

            onSelectDepot={
              setSelectedDepot
            }

            viewMode={
              viewMode
            }

            onViewModeChange={
              setViewMode
            }

            onOpenTechnicianModal={() =>
              setIsTechnicianModalOpen(
                true
              )
            }
          />


          {/* ==================================================
              STEP 1
              ================================================== */}

          <Step1Subsystem
            selectedSubsystem={
              selectedSubsystem
            }

            onSelectSubsystem={
              setSelectedSubsystem
            }

            isMobileLayout={
              isStrictMobile
            }
          />


          {/* ==================================================
              STEP 2
              ================================================== */}

          <Step2Upload
            currentFile={
              currentFile
            }

            onFileChange={
              handleFileChange
            }

            onResetFile={
              handleResetFile
            }

            subsystemId={
              selectedSubsystem
            }

            onRunPrediction={
              runPrediction
            }

            isAnalyzing={
              isAnalyzing
            }
          />


          {/* ==================================================
              STEP 3
              ================================================== */}

          <Step3Diagnostics
            subsystemId={
              selectedSubsystem
            }

            onInspectRail={() =>
              setIsInspectModalOpen(
                true
              )
            }

            isDoorRepaired={
              isDoorRepaired
            }

            predictionResult={
              currentPrediction
            }

            isAnalyzing={
              isAnalyzing
            }
          />


          {/* ==================================================
              DESKTOP FOOTER
              ================================================== */}

          {!isStrictMobile && (
            <footer className="hidden md:flex items-center justify-between py-4 border-t border-outline-variant/30 mt-2 shrink-0">

              <div className="flex items-center gap-2 text-outline font-body-sm text-[11px]">

                <span className="material-symbols-outlined text-[16px] text-secondary-fixed-dim">
                  verified
                </span>

                <span>
                  Compliant with CENELEC EN 50126
                </span>

              </div>


              <button
                type="button"
                onClick={
                  triggerDownloadAction
                }
                className="py-2.5 px-5 rounded-xl bg-primary-container hover:bg-primary-fixed-dim active:scale-[0.99] text-on-primary-container font-headline-sm text-xs font-bold flex items-center gap-2 transition-all shadow-md shadow-primary-container/20 cursor-pointer"
              >

                <span className="material-symbols-outlined text-[18px]">
                  {downloadingReport
                    ? 'sync'
                    : 'download'}
                </span>

                <span>
                  {downloadingReport
                    ? 'Preparing PDF...'
                    : 'Download Prediction & Diagnostic Report (PDF)'}
                </span>

              </button>

            </footer>
          )}

        </main>

      </div>


      {/* ======================================================
          MOBILE DOWNLOAD BAR
          ====================================================== */}

      {(isStrictMobile ||
        viewMode === 'responsive') && (

        <div
          className={`fixed bottom-0 left-0 right-0 max-w-[420px] mx-auto bg-surface/95 backdrop-blur-md p-3 border-t border-outline-variant/40 shadow-2xl z-40 flex flex-col gap-1.5 box-border ${
            viewMode === 'responsive'
              ? 'md:hidden'
              : ''
          }`}
        >

          <button
            type="button"
            onClick={
              triggerDownloadAction
            }
            className="w-full py-3 px-4 rounded-xl bg-primary-container hover:bg-primary-fixed-dim active:scale-[0.99] text-on-primary-container font-headline-sm text-xs font-bold flex items-center justify-center gap-2 transition-all shadow-lg shadow-primary-container/25 cursor-pointer"
          >

            <span className="material-symbols-outlined text-[18px]">

              {downloadingReport
                ? 'sync'
                : 'download'}

            </span>

            <span>
              {downloadingReport
                ? 'Generating PDF...'
                : currentSubsystemDetail.reportTitle}
            </span>

          </button>


          <div className="flex items-center justify-center gap-1.5 text-outline font-body-sm text-[10px]">

            <span className="material-symbols-outlined text-[14px] text-secondary-fixed-dim">
              verified
            </span>

            <span>
              Verified for safety standard compliance (CENELEC EN 50126)
            </span>

          </div>

        </div>
      )}


      {/* ======================================================
          INSPECT MODAL
          ====================================================== */}

      <InspectModal
        isOpen={
          isInspectModalOpen
        }

        onClose={() =>
          setIsInspectModalOpen(
            false
          )
        }

        onCompleteRepair={() =>
          setIsDoorRepaired(
            true
          )
        }

        isRepaired={
          isDoorRepaired
        }
      />


      {/* ======================================================
          REPORT MODAL
          ====================================================== */}

      <ReportModal
        isOpen={
          isReportModalOpen
        }

        onClose={() =>
          setIsReportModalOpen(
            false
          )
        }

        subsystem={
          currentSubsystemDetail
        }

        depot={
          currentDepot
        }

        isRepaired={
          isDoorRepaired
        }
      />


      {/* ======================================================
          TECHNICIAN MODAL
          ====================================================== */}

      <TechnicianModal
        isOpen={
          isTechnicianModalOpen
        }

        onClose={() =>
          setIsTechnicianModalOpen(
            false
          )
        }

        depot={
          currentDepot
        }

      />

    </div>
  );
}