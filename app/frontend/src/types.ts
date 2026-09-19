export type SubsystemId = 'door' | 'acv' | 'corrugation' | 'shm';

export type DepotId =
  | 'tuas-west'
  | 'bishan'
  | 'ulu-pandan'
  | 'changi'
  | 'kim-chuan';

export interface DepotInfo {
  id: DepotId;
  name: string;
  line: string;
  activeTrains: number;
}

export interface TelemetryFile {
  name: string;
  sizeMb: number;
  format: string;
  fleetId: string;
  status: 'active' | 'analyzing' | 'error';
  uploadTime: string;

  // The actual browser File object.
  // This is what gets sent to FastAPI.
  file?: File;
}

export interface PredictionResult {
  subsystemId: SubsystemId;

  modelName: string;

  prediction: number | string;

  confidence?: number;

  probabilities?: Record<string, number>;

  filename: string;

  rowCount?: number;

  featureCount?: number;

  status: 'success' | 'error';

  message?: string;
}

export interface DiagnosticMetric {
  title: string;
  icon: string;
  headline: string;
  description: string;
  metricValue: string;
  badgeText: string;
  isAbnormal: boolean;
  normalRange?: string;
}

export interface SubsystemDetail {
  id: SubsystemId;
  name: string;
  shortName: string;
  icon: string;
  activePillText: string;

  defaultFile: TelemetryFile;

  hasCriticalAlert: boolean;

  alertHeadline: string;

  alertDescription: string;

  actionRequiredBadge: string;

  whatToDoText: string;

  recommendations: Array<{
    id: number;
    title: string;
    description: string;
  }>;

  reportTitle: string;
}