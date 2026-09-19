import { SubsystemDetail, DepotInfo } from '../types';

export const DEPOTS: DepotInfo[] = [
  { id: 'tuas-west', name: 'Tuas West Depot', line: 'East-West Line', activeTrains: 28 },
  { id: 'bishan', name: 'Bishan Depot', line: 'North-South Line', activeTrains: 34 },
  { id: 'ulu-pandan', name: 'Ulu Pandan Depot', line: 'East-West Line', activeTrains: 31 },
  { id: 'changi', name: 'Changi Depot', line: 'East-West / Changi Branch', activeTrains: 19 },
  { id: 'kim-chuan', name: 'Kim Chuan Depot', line: 'Circle / Downtown Line', activeTrains: 42 },
];

export const SUBSYSTEM_DATA: Record<string, SubsystemDetail> = {
  door: {
    id: 'door',
    name: 'Door',
    shortName: 'Door',
    icon: 'sensor_door',
    activePillText: 'Door Active',
    defaultFile: {
      name: 'Set541_Door4_Telemetry.csv',
      sizeMb: 42.8,
      format: 'csv',
      fleetId: 'SMRT East-West Line Fleet #541',
      status: 'active',
      uploadTime: '10 mins ago'
    },
    hasCriticalAlert: true,
    alertHeadline: 'Abnormal Resistance',
    alertDescription: 'Door subsystem Abnormal Resistance detected. Object or dirt in track.',
    actionRequiredBadge: 'ACTION NEEDED',
    whatToDoText: 'Check guide rails & clean debris around door drive belt.',
    recommendations: [
      {
        id: 1,
        title: 'Isolate & Clear Rail Track',
        description: 'Check guide rails & Door 4 runner track for obstruction or foreign debris.'
      },
      {
        id: 2,
        title: 'Perform Precision Inspection',
        description: 'Inspect drive belt tension and motor actuator linkage before manual cycling.'
      },
      {
        id: 3,
        title: 'Verify Laser Tolerance & Re-test',
        description: 'Perform optical clearance calibration & automated 5-cycle door stroke test.'
      }
    ],
    reportTitle: 'Download Door Diagnostic Report (PDF)'
  },
  acv: {
    id: 'acv',
    name: 'ACV',
    shortName: 'ACV',
    icon: 'mode_fan',
    activePillText: 'ACV Active',
    defaultFile: {
      name: 'Set541_ACV_Formation_Log.parquet',
      sizeMb: 18.4,
      format: 'parquet',
      fleetId: 'SMRT East-West Line Fleet #541',
      status: 'active',
      uploadTime: '14 mins ago'
    },
    hasCriticalAlert: true,
    alertHeadline: 'Car 3 ACV Compressor High Load',
    alertDescription: 'HVAC evaporator coil icing or refrigerant leak detected on Carriage 3.',
    actionRequiredBadge: 'ACTION NEEDED',
    whatToDoText: 'Inspect Car 3 HVAC compressor unit and check refrigerant pressure levels.',
    recommendations: [
      {
        id: 1,
        title: 'Isolate Car 3 Inverter Power',
        description: 'De-energize auxiliary converter feed before servicing roof-mounted HVAC unit.'
      },
      {
        id: 2,
        title: 'Clean Evaporator & Intake Filters',
        description: 'Remove dust accumulation on air filters and check return air temperature differential.'
      },
      {
        id: 3,
        title: 'Verify R134a Refrigerant Pressure',
        description: 'Re-charge refrigerant if below 4.2 bar gauge pressure and conduct leak test.'
      }
    ],
    reportTitle: 'Download ACV Diagnostic Report (PDF)'
  },
  corrugation: {
    id: 'corrugation',
    name: 'Rail Corrugation',
    shortName: 'Rail Corrugation',
    icon: 'waves',
    activePillText: 'Rail Corrugation Active',
    defaultFile: {
      name: 'TuasWest_TrackGeo_Ch214.las',
      sizeMb: 112.6,
      format: 'las',
      fleetId: 'Tuas West Extension Ch 214+50 to 216+00',
      status: 'active',
      uploadTime: '22 mins ago'
    },
    hasCriticalAlert: true,
    alertHeadline: 'Side II (Right Rail) Corrugation Exceeded',
    alertDescription: 'Wavelength 30–80mm detected with 0.28mm roughness depth exceeding safety margin.',
    actionRequiredBadge: 'ACTION NEEDED',
    whatToDoText: 'Dispatch Rail Grinder RGH20C during tonight non-revenue engineering window.',
    recommendations: [
      {
        id: 1,
        title: 'Schedule Rail Grinding Train',
        description: 'Deploy Speno Rail Grinder for 3-pass reprofiling between Ch 214+50 and 215+80.'
      },
      {
        id: 2,
        title: 'Acoustic Roughness Validation',
        description: 'Measure acoustic rail roughness compliance against ISO 3095 standards post-grind.'
      },
      {
        id: 3,
        title: 'Check Fastener & Pad Degradation',
        description: 'Inspect Vossloh W14 rail fastening clips and elastic baseplate pads for loosening.'
      }
    ],
    reportTitle: 'Download Rail Corrugation Report (PDF)'
  },
  shm: {
    id: 'shm',
    name: 'SHM',
    shortName: 'SHM',
    icon: 'monitor_heart',
    activePillText: 'SHM Active',
    defaultFile: {
      name: 'Bogie_StrainGauge_Array.h5',
      sizeMb: 29.1,
      format: 'h5',
      fleetId: 'SMRT East-West Line Fleet #541 Bogie 1 & 2',
      status: 'active',
      uploadTime: '35 mins ago'
    },
    hasCriticalAlert: false,
    alertHeadline: 'Structural Health Nominal',
    alertDescription: 'All bogie frame multiaxial stress cycles and axle box accelerations within safe boundaries.',
    actionRequiredBadge: 'HEALTHY',
    whatToDoText: 'No corrective action required. Routine acoustic emission scan scheduled in 4,500 km.',
    recommendations: [
      {
        id: 1,
        title: 'Record Bogie Life Cumulative Load',
        description: 'Log 68% cumulative fatigue damage index into Depot Maximo Asset Management.'
      },
      {
        id: 2,
        title: 'Axle Box Bearing Temperature Check',
        description: 'Infrared temperature scan during wheel-turn inspection confirms nominal 42°C delta.'
      },
      {
        id: 3,
        title: 'Next Scheduled Ultrasonic Test',
        description: 'Perform standard phased array ultrasonic axle testing at 750,000 km overhaul.'
      }
    ],
    reportTitle: 'Download SHM Diagnostic Report (PDF)'
  }
};
