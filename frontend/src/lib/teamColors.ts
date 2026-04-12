export interface TeamColors {
  primary: string
  secondary: string
}

export const TEAM_COLORS: Record<string, TeamColors> = {
  ANA: { primary: '#F47A38', secondary: '#B09862' },
  BOS: { primary: '#FFB81C', secondary: '#000000' },
  BUF: { primary: '#003087', secondary: '#FCB514' },
  CGY: { primary: '#D2001C', secondary: '#FAAF19' },
  CAR: { primary: '#CC0000', secondary: '#000000' },
  CHI: { primary: '#CF0A2C', secondary: '#000000' },
  COL: { primary: '#6F263D', secondary: '#236192' },
  CBJ: { primary: '#002654', secondary: '#CE1126' },
  DAL: { primary: '#006847', secondary: '#8F8F8C' },
  DET: { primary: '#CE1126', secondary: '#FFFFFF' },
  EDM: { primary: '#FF4C00', secondary: '#041E42' },
  FLA: { primary: '#C8102E', secondary: '#041E42' },
  LAK: { primary: '#111111', secondary: '#A2AAAD' },
  MIN: { primary: '#154734', secondary: '#A6192E' },
  MTL: { primary: '#AF1E2D', secondary: '#192168' },
  NSH: { primary: '#FFB81C', secondary: '#041E42' },
  NJD: { primary: '#CE1126', secondary: '#000000' },
  NYI: { primary: '#00539B', secondary: '#F47D30' },
  NYR: { primary: '#0038A8', secondary: '#CE1126' },
  OTT: { primary: '#E31837', secondary: '#000000' },
  PHI: { primary: '#F74902', secondary: '#000000' },
  PIT: { primary: '#FCB514', secondary: '#000000' },
  SEA: { primary: '#001628', secondary: '#99D9D9' },
  SJS: { primary: '#006D75', secondary: '#EA7200' },
  STL: { primary: '#002F87', secondary: '#FCB514' },
  TBL: { primary: '#002868', secondary: '#FFFFFF' },
  TOR: { primary: '#00205B', secondary: '#FFFFFF' },
  UTA: { primary: '#69B3E7', secondary: '#010101' },
  VAN: { primary: '#00843D', secondary: '#00205B' },
  VGK: { primary: '#B4975A', secondary: '#333F48' },
  WSH: { primary: '#C8102E', secondary: '#041E42' },
  WPG: { primary: '#041E42', secondary: '#004C97' },
}

export function getTeamColors(abbreviation: string): TeamColors {
  return TEAM_COLORS[abbreviation] ?? { primary: '#3b82f6', secondary: '#1e293b' }
}

export function getTeamLogo(abbreviation: string): string {
  return `https://assets.nhle.com/logos/nhl/svg/${abbreviation}_light.svg`
}
