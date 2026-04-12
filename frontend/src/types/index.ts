export interface Team {
  team_id: number
  team_name: string
  abbreviation: string
  city?: string
  conference?: string
  division?: string
  final_standing?: number
  lottery_odds_pct?: number
  wins?: number
  losses?: number
  otl?: number
  points?: number
  combinations_count?: number
  gm_name?: string
  gm_archetype?: ArchetypeType
  logo_url?: string
}

export type ArchetypeType = 'BPA' | 'need-based' | 'safe' | 'boom-bust' | 'system-fit'

export interface LotteryResult {
  pick_order: LotteryPick[]
  seed?: number
}

export interface LotteryPick {
  pick: number
  team_id: number
  team_name: string
  abbreviation: string
  gm_name?: string
  original_standing?: number
  lottery_odds_pct?: number
}

export interface LotterySimulationTeam {
  team_id: number
  team_name: string
  abbreviation: string
  odds_pct: number
  standing: number
  overall_rank?: number
  in_playoffs?: boolean
  lottery_slot?: number | null
  wins?: number
  losses?: number
  otl?: number
  points?: number
}

export interface DraftSimulationPick {
  pick: number
  team_id: number
  team_name: string
  abbreviation: string
  prospect_name: string
  prospect_id: number
  position?: string
  nationality?: string
  draft_league?: string
  css_rank?: number
  points_per_game?: number
  ml_score?: number
  in_prediction_set?: boolean | null
  nc_score?: number | null
  confidence?: number | null
}

export interface Prospect {
  id: number
  name: string
  position: string
  nationality?: string
  css_ranking?: number
  css_category?: string
  draft_league?: string
  draft_league_tier?: number
  points?: number
  goals?: number
  assists?: number
  games_played?: number
  points_per_game?: number
  age_at_draft?: number
  height_cm?: number
  weight_kg?: number
}

export interface TeamTendency {
  team_id: number
  team_name: string
  abbreviation: string
  gm?: { id: number; name: string; start_date?: string }
  archetype?: ArchetypeType
  tendency: {
    position_weights?: Record<string, number>
    league_weights?: Record<string, number>
    nationality_weights?: Record<string, number>
    avg_ranking_deviation?: number
  }
  team_needs?: Record<string, number>
  prospect_counts?: Record<string, number>
  ideal_prospect_depth?: Record<string, number>
  draft_history?: DraftPickRecord[]
  total_picks?: number
  year_range?: string
}

export interface DraftPickRecord {
  year: number
  round: number
  pick_number: number
  overall_pick: number
  player_name?: string
  position?: string
  draft_league?: string
  nationality?: string
  games_played_nhl?: number
}
