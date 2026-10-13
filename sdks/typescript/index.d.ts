export type Text = string | Record<string, unknown> | unknown[];
export type State = Text;

export interface Choice { type: "choice"; instructions: Text; criteria: Record<string, Text | null> }
export interface Score { type: "score"; instructions: Text; criteria: Text[] }
export interface NoulCriteria { true?: Text | null; false?: Text | null }
export interface Noul { type: "noul"; instructions?: Text | null; criteria?: NoulCriteria }
export type Question = Choice | Score | Noul;

export interface ChoiceAnswer { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number }
export interface ScoreAnswer { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number }
export interface NoulAnswer { type: "noul"; noul: number }
export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface SystemOneResponse<Q extends Record<string, Question>> {
  model: string;
  answers: { [K in keyof Q]: Q[K] extends Choice ? ChoiceAnswer : Q[K] extends Score ? ScoreAnswer : NoulAnswer };
  usage: { input_tokens: number; output_tokens: number };
}

export interface ModelInfo {
  name: string; description: string; release_date: string;
  id: string; base: string; revision: string | null; adapter: string | null;
  calibration: Record<string, number>; max_input_tokens: number; max_branch_tokens: number;
  max_options: number; rotations: number; prior_debias: boolean;
}

export interface ApiError { code: string; message: string; param?: string }

export function choice(instructions: Text, criteria: Record<string, Text | null>): Choice;
export function score(instructions: Text, criteria: Text[]): Score;
export function noul(instructions: Text | null, criteria?: NoulCriteria): Noul;

export class RulingError extends Error { status: number; detail: ApiError | unknown }

export class RulingClient {
  constructor(options?: { baseUrl?: string; apiKey?: string; fetch?: typeof fetch });
  systemOne<Q extends Record<string, Question>>(input: { state: State; questions: Q; model?: string }): Promise<SystemOneResponse<Q>>;
  models(): Promise<ModelInfo[]>;
}
