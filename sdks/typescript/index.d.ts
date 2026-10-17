export type Text = string | Record<string, unknown> | unknown[];
export type State = Text;

export interface Choice { type: "choice"; instructions: Text; criteria: Record<string, Text | null> }
export interface Score { type: "score"; instructions: Text; criteria: Text[] }
export interface NoulCriteria { true?: Text | null; false?: Text | null }
export interface Noul { type: "noul"; instructions?: Text | null; criteria?: NoulCriteria }
/** Where something is in an image: what to find, naming the image as `<image N>` when the state holds several. */
export interface Box { type: "box"; instructions: Text }
export type Question = Choice | Score | Noul | Box;

/** An image in the state, or an asset the text names: PNG, JPEG or WebP, base64 without a data: prefix. */
export interface ImageBlock { type: "image"; media_type: "image/png" | "image/jpeg" | "image/webp"; data: string }

export interface ChoiceAnswer { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number }
export interface ScoreAnswer { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number }
export interface NoulAnswer { type: "noul"; noul: number }
/** Edges as fractions of the image from its top-left corner, each from zero to one. */
export interface BoxCoordinates { top: number; bottom: number; left: number; right: number }
export interface BoxAnswer { type: "box"; box: BoxCoordinates; confidence: number }
export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer | BoxAnswer;

export interface SystemOneResponse<Q extends Record<string, Question>> {
  /** Present when the server keeps a ledger: the decision to report an outcome for. */
  id?: string;
  model: string;
  answers: { [K in keyof Q]: Q[K] extends Choice ? ChoiceAnswer : Q[K] extends Score ? ScoreAnswer : Q[K] extends Box ? BoxAnswer : NoulAnswer };
  usage: { input_tokens: number; output_tokens: number };
}

export interface ModelInfo {
  name: string; description: string; release_date: string;
  id: string; base: string; revision: string | null; adapter: string | null;
  calibration: Record<string, number>; max_input_tokens: number; max_branch_tokens: number;
  long_context_tokens: number | null;
  max_options: number; rotations: number; prior_debias: boolean;
  modalities: ("text" | "image")[]; question_types: ("choice" | "score" | "noul" | "box")[];
}

/** A hard label (option key, level index, true/false) or a distribution over every option. */
export type Label = string | number | boolean | number[] | Record<string, number>;

export interface ApiError { code: string; message: string; param?: string }

export function choice(instructions: Text, criteria: Record<string, Text | null>): Choice;
export function score(instructions: Text, criteria: Text[]): Score;
export function noul(instructions: Text | null, criteria?: NoulCriteria): Noul;
export function box(instructions: Text): Box;
/** An image block from raw bytes or base64 text, for the state or for `assets`. */
export function image(data: Uint8Array | string, mediaType: ImageBlock["media_type"]): ImageBlock;

export class RulingError extends Error { status: number; detail: ApiError | unknown }

export class RulingClient {
  constructor(options?: { baseUrl?: string; apiKey?: string; fetch?: typeof fetch });
  systemOne<Q extends Record<string, Question>>(input: {
    state: State; questions: Q; model?: string; excludeFromTraining?: boolean;
    /** Images the text names, by the name it uses: `{ "<photo>": image(bytes, "image/png") }`. */
    assets?: Record<string, ImageBlock>;
    /** Ask the server for its long-context limits, as the hosted API's danger-zone header does. */
    longContext?: boolean;
  }): Promise<SystemOneResponse<Q>>;
  outcome(decisionId: string, labels: Record<string, Label>): Promise<Record<string, Label>>;
  models(): Promise<ModelInfo[]>;
}
