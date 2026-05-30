/**
 * Human take-over for one board.
 *
 * The board autoplays until the visitor interacts; then control is theirs until they
 * release it. Wiring:
 *
 * - **pointer down / focus** on the canvas → switch the wasm `Game` to human mode and
 *   focus it for keyboard input;
 * - **keydown / keyup** map `KeyboardEvent.code` to the engine's action indices and
 *   forward them (game keys are `preventDefault`-ed so Space/Arrows don't scroll the
 *   page);
 * - **Escape**, **blur**, or **a few seconds idle** → release control back to the AI.
 *
 * Action indices match the wasm `action_bit` mapping: 0 left, 1 right, 2 soft, 3
 * hard, 4 CW, 5 CCW, 6 hold (7 pause is unused here).
 */

import type { WasmGame } from "./wasm";

export type ControlState = "ai" | "human";

const KEYMAP: Readonly<Record<string, number>> = {
  ArrowLeft: 0,
  ArrowRight: 1,
  ArrowDown: 2,
  Space: 3,
  ArrowUp: 4,
  KeyX: 4,
  KeyZ: 5,
  ControlLeft: 5,
  ControlRight: 5,
  ShiftLeft: 6,
  KeyC: 6,
};

export interface TakeoverOptions {
  /** Auto-release control after this many ms with no key activity. */
  idleMs: number;
  /** Notified whenever control switches (drives the overlay hint). */
  onState: (state: ControlState) => void;
}

export class Takeover {
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private attached = false;

  constructor(
    private canvas: HTMLCanvasElement,
    private game: WasmGame,
    private opts: TakeoverOptions,
  ) {}

  attach(): void {
    if (this.attached) return;
    this.attached = true;
    this.canvas.tabIndex = 0;
    this.canvas.style.cursor = "pointer";
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("keydown", this.onKeyDown);
    this.canvas.addEventListener("keyup", this.onKeyUp);
    this.canvas.addEventListener("blur", this.onBlur);
  }

  detach(): void {
    if (!this.attached) return;
    this.attached = false;
    this.clearIdle();
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("keydown", this.onKeyDown);
    this.canvas.removeEventListener("keyup", this.onKeyUp);
    this.canvas.removeEventListener("blur", this.onBlur);
  }

  private enter(): void {
    if (this.game.is_human()) return;
    this.game.set_mode_human();
    this.opts.onState("human");
    this.bumpIdle();
  }

  private exit(): void {
    if (!this.game.is_human()) return;
    this.game.set_mode_ai();
    this.opts.onState("ai");
    this.clearIdle();
  }

  private onPointerDown = (): void => {
    this.canvas.focus();
    this.enter();
  };

  private onBlur = (): void => {
    this.exit();
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    if (e.key === "Escape") {
      this.exit();
      this.canvas.blur();
      return;
    }
    const action = KEYMAP[e.code];
    if (action === undefined) return;
    e.preventDefault(); // stop Space/Arrows from scrolling the page
    this.enter();
    if (!e.repeat) this.game.key_down(action);
    this.bumpIdle();
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    const action = KEYMAP[e.code];
    if (action === undefined) return;
    e.preventDefault();
    this.game.key_up(action);
  };

  private bumpIdle(): void {
    this.clearIdle();
    this.idleTimer = setTimeout(() => this.exit(), this.opts.idleMs);
  }

  private clearIdle(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }
}
