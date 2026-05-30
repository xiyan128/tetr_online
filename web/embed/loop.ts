/**
 * The render/sim loop for one board.
 *
 * Each `requestAnimationFrame` it feeds the real elapsed time to `game.tick` (the
 * wasm side owns the fixed 60 Hz accumulator), decays effects, and redraws. It also:
 *
 * - **pauses off-screen** via an `IntersectionObserver` (a blog page can have many
 *   boards; only visible ones run), and when the tab is hidden (rAF stops anyway);
 * - honors **`prefers-reduced-motion`** by running the sim at half speed and
 *   suppressing flashes — a calmer animation rather than a frozen one;
 * - **auto-restarts** the autoplay on top-out (so the animation never dead-ends),
 *   unless a human is in control.
 */

import { drawFrame, setupCanvas, type Fx, type Layout, type PreparedTheme } from "./renderer";
import type { WasmGame } from "./wasm";

export interface RunnerOptions {
  /** Restart the game with a fresh seed shortly after a top-out (autoplay only). */
  autoReset: boolean;
  /** Produces the next seed for an auto-reset. */
  nextSeed: () => number;
  /** Run at half speed and suppress flashes (set from `prefers-reduced-motion`). */
  reducedMotion: boolean;
  /** Device pixel ratio to render at. */
  dpr: number;
  /** Called when a human top-out auto-releases control back to the AI. */
  onControlReturn?: () => void;
}

const FLASH_DECAY = 0.86; // per-frame multiplier
const GAME_OVER_HOLD = 1.1; // seconds to linger on a topped-out board before reset

export class BoardRunner {
  private ctx: CanvasRenderingContext2D;
  private raf = 0;
  private last = 0;
  private running = false;
  private visible = true;
  private fx: Fx = { flash: 0, dim: 0 };
  private overElapsed = 0;
  private io: IntersectionObserver | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    private game: WasmGame,
    private prepared: PreparedTheme,
    private layout: Layout,
    private opts: RunnerOptions,
  ) {
    this.ctx = setupCanvas(canvas, layout, opts.dpr);
  }

  /** Swap the theme without restarting. */
  setPrepared(pt: PreparedTheme): void {
    this.prepared = pt;
    this.draw();
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.observeVisibility();
    this.last = 0;
    this.draw(); // initial frame, so a board scrolled into view later isn't blank
    this.raf = requestAnimationFrame(this.frame);
  }

  stop(): void {
    this.running = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.io?.disconnect();
    this.io = null;
  }

  private observeVisibility(): void {
    if (typeof IntersectionObserver === "undefined") return;
    this.io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) this.visible = e.isIntersecting;
      },
      // threshold 0 = run as soon as any pixel is on screen; rootMargin starts a
      // board a little before it scrolls in (and is robust to layout timing).
      { rootMargin: "150px 0px", threshold: 0 },
    );
    this.io.observe(this.canvas);
  }

  private frame = (now: number): void => {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this.frame);

    const dt = this.last ? Math.min(0.1, (now - this.last) / 1000) : 0;
    this.last = now;

    // Off-screen: keep the rAF alive (cheap) but don't simulate or draw.
    if (!this.visible || dt === 0) return;

    const scale = this.opts.reducedMotion ? 0.5 : 1;
    const tags = this.game.tick(dt * scale);

    // Line-clear flash (event tag 2). Game-over is polled via game_over() below.
    if (!this.opts.reducedMotion) {
      for (let i = 0; i < tags.length; i++) {
        if (tags[i] === 2) this.fx.flash = 1;
      }
    }
    this.fx.flash *= FLASH_DECAY;

    // Top-out handling: dim the board, then after a beat restart it. This runs for
    // BOTH a human and the AI topping out — a human's game ends and the board returns
    // to ambient autoplay (so it never dead-ends on a frozen, lost board).
    if (this.game.game_over()) {
      this.fx.dim = Math.min(1, this.fx.dim + dt * 2);
      this.overElapsed += dt;
      if (this.opts.autoReset && this.overElapsed >= GAME_OVER_HOLD) {
        const wasHuman = this.game.is_human();
        this.game.reset(this.opts.nextSeed());
        if (wasHuman) {
          this.game.set_mode_ai();
          this.opts.onControlReturn?.();
        }
        this.fx = { flash: 0, dim: 0 };
        this.overElapsed = 0;
      }
    } else {
      this.fx.dim = 0;
      this.overElapsed = 0;
    }

    this.draw();
  };

  private draw(): void {
    drawFrame(this.ctx, this.game, this.prepared, this.layout, this.fx);
  }
}
