/**
 * `<TetrisGame>` — the embeddable Preact component.
 *
 * Boots the wasm `Game`, renders it to a canvas via [`BoardRunner`], and (when
 * `interactive`) wires [`Takeover`] so a click hands control to the visitor. It owns
 * the lifecycle: one wasm `Game` per mounted component, torn down on unmount.
 *
 * Preact's API is React-compatible (`preact/compat`), so this is usable from React
 * too; for non-framework pages use `mount()` / the `<tetris-game>` element.
 */

import { useEffect, useRef, useState } from "preact/hooks";

import { Takeover, type ControlState } from "./controls";
import { BoardRunner } from "./loop";
import { computeLayout, prepareTheme } from "./renderer";
import { monoTheme, resolveTheme, type Theme } from "./themes";
import { loadGameClass, type WasmGame } from "./wasm";

export interface TetrisGameProps {
  /** Theme name (`fourlol` | `light` | `blog` | `mono`) or a partial override. */
  theme?: string | Partial<Theme>;
  /** Engine seed; random per mount if omitted. */
  seed?: number;
  /** AI reaction delay in ms (higher = more human, more beatable). */
  reaction?: number;
  /** AI error rate, 0..1 (higher = weaker). */
  imperfection?: number;
  /** Pixels per cell. Overrides `height`. */
  cell?: number;
  /** Target board height in px; the cell size is derived from it. */
  height?: number;
  /** Draw the HOLD + NEXT rail. */
  sidebar?: boolean;
  /** Allow click-to-take-over. When false the board is a pure animation. */
  interactive?: boolean;
  /**
   * Fill the parent's width: cell size is derived from the container so the board
   * spans the full column (overrides `cell`/`height`). The board height stays the
   * full 20 rows, a constant — it never resizes, so it never reflows the page.
   */
  fullWidth?: boolean;
  /**
   * While the AI plays, draw the board monochrome in `monoColor` (default: the
   * canvas's computed CSS `color`, so it matches the page/theme). On take-over it
   * switches to the full-colour `theme`. Set `false` to always use `theme`.
   */
  monochromeWhileAI?: boolean;
  /** Override the monochrome ink colour; defaults to the canvas's computed text colour. */
  monoColor?: string;
  /**
   * Show the built-in floating "Click to play / Esc to release" hint over the board.
   * Default true; set false when the host page provides its own affordance.
   */
  showHint?: boolean;
  /** URL of the wasm-bindgen glue (`tetr_embed.js`); resolved against the page base. */
  glueUrl?: string;
  /** Accessible label for the canvas. */
  label?: string;
  /**
   * Notified whenever control switches between the AI (`"ai"`) and the visitor
   * (`"human"`) — on take-over, release, and an auto-release after a human top-out.
   * Lets the host page reflect the state (e.g. swap a caption).
   */
  onControlChange?: (state: ControlState) => void;
}

const ROWS_FALLBACK = 20;

function randomSeed(): number {
  return (Math.random() * 0x1_0000_0000) >>> 0;
}

function prefersReducedMotion(): boolean {
  return typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function TetrisGame(props: TetrisGameProps) {
  const {
    seed,
    reaction = 220,
    imperfection = 0.12,
    cell,
    height = 460,
    sidebar = true,
    interactive = true,
    fullWidth = false,
    monochromeWhileAI = false,
    monoColor,
    showHint = true,
    glueUrl,
    label = "Tetris — an AI plays automatically; click to take over",
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runnerRef = useRef<BoardRunner | null>(null);
  const gameRef = useRef<WasmGame | null>(null);
  const takeoverRef = useRef<Takeover | null>(null);
  const themeObserverRef = useRef<MutationObserver | null>(null);

  const [status, setStatus] = useState<"loading" | "running" | "error">("loading");
  const [control, setControl] = useState<ControlState>("ai");

  // Boot once. The wasm `Game`, runner, and take-over are torn down on unmount.
  useEffect(() => {
    let disposed = false;
    const canvas = canvasRef.current!;

    (async () => {
      try {
        const Game = await loadGameClass(glueUrl);
        if (disposed) return;
        const game = new Game(seed ?? randomSeed(), reaction, imperfection);
        gameRef.current = game;

        const cols = game.board_width();
        const rows = game.visible_height() || ROWS_FALLBACK;

        // Cell size: from the container width (full-width), an explicit `cell`, or
        // derived from the target `height`. `fullWidth` falls back to a sane cell if
        // the container hasn't been laid out yet (clientWidth 0).
        const containerW = canvas.parentElement?.clientWidth || cols * 24;
        const px = fullWidth
          ? Math.max(8, Math.floor(containerW / cols))
          : cell ?? Math.max(12, Math.floor(height / rows));

        // Resolved colour theme (used on take-over) and the monochrome AI theme.
        // The mono ink is read LIVE from the canvas's computed text colour each time,
        // so a light/dark theme toggle is reflected immediately (see the observer
        // below) rather than frozen at mount.
        const colorTheme = resolveTheme(props.theme);
        const inkNow = () => monoColor ?? (getComputedStyle(canvas).color || "#888");
        const paletteFor = (c: ControlState) =>
          prepareTheme(
            c === "human" || !monochromeWhileAI ? colorTheme : monoTheme(inkNow(), colorTheme),
          );

        // The board shows its full height (a constant), so the canvas never resizes
        // and never reflows the page.
        const layout = computeLayout(cols, rows, {
          cell: px,
          sidebar,
          pad: fullWidth ? 0 : Math.round(px * 0.5),
        });

        const dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
        // One place that applies a control change: repaint the palette (mono ↔ colour
        // without restarting), update local + parent state, and remember it for the
        // theme observer. Used by take-over AND by the loop's human-top-out release.
        let controlNow: ControlState = "ai";
        const applyControl = (c: ControlState) => {
          controlNow = c;
          setControl(c);
          if (monochromeWhileAI) runner.setPrepared(paletteFor(c));
          props.onControlChange?.(c);
        };

        const runner = new BoardRunner(canvas, game, paletteFor("ai"), layout, {
          autoReset: true,
          nextSeed: randomSeed,
          reducedMotion: prefersReducedMotion(),
          dpr,
          // A human top-out ends their turn → reflect the return to autoplay.
          onControlReturn: () => applyControl("ai"),
        });
        runnerRef.current = runner;
        runner.start();

        if (interactive) {
          const takeover = new Takeover(canvas, game, { idleMs: 6000, onState: applyControl });
          takeover.attach();
          takeoverRef.current = takeover;
        }

        // Re-derive the monochrome ink when the site theme toggles. The blog flips a
        // `class` on <body> (light-theme/dark-theme); when it changes, the canvas's
        // inherited `color` changes too, so repaint the AI palette from the new ink.
        if (monochromeWhileAI && typeof MutationObserver !== "undefined") {
          const obs = new MutationObserver(() => {
            if (controlNow !== "human") runner.setPrepared(paletteFor("ai"));
          });
          obs.observe(document.body, { attributes: true, attributeFilter: ["class"] });
          themeObserverRef.current = obs;
        }

        setStatus("running");
      } catch (err) {
        console.error("tetris-embed failed to start:", err);
        if (!disposed) setStatus("error");
      }
    })();

    return () => {
      disposed = true;
      themeObserverRef.current?.disconnect();
      takeoverRef.current?.detach();
      runnerRef.current?.stop();
      gameRef.current?.free();
      themeObserverRef.current = null;
      takeoverRef.current = null;
      runnerRef.current = null;
      gameRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to colour-theme changes without restarting. Skipped while
  // `monochromeWhileAI` owns the palette (the take-over callback drives it instead,
  // so this would otherwise force the colour theme over the AI's monochrome).
  useEffect(() => {
    if (monochromeWhileAI) return;
    runnerRef.current?.setPrepared(prepareTheme(resolveTheme(props.theme)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [typeof props.theme === "string" ? props.theme : JSON.stringify(props.theme)]);

  const wrapStyle = fullWidth
    ? { position: "relative" as const, display: "block", width: "100%", lineHeight: 0 }
    : { position: "relative" as const, display: "inline-block", lineHeight: 0 };

  return (
    <div style={wrapStyle}>
      <canvas ref={canvasRef} aria-label={label} role="img" style={{ display: "block", outline: "none", touchAction: "none" }} />
      {status !== "running" && (
        <div style={overlayStyle}>{status === "error" ? "Couldn’t start" : "…"}</div>
      )}
      {status === "running" && interactive && showHint && <Hint control={control} />}
    </div>
  );
}

function Hint({ control }: { control: ControlState }) {
  const text = control === "human" ? "Esc to release" : "Click to play";
  return (
    <div
      style={{
        position: "absolute",
        bottom: 8,
        left: "50%",
        transform: "translateX(-50%)",
        padding: "3px 10px",
        borderRadius: 999,
        font: "500 11px ui-sans-serif, system-ui, sans-serif",
        letterSpacing: "0.02em",
        color: control === "human" ? "#fff" : "rgba(255,255,255,0.82)",
        background: control === "human" ? "rgba(40,120,220,0.82)" : "rgba(20,22,28,0.55)",
        backdropFilter: "blur(4px)",
        pointerEvents: "none",
        opacity: 0.92,
        transition: "background 0.2s ease, color 0.2s ease",
        userSelect: "none",
      }}
    >
      {text}
    </div>
  );
}

const overlayStyle = {
  position: "absolute" as const,
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  font: "13px ui-sans-serif, system-ui, sans-serif",
  color: "#888",
};
