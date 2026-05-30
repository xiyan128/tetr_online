/**
 * Framework-free entry points: `mount()` and the `<tetris-game>` custom element.
 *
 * These let the board drop into any page — including the static Hugo blog — with no
 * React/Preact runtime on the host page (Preact is bundled in). The custom element
 * renders into a shadow root so its inline styles never touch the host page.
 */

import { render } from "preact";

import { TetrisGame, type TetrisGameProps } from "./component";

/** Imperatively mount a board into `target`. Returns a handle to tear it down. */
export function mount(target: Element, props: TetrisGameProps = {}): { destroy: () => void } {
  render(<TetrisGame {...props} />, target);
  return {
    destroy() {
      render(null, target);
    },
  };
}

const OBSERVED = [
  "theme",
  "seed",
  "cell",
  "height",
  "reaction",
  "imperfection",
  "sidebar",
  "interactive",
  "glue-url",
  "label",
] as const;

class TetrisGameElement extends HTMLElement {
  private root: ShadowRoot | null = null;

  static get observedAttributes(): readonly string[] {
    return OBSERVED;
  }

  connectedCallback(): void {
    if (!this.root) this.root = this.attachShadow({ mode: "open" });
    this.renderInto();
  }

  disconnectedCallback(): void {
    if (this.root) render(null, this.root);
  }

  attributeChangedCallback(): void {
    if (this.root) this.renderInto();
  }

  private renderInto(): void {
    render(<TetrisGame {...this.readProps()} />, this.root!);
  }

  private readProps(): TetrisGameProps {
    // Return undefined (→ the component's default) for an absent OR malformed
    // numeric attribute. Without the finite check, `Number("abc")` → NaN and
    // `Number("")` → 0 would slip past the component's `= default` / `?? default`
    // fallbacks (which only fire on undefined) and corrupt the seed / geometry / AI.
    const numAttr = (name: string): number | undefined => {
      const raw = this.getAttribute(name)?.trim();
      if (!raw) return undefined; // absent or empty → the component's default
      const n = Number(raw);
      return Number.isFinite(n) ? n : undefined;
    };
    const boolAttr = (name: string, dflt: boolean): boolean =>
      this.hasAttribute(name) ? this.getAttribute(name) !== "false" : dflt;
    return {
      theme: this.getAttribute("theme") ?? undefined,
      seed: numAttr("seed"),
      cell: numAttr("cell"),
      height: numAttr("height"),
      reaction: numAttr("reaction"),
      imperfection: numAttr("imperfection"),
      sidebar: boolAttr("sidebar", true),
      interactive: boolAttr("interactive", true),
      glueUrl: this.getAttribute("glue-url") ?? undefined,
      label: this.getAttribute("label") ?? undefined,
    };
  }
}

/** Define `<tetris-game>` (idempotent). Call once after the bundle loads. */
export function registerElement(tag = "tetris-game"): void {
  if (typeof customElements !== "undefined" && !customElements.get(tag)) {
    customElements.define(tag, TetrisGameElement);
  }
}
