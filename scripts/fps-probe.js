// Real in-game FPS / frame-time / main-thread-stall probe.
//
// Why: the shipping wasm build (`--profile web`) has `debug_assertions` off, so
// Bevy's diagnostics are compiled out — there is no FPS readout in the real game.
// This measures the *actual* rendered-frame cadence (what you feel) plus the
// main-thread long-tasks that cause it (the AI polls), with zero rebuild.
//
// Usage: open the running game (dev server or the deployed page), paste this into
// the browser devtools Console, play a match with the bot you want to measure,
// then call  __fpsReport()  . __fpsReset() clears the buffers between trials.
//
// What to read:
//   - frame time p99 / max   = the stutter you perceive (16.7ms = one 60Hz frame)
//   - "janky frames"         = % of frames that missed the 60Hz budget
//   - "long tasks"           = main-thread blocks >50ms — these ARE the AI polls
//     running on the render thread; their max ≈ the worst single freeze.
(() => {
  const frames = []; // inter-frame deltas (ms), i.e. perceived frame time
  const longtasks = []; // main-thread blocks >50ms (Long Tasks API)
  let last = performance.now();

  const loop = (t) => {
    frames.push(t - last);
    last = t;
    // Keep the buffer bounded over long sessions.
    if (frames.length > 200000) frames.splice(0, 100000);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) longtasks.push(e.duration);
    }).observe({ entryTypes: ["longtask"] });
  } catch (e) {
    console.warn("[fps-probe] Long Tasks API unavailable in this browser:", e);
  }

  const pct = (arr, p) => {
    if (!arr.length) return 0;
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(s.length * p))];
  };

  window.__fpsReport = () => {
    const f = frames;
    if (!f.length) return console.log("[fps-probe] no frames captured yet");
    const n = f.length;
    const mean = f.reduce((a, b) => a + b, 0) / n;
    const over = (ms) => f.filter((d) => d > ms).length;
    console.log(
      `%c[fps-probe] ${n} frames  mean ${mean.toFixed(1)}ms (${(1000 / mean).toFixed(0)} fps)`,
      "font-weight:bold",
    );
    console.log(
      `  frame time   p50 ${pct(f, 0.5).toFixed(1)}  p95 ${pct(f, 0.95).toFixed(1)}  p99 ${pct(f, 0.99).toFixed(1)}  max ${Math.max(...f).toFixed(1)} ms`,
    );
    console.log(
      `  janky frames  >16.7ms ${over(16.7)} (${((100 * over(16.7)) / n).toFixed(1)}%)   >33ms ${over(33)}   >50ms ${over(50)}`,
    );
    if (longtasks.length) {
      console.log(
        `  long tasks (main-thread blocks >50ms = AI polls)  count ${longtasks.length}  p50 ${pct(longtasks, 0.5).toFixed(0)}  p99 ${pct(longtasks, 0.99).toFixed(0)}  max ${Math.max(...longtasks).toFixed(0)} ms`,
      );
    } else {
      console.log("  long tasks: none recorded (or API unsupported)");
    }
    return {
      frames: n,
      meanMs: +mean.toFixed(2),
      p99Ms: +pct(f, 0.99).toFixed(2),
      maxMs: +Math.max(...f).toFixed(2),
      longTasks: longtasks.length,
      worstStallMs: longtasks.length ? +Math.max(...longtasks).toFixed(0) : 0,
    };
  };
  window.__fpsReset = () => {
    frames.length = 0;
    longtasks.length = 0;
    last = performance.now();
    console.log("[fps-probe] reset");
  };
  console.log("[fps-probe] armed. Play a match, then call __fpsReport(). __fpsReset() to clear.");
})();
