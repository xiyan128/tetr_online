//! TBP client for driving **Cold Clear 2** as a subprocess — the baseline opponent.
//!
//! Spawns the `cold-clear-2` binary and speaks the Tetris Bot Protocol over its
//! stdin/stdout (newline-delimited JSON), so we can have CC2 play the *same seeded
//! task* our bot plays and score its attack-per-piece with the *same* attack table
//! ([`tetr_core::engine::attack_lines`]). Native-only (subprocess); `tetr-research`
//! is already native-only.
//!
//! Protocol (verified against CC2 `0.1.0`): on launch the bot emits `info`; we send
//! `rules` and it replies `ready`; per game we send `start` (board + queue), then
//! loop `suggest` → `suggestion` → apply the move in our engine → `play` +
//! `new_piece`. CC2 searches continuously in the background, so think time is the
//! delay between advancing its state and asking it to `suggest`.

use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

/// A TBP board: 40 rows × 10 cells, row 0 = bottom. A cell is `None` (empty), a
/// piece letter (`"I"`..`"L"`), or `"G"` (garbage).
pub type TbpBoard = Vec<Vec<Option<String>>>;

/// Where a piece rests: SRS true-rotation center `(x, y)` + orientation.
#[derive(Debug, Clone, Deserialize)]
pub struct TbpLocation {
    #[serde(rename = "type")]
    pub piece: String,
    pub orientation: String,
    pub x: i32,
    pub y: i32,
}

/// A suggested placement: where the piece goes + the spin it was placed with.
#[derive(Debug, Clone, Deserialize)]
pub struct TbpMove {
    pub location: TbpLocation,
    #[serde(default)]
    pub spin: String,
}

/// Messages the bot sends us. Unknown variants/fields are ignored (forward-compat).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BotMsg {
    Info {
        name: String,
    },
    Ready,
    Error {
        reason: Option<String>,
    },
    Suggestion {
        moves: Vec<TbpMove>,
    },
    #[serde(other)]
    Unknown,
}

/// A handle to a running Cold Clear 2 process speaking TBP.
pub struct Cc2 {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// The bot's self-reported name (from its `info` message).
    pub name: String,
}

impl Cc2 {
    /// Spawn the CC2 binary at `bin` and complete the handshake (info → rules → ready).
    pub fn spawn(bin: &str) -> io::Result<Self> {
        let mut child = Command::new(bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        let mut bot = Cc2 {
            child,
            stdin,
            stdout,
            name: String::new(),
        };
        match bot.recv()? {
            BotMsg::Info { name } => bot.name = name,
            other => return Err(protocol_err(format!("expected info, got {other:?}"))),
        }
        bot.send(json!({ "type": "rules" }))?;
        match bot.recv()? {
            BotMsg::Ready => Ok(bot),
            BotMsg::Error { reason } => {
                Err(protocol_err(format!("bot rejected rules: {reason:?}")))
            }
            other => Err(protocol_err(format!("expected ready, got {other:?}"))),
        }
    }

    fn send(&mut self, v: serde_json::Value) -> io::Result<()> {
        writeln!(self.stdin, "{v}")?;
        self.stdin.flush()
    }

    fn recv(&mut self) -> io::Result<BotMsg> {
        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "CC2 closed stdout",
                ));
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            return serde_json::from_str(line)
                .map_err(|e| protocol_err(format!("bad TBP message {line:?}: {e}")));
        }
    }

    /// Begin a new game from the given initial state (seven-bag randomizer).
    pub fn start(
        &mut self,
        board: &TbpBoard,
        queue: &[String],
        hold: Option<&str>,
        combo: u32,
        back_to_back: bool,
        bag_state: &[String],
    ) -> io::Result<()> {
        self.send(json!({
            "type": "start",
            "hold": hold,
            "queue": queue,
            "combo": combo,
            "back_to_back": back_to_back,
            "board": board,
            "randomizer": { "type": "seven_bag", "bag_state": bag_state },
        }))
    }

    /// Let CC2 think for `think`, then return its most-preferred move (if any).
    pub fn suggest(&mut self, think: Duration) -> io::Result<Option<TbpMove>> {
        // CC2 searches in the background after each state update; the sleep is its
        // think budget for this piece.
        std::thread::sleep(think);
        self.send(json!({ "type": "suggest" }))?;
        match self.recv()? {
            BotMsg::Suggestion { moves } => Ok(moves.into_iter().next()),
            other => Err(protocol_err(format!("expected suggestion, got {other:?}"))),
        }
    }

    /// Advance CC2's mirror: the move just played, then the next revealed piece.
    pub fn play(&mut self, mv: &TbpMove, next_piece: &str) -> io::Result<()> {
        self.send(json!({
            "type": "play",
            "move": {
                "location": {
                    "type": mv.location.piece,
                    "orientation": mv.location.orientation,
                    "x": mv.location.x,
                    "y": mv.location.y,
                },
                "spin": mv.spin,
            },
        }))?;
        self.send(json!({ "type": "new_piece", "piece": next_piece }))
    }

    /// Tell CC2 to stop thinking about the current game (between games).
    pub fn stop(&mut self) -> io::Result<()> {
        self.send(json!({ "type": "stop" }))
    }
}

impl Drop for Cc2 {
    fn drop(&mut self) {
        let _ = self.send(json!({ "type": "quit" }));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn protocol_err(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}
