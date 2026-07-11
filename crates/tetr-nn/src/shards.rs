//! Training shards: one row per played decision, **storing what was served**.
//!
//! A row is the *encoded observation the net actually consumed* for the state
//! the mover chose (packed plane + feature f32s) plus the game outcome
//! backfilled at game end. A trainer reads these bytes; it never re-derives an
//! input, so training and serving cannot disagree about what an input was.
//!
//! Layout per shard (safetensors, one file `shard-NNNNN.safetensors`):
//!
//! | tensor | shape | dtype | |
//! |---|---|---|---|
//! | `decision`  | `[d, 6]`  | i32 | `game_id, seat, ply, z, end_reason, plies_total` |
//! | `own`       | `[d, 50]` | u8  | packed plane of the played (post-placement) state |
//! | `feats`     | `[d, 70]` | f32 | served feature vector of that state |
//! | `alt_own`   | `[d, 50]` | u8  | a random NON-best sibling's post-placement plane |
//! | `alt_feats` | `[d, 70]` | f32 | that sibling's feature vector |
//! | `has_alt`   | `[d]`     | u8  | 1 when a sibling existed (row had ≥2 placements) |
//!
//! The alt row carries the implicit label "the search preferred `own` over
//! `alt_own`" — pairwise, unit-free ranking supervision (outcome-only labels
//! measurably cannot rank sibling placements; absolute search scores carry
//! generator-specific units that once poisoned a training run).
//!
//! Game-aligned flushes (a game's rows never span shards — so a shard-level
//! train/holdout split is a game-level split), atomic writes (tmp + rename — a
//! torn shard cannot exist under its final name), and an FNV payload checksum
//! in the metadata, verified on read.
//!
//! (The previous schema also stored every counterfactual sibling child, the
//! search's backed-up scores, parent observations, and action-slot ids — ~60×
//! the bytes, and the channel for three separate campaign-voiding defects.
//! Outcome-supervised value learning needs none of it.)

use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use crate::obs::{FEATURE_LEN, Obs, PACKED_PLANE, fnv1a, pack_plane};

/// Shard schema tag written to (and required from) every shard file.
pub const SHARD_SCHEMA: &str = "3";

/// One played decision (the `decision` tensor row + its observation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecisionMeta {
    /// Global game id (`seed_start + i` — workers partition this space).
    pub game_id: u32,
    /// Seat 0/1 within the game.
    pub seat: u8,
    /// Ply index within the game (per seat).
    pub ply: u16,
    /// Game outcome for this seat, backfilled at game end (`−1`/`0`/`+1`).
    pub z: i8,
    /// How the game ended (a venue `EndReason` as u8), backfilled.
    pub end_reason: u8,
    /// Total plies the game ran, backfilled.
    pub plies_total: u16,
}

/// One row: meta + the served bytes of the state the mover chose, plus a
/// randomly-chosen non-best sibling for ranking supervision.
pub struct DecisionRecord {
    pub meta: DecisionMeta,
    pub own: [u8; PACKED_PLANE],
    pub feats: [f32; FEATURE_LEN],
    pub alt_own: [u8; PACKED_PLANE],
    pub alt_feats: [f32; FEATURE_LEN],
    pub has_alt: bool,
}

impl DecisionRecord {
    /// Build from the served observations: the played state, and (when the
    /// decision had ≥2 placements) one non-best sibling.
    pub fn from_served(meta: DecisionMeta, obs: &Obs, alt: Option<&Obs>) -> Self {
        Self {
            meta,
            own: pack_plane(&obs.board),
            feats: obs.features,
            alt_own: alt
                .map(|a| pack_plane(&a.board))
                .unwrap_or([0; PACKED_PLANE]),
            alt_feats: alt.map(|a| a.features).unwrap_or([0.0; FEATURE_LEN]),
            has_alt: alt.is_some(),
        }
    }
}

/// Writes shards of decisions with game-aligned, atomic flushes.
pub struct ShardWriter {
    dir: PathBuf,
    /// Flush once at least this many decisions are buffered (at game ends).
    shard_size: usize,
    /// The in-flight game (z not yet known).
    game: Vec<DecisionRecord>,
    /// Completed games awaiting flush.
    buf: Vec<DecisionRecord>,
    next_shard: usize,
    total: usize,
}

impl ShardWriter {
    /// Open a dir for writing; numbering continues after any shards already
    /// present (a resumed run appends — completed shards are immutable).
    pub fn create(dir: impl AsRef<Path>, shard_size: usize) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let next_shard = shard_paths(&dir)?
            .last()
            .and_then(|p| {
                p.file_stem()?
                    .to_str()?
                    .strip_prefix("shard-")?
                    .parse::<usize>()
                    .ok()
            })
            .map_or(0, |i| i + 1);
        Ok(Self {
            dir,
            shard_size: shard_size.max(1),
            game: Vec::new(),
            buf: Vec::new(),
            next_shard,
            total: 0,
        })
    }

    /// Buffer one decision of the in-flight game (outcome unknown yet).
    pub fn push(&mut self, record: DecisionRecord) {
        self.game.push(record);
    }

    /// Seal the in-flight game: backfill each buffered row's outcome via
    /// `z_for_seat`, move the game into the flush buffer, and flush a shard if
    /// enough whole games are buffered. Rows only ever reach disk through
    /// here, so a shard can never hold a partial game.
    pub fn finish_game(
        &mut self,
        z_for_seat: impl Fn(u8) -> i8,
        end_reason: u8,
        plies_total: u16,
    ) -> io::Result<()> {
        for r in &mut self.game {
            r.meta.z = z_for_seat(r.meta.seat);
            r.meta.end_reason = end_reason;
            r.meta.plies_total = plies_total;
        }
        self.buf.append(&mut self.game);
        if self.buf.len() >= self.shard_size {
            self.flush()?;
        }
        Ok(())
    }

    /// Write the buffered whole games as one shard (atomically), if any.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let d = self.buf.len();
        let mut decision = Vec::with_capacity(d * 6);
        let mut own = Vec::with_capacity(d * PACKED_PLANE);
        let mut feats: Vec<f32> = Vec::with_capacity(d * FEATURE_LEN);
        let mut alt_own = Vec::with_capacity(d * PACKED_PLANE);
        let mut alt_feats: Vec<f32> = Vec::with_capacity(d * FEATURE_LEN);
        let mut has_alt: Vec<u8> = Vec::with_capacity(d);
        for r in &self.buf {
            let m = &r.meta;
            decision.extend_from_slice(&[
                m.game_id as i32,
                m.seat as i32,
                m.ply as i32,
                m.z as i32,
                m.end_reason as i32,
                m.plies_total as i32,
            ]);
            own.extend_from_slice(&r.own);
            feats.extend_from_slice(&r.feats);
            alt_own.extend_from_slice(&r.alt_own);
            alt_feats.extend_from_slice(&r.alt_feats);
            has_alt.push(r.has_alt as u8);
        }

        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let tensors: Vec<(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)> = vec![
            (
                "decision",
                safetensors::Dtype::I32,
                vec![d, 6],
                i32_bytes(&decision),
            ),
            ("own", safetensors::Dtype::U8, vec![d, PACKED_PLANE], own),
            (
                "feats",
                safetensors::Dtype::F32,
                vec![d, FEATURE_LEN],
                f32_bytes(&feats),
            ),
            (
                "alt_own",
                safetensors::Dtype::U8,
                vec![d, PACKED_PLANE],
                alt_own,
            ),
            (
                "alt_feats",
                safetensors::Dtype::F32,
                vec![d, FEATURE_LEN],
                f32_bytes(&alt_feats),
            ),
            ("has_alt", safetensors::Dtype::U8, vec![d], has_alt),
        ];
        let checksum = {
            let mut h = 0u64;
            for (_, _, _, bytes) in &tensors {
                h ^= fnv1a(bytes);
            }
            h
        };
        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    *name,
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)
                        .expect("shapes match byte lengths by construction"),
                )
            })
            .collect();
        let meta = std::collections::HashMap::from([
            ("schema".to_string(), SHARD_SCHEMA.to_string()),
            ("checksum".to_string(), format!("{checksum:016x}")),
        ]);

        // Atomic: serialize to a tmp name, fsync, rename. A torn write can
        // only ever leave a *.tmp corpse, never a readable-looking shard.
        let stem = format!("shard-{:05}", self.next_shard);
        let tmp = self.dir.join(format!("{stem}.tmp"));
        let final_path = self.dir.join(format!("{stem}.safetensors"));
        let data = safetensors::serialize(views, &Some(meta))
            .map_err(|e| io::Error::other(format!("serialize: {e}")))?;
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&data)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &final_path)?;

        self.next_shard += 1;
        self.total += d;
        self.buf.clear();
        Ok(())
    }

    /// Flush the trailing partial shard; returns total decisions written.
    pub fn finish(mut self) -> io::Result<usize> {
        self.flush()?;
        Ok(self.total)
    }
}

/// Sorted `shard-*.safetensors` paths in a dataset dir.
pub fn shard_paths(dir: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = match std::fs::read_dir(dir.as_ref()) {
        Ok(rd) => rd
            .filter_map(|e| {
                let p = e.ok()?.path();
                let name = p.file_name()?.to_str()?;
                (name.starts_with("shard-") && name.ends_with(".safetensors")).then_some(p)
            })
            .collect(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(e),
    };
    out.sort();
    Ok(out)
}

/// One shard read back: schema- and checksum-verified.
#[derive(Debug)]
pub struct Shard {
    pub decisions: Vec<DecisionMeta>,
    pub own: Vec<[u8; PACKED_PLANE]>,
    pub feats: Vec<[f32; FEATURE_LEN]>,
    pub alt_own: Vec<[u8; PACKED_PLANE]>,
    pub alt_feats: Vec<[f32; FEATURE_LEN]>,
    pub has_alt: Vec<u8>,
}

impl Shard {
    /// Read + verify one shard file.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let buf = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&buf)
            .map_err(|e| io::Error::other(format!("{}: {e}", path.display())))?;
        let raw = |name: &str| -> io::Result<&[u8]> {
            Ok(st
                .tensor(name)
                .map_err(|e| io::Error::other(format!("{}: {name}: {e}", path.display())))?
                .data())
        };

        // Schema + checksum first: an old-schema or corrupt shard fails loudly
        // here, not as garbage labels or NaN loss.
        let (_, meta) = safetensors::SafeTensors::read_metadata(&buf)
            .map_err(|e| io::Error::other(format!("{}: {e}", path.display())))?;
        let get = |k: &str| meta.metadata().as_ref().and_then(|m| m.get(k).cloned());
        match get("schema") {
            Some(s) if s == SHARD_SCHEMA => {}
            other => {
                return Err(io::Error::other(format!(
                    "{}: shard schema {other:?} != {SHARD_SCHEMA:?} (legacy corpora are not loadable)",
                    path.display()
                )));
            }
        }
        let stored = get("checksum")
            .ok_or_else(|| io::Error::other(format!("{}: no checksum", path.display())))?;
        let mut h = 0u64;
        for name in [
            "decision",
            "own",
            "feats",
            "alt_own",
            "alt_feats",
            "has_alt",
        ] {
            h ^= fnv1a(raw(name)?);
        }
        if format!("{h:016x}") != stored {
            return Err(io::Error::other(format!(
                "{}: checksum mismatch (stored {stored}, computed {h:016x})",
                path.display()
            )));
        }

        let i32s = |b: &[u8]| -> Vec<i32> {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let decisions = i32s(raw("decision")?)
            .chunks_exact(6)
            .map(|r| DecisionMeta {
                game_id: r[0] as u32,
                seat: r[1] as u8,
                ply: r[2] as u16,
                z: r[3] as i8,
                end_reason: r[4] as u8,
                plies_total: r[5] as u16,
            })
            .collect();
        let planes = |b: &[u8]| -> Vec<[u8; PACKED_PLANE]> {
            b.chunks_exact(PACKED_PLANE)
                .map(|c| c.try_into().expect("chunk size"))
                .collect()
        };
        let feat_rows = |b: &[u8]| -> Vec<[f32; FEATURE_LEN]> {
            b.chunks_exact(4 * FEATURE_LEN)
                .map(|row| {
                    let mut f = [0.0f32; FEATURE_LEN];
                    for (i, c) in row.chunks_exact(4).enumerate() {
                        f[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                    f
                })
                .collect()
        };
        Ok(Self {
            decisions,
            own: planes(raw("own")?),
            feats: feat_rows(raw("feats")?),
            alt_own: planes(raw("alt_own")?),
            alt_feats: feat_rows(raw("alt_feats")?),
            has_alt: raw("has_alt")?.to_vec(),
        })
    }

    /// Game ids present in this shard (whole games, by the game-aligned flush).
    pub fn game_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.decisions.iter().map(|d| d.game_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::BOARD_LEN;

    fn obs_with(bit: usize, feat: f32) -> Obs {
        let mut board = [0.0f32; BOARD_LEN];
        board[bit % BOARD_LEN] = 1.0;
        let mut features = [0.0f32; FEATURE_LEN];
        features[bit % FEATURE_LEN] = feat;
        Obs { board, features }
    }

    fn record(gid: u32, seat: u8, ply: u16) -> DecisionRecord {
        DecisionRecord::from_served(
            DecisionMeta {
                game_id: gid,
                seat,
                ply,
                ..Default::default()
            },
            &obs_with(gid as usize + ply as usize, ply as f32 + 0.5),
            Some(&obs_with(gid as usize + ply as usize + 13, ply as f32)),
        )
    }

    #[test]
    fn roundtrip_preserves_served_bytes_exactly() {
        let dir = std::env::temp_dir().join(format!("tetr-nn-shards-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = ShardWriter::create(&dir, 1000).unwrap();
        for (gid, plies) in [(5u32, 3u16), (6, 2)] {
            for ply in 0..plies {
                w.push(record(gid, (ply % 2) as u8, ply));
            }
            w.finish_game(|seat| if seat == 0 { 1 } else { -1 }, 0, plies)
                .unwrap();
        }
        assert_eq!(w.finish().unwrap(), 5);

        let paths = shard_paths(&dir).unwrap();
        assert_eq!(paths.len(), 1);
        let shard = Shard::read(&paths[0]).unwrap();
        assert_eq!(shard.decisions.len(), 5);
        // z backfilled per seat.
        assert_eq!(shard.decisions[0].z, 1);
        assert_eq!(shard.decisions[1].z, -1);
        // Served bytes exact: re-derive one row and compare.
        let expect = record(5, 0, 0);
        assert_eq!(shard.own[0], expect.own);
        assert_eq!(shard.feats[0], expect.feats);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flushes_are_game_aligned_and_numbering_resumes() {
        let dir = std::env::temp_dir().join(format!("tetr-nn-align-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = ShardWriter::create(&dir, 4).unwrap();
        // Game 1: 3 decisions (below threshold — no flush). Game 2: 3 more
        // (crosses threshold at a game END — flush of BOTH whole games).
        for gid in [1u32, 2] {
            for ply in 0..3u16 {
                w.push(record(gid, 0, ply));
            }
            w.finish_game(|_| 1, 0, 3).unwrap();
        }
        // Game 3 stays in the buffer until finish().
        w.push(record(3, 0, 0));
        w.finish_game(|_| -1, 1, 1).unwrap();
        w.finish().unwrap();

        let paths = shard_paths(&dir).unwrap();
        assert_eq!(paths.len(), 2);
        let (a, b) = (
            Shard::read(&paths[0]).unwrap(),
            Shard::read(&paths[1]).unwrap(),
        );
        let gids = |s: &Shard| {
            let mut v: Vec<u32> = s.game_ids().collect();
            v.dedup();
            v
        };
        assert_eq!(gids(&a), vec![1, 2]);
        assert_eq!(gids(&b), vec![3]);

        // A new writer continues numbering (resume appends, never clobbers).
        let w2 = ShardWriter::create(&dir, 4).unwrap();
        assert_eq!(w2.next_shard, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corruption_fails_the_checksum_loudly() {
        let dir = std::env::temp_dir().join(format!("tetr-nn-sum-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = ShardWriter::create(&dir, 1).unwrap();
        w.push(record(9, 0, 0));
        w.finish_game(|_| 1, 0, 1).unwrap();
        w.finish().unwrap();

        let path = &shard_paths(&dir).unwrap()[0];
        let mut bytes = std::fs::read(path).unwrap();
        let n = bytes.len();
        bytes[n - 3] ^= 0xff; // flip payload bits near the tail
        std::fs::write(path, bytes).unwrap();
        let err = Shard::read(path).unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
        // And no tmp corpse is ever left behind by a completed writer.
        assert!(shard_paths(&dir).unwrap().len() == 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
