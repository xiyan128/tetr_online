//! Decision shards: training data that **stores what was served**.
//!
//! A shard row is the *encoded observation the net actually consumed* — the
//! packed own plane and the f32 feature vector per child, the packed opponent
//! plane per decision — plus the search's integer outputs (per-child root
//! scores) and the game outcome backfilled at game end. A trainer reads these
//! bytes; it never re-derives an input, so training and serving cannot
//! disagree about what an input was. (The reference design stored raw state
//! components and re-encoded them in Python; an entire class of its bugs —
//! truncated pending lists, mirrored encoders drifting — does not exist here.)
//!
//! Layout per shard (safetensors, one file `shard-NNNNN.safetensors`):
//!
//! | tensor | shape | dtype | |
//! |---|---|---|---|
//! | `decision`     | `[d, 8]`  | i32 | `game_id, seat, ply, played, argmax, z, end_reason, plies_total` |
//! | `opp_plane`    | `[d, 50]` | u8  | packed opponent plane (per decision) |
//! | `child_offset` | `[d + 1]` | i32 | ragged prefix offsets into the child axis |
//! | `child_own`    | `[c, 50]` | u8  | packed own plane (per child) |
//! | `child_feats`  | `[c, 85]` | f32 | served feature vector (per child) |
//! | `child_score`  | `[c]`     | i32 | the search's backed-up root score |
//!
//! Ragged children (no fixed-width padding), game-aligned flushes (a game's
//! rows never span shards), atomic writes (tmp + rename — a torn shard cannot
//! exist under its final name), and an FNV payload checksum in the metadata,
//! verified on read.

use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use crate::obs::{FEATURE_LEN, Obs, PACKED_PLANE, fnv1a, pack_plane};

/// One decision's fixed-width record (the `decision` tensor row).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecisionMeta {
    /// Global game id (`seed_start + i` — workers partition this space).
    pub game_id: u32,
    /// Seat 0/1 within the game.
    pub seat: u8,
    /// Ply index within the game (per seat).
    pub ply: u16,
    /// Index of the child that was PLAYED (π-sampled).
    pub played: u16,
    /// Index of the argmax child (what greedy-on-scores would play).
    pub argmax: u16,
    /// Game outcome for this seat, backfilled at game end (`−1`/`0`/`+1`).
    pub z: i8,
    /// How the game ended (a venue `EndReason` as u8), backfilled.
    pub end_reason: u8,
    /// Total plies the game ran, backfilled.
    pub plies_total: u16,
}

/// One decision: its meta, the frozen opponent plane, and every sibling child
/// as served bytes + search score.
pub struct DecisionRecord {
    pub meta: DecisionMeta,
    /// Packed opponent plane (from the decision's [`Obs`]).
    pub opp_plane: [u8; PACKED_PLANE],
    /// Per child: packed own plane, served features, backed-up root score,
    /// action slot ([`crate::obs::placement_slot`]).
    pub children: Vec<([u8; PACKED_PLANE], [f32; FEATURE_LEN], i32, u8)>,
}

impl DecisionRecord {
    /// Build from served observations (one per child, sharing the decision's
    /// opponent plane) + the search's per-child scores.
    pub fn from_served(meta: DecisionMeta, children: &[(&Obs, i32, u8)]) -> Self {
        let opp_plane = children
            .first()
            .map(|(o, _, _)| pack_plane(&o.opp_board))
            .unwrap_or([0; PACKED_PLANE]);
        Self {
            meta,
            opp_plane,
            children: children
                .iter()
                .map(|(o, score, slot)| (pack_plane(&o.own_board), o.features, *score, *slot))
                .collect(),
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
        let c: usize = self.buf.iter().map(|r| r.children.len()).sum();

        let mut decision = Vec::with_capacity(d * 8);
        let mut opp_plane = Vec::with_capacity(d * PACKED_PLANE);
        let mut child_offset: Vec<i32> = Vec::with_capacity(d + 1);
        let mut child_own = Vec::with_capacity(c * PACKED_PLANE);
        let mut child_feats: Vec<f32> = Vec::with_capacity(c * FEATURE_LEN);
        let mut child_score: Vec<i32> = Vec::with_capacity(c);
        let mut child_slot: Vec<u8> = Vec::with_capacity(c);
        child_offset.push(0);
        for r in &self.buf {
            let m = &r.meta;
            decision.extend_from_slice(&[
                m.game_id as i32,
                m.seat as i32,
                m.ply as i32,
                m.played as i32,
                m.argmax as i32,
                m.z as i32,
                m.end_reason as i32,
                m.plies_total as i32,
            ]);
            opp_plane.extend_from_slice(&r.opp_plane);
            for (own, feats, score, slot) in &r.children {
                child_own.extend_from_slice(own);
                child_feats.extend_from_slice(feats);
                child_score.push(*score);
                child_slot.push(*slot);
            }
            child_offset.push(child_score.len() as i32);
        }

        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let tensors: Vec<(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)> = vec![
            (
                "decision",
                safetensors::Dtype::I32,
                vec![d, 8],
                i32_bytes(&decision),
            ),
            (
                "opp_plane",
                safetensors::Dtype::U8,
                vec![d, PACKED_PLANE],
                opp_plane,
            ),
            (
                "child_offset",
                safetensors::Dtype::I32,
                vec![d + 1],
                i32_bytes(&child_offset),
            ),
            (
                "child_own",
                safetensors::Dtype::U8,
                vec![c, PACKED_PLANE],
                child_own,
            ),
            (
                "child_feats",
                safetensors::Dtype::F32,
                vec![c, FEATURE_LEN],
                f32_bytes(&child_feats),
            ),
            (
                "child_score",
                safetensors::Dtype::I32,
                vec![c],
                i32_bytes(&child_score),
            ),
            ("child_slot", safetensors::Dtype::U8, vec![c], child_slot),
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
            ("schema".to_string(), "1".to_string()),
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

/// One shard read back: checksum-verified, ragged accessors.
#[derive(Debug)]
pub struct Shard {
    pub decisions: Vec<DecisionMeta>,
    pub opp_planes: Vec<[u8; PACKED_PLANE]>,
    child_offset: Vec<i32>,
    pub child_own: Vec<[u8; PACKED_PLANE]>,
    pub child_feats: Vec<[f32; FEATURE_LEN]>,
    pub child_scores: Vec<i32>,
    pub child_slots: Vec<u8>,
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

        // Checksum first: a corrupt shard fails loudly here, not as NaN loss.
        let (_, meta) = safetensors::SafeTensors::read_metadata(&buf)
            .map_err(|e| io::Error::other(format!("{}: {e}", path.display())))?;
        let stored = meta
            .metadata()
            .as_ref()
            .and_then(|m| m.get("checksum").cloned())
            .ok_or_else(|| io::Error::other(format!("{}: no checksum", path.display())))?;
        let mut h = 0u64;
        for name in [
            "decision",
            "opp_plane",
            "child_offset",
            "child_own",
            "child_feats",
            "child_score",
            "child_slot",
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
        let dec = i32s(raw("decision")?);
        let decisions = dec
            .chunks_exact(8)
            .map(|r| DecisionMeta {
                game_id: r[0] as u32,
                seat: r[1] as u8,
                ply: r[2] as u16,
                played: r[3] as u16,
                argmax: r[4] as u16,
                z: r[5] as i8,
                end_reason: r[6] as u8,
                plies_total: r[7] as u16,
            })
            .collect();
        let planes = |b: &[u8]| -> Vec<[u8; PACKED_PLANE]> {
            b.chunks_exact(PACKED_PLANE)
                .map(|c| c.try_into().expect("chunk size"))
                .collect()
        };
        let child_feats = raw("child_feats")?
            .chunks_exact(4 * FEATURE_LEN)
            .map(|row| {
                let mut f = [0.0f32; FEATURE_LEN];
                for (i, c) in row.chunks_exact(4).enumerate() {
                    f[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                f
            })
            .collect();
        Ok(Self {
            decisions,
            opp_planes: planes(raw("opp_plane")?),
            child_offset: i32s(raw("child_offset")?),
            child_own: planes(raw("child_own")?),
            child_feats,
            child_scores: i32s(raw("child_score")?),
            child_slots: raw("child_slot")?.to_vec(),
        })
    }

    /// The child index range of decision `d`.
    pub fn children_of(&self, d: usize) -> std::ops::Range<usize> {
        self.child_offset[d] as usize..self.child_offset[d + 1] as usize
    }

    /// Game ids present in this shard (whole games, by the game-aligned flush).
    pub fn game_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.decisions.iter().map(|d| d.game_id)
    }
}

/// The game ids durably recorded across a dataset dir's shards — the resume
/// done-set half that lives in shard files (games are shard-atomic, so
/// presence means complete).
pub fn recorded_game_ids(dir: impl AsRef<Path>) -> io::Result<std::collections::HashSet<u32>> {
    let mut gids = std::collections::HashSet::new();
    for p in shard_paths(dir)? {
        gids.extend(Shard::read(&p)?.game_ids());
    }
    Ok(gids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::BOARD_LEN;

    fn obs_with(bit: usize, feat: f32) -> Obs {
        let mut own = [0.0f32; BOARD_LEN];
        own[bit % BOARD_LEN] = 1.0;
        let mut opp = [0.0f32; BOARD_LEN];
        opp[(bit * 7) % BOARD_LEN] = 1.0;
        let mut features = [0.0f32; FEATURE_LEN];
        features[bit % FEATURE_LEN] = feat;
        Obs {
            own_board: own,
            opp_board: opp,
            features,
        }
    }

    fn record(gid: u32, seat: u8, ply: u16, n_children: usize) -> DecisionRecord {
        let children: Vec<(Obs, i32)> = (0..n_children)
            .map(|c| (obs_with(gid as usize + c, c as f32 + 0.5), c as i32 * 100))
            .collect();
        let refs: Vec<(&Obs, i32, u8)> = children
            .iter()
            .enumerate()
            .map(|(c, (o, s))| (o, *s, c as u8))
            .collect();
        DecisionRecord::from_served(
            DecisionMeta {
                game_id: gid,
                seat,
                ply,
                played: 0,
                argmax: (n_children - 1) as u16,
                ..Default::default()
            },
            &refs,
        )
    }

    #[test]
    fn roundtrip_preserves_served_bytes_exactly() {
        let dir = std::env::temp_dir().join(format!("tetr-nn-shards-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = ShardWriter::create(&dir, 1000).unwrap();
        // Two games with ragged child counts.
        for (gid, plies) in [(5u32, 3u16), (6, 2)] {
            for ply in 0..plies {
                w.push(record(gid, (ply % 2) as u8, ply, 3 + ply as usize));
            }
            w.finish_game(|seat| if seat == 0 { 1 } else { -1 }, 0, plies)
                .unwrap();
        }
        assert_eq!(w.finish().unwrap(), 5);

        let paths = shard_paths(&dir).unwrap();
        assert_eq!(paths.len(), 1);
        let shard = Shard::read(&paths[0]).unwrap();
        assert_eq!(shard.decisions.len(), 5);
        // Ragged children preserved.
        assert_eq!(shard.children_of(0).len(), 3);
        assert_eq!(shard.children_of(2).len(), 5);
        // z backfilled per seat.
        assert_eq!(shard.decisions[0].z, 1);
        assert_eq!(shard.decisions[1].z, -1);
        // Served bytes exact: re-derive one child and compare.
        let expect = record(5, 0, 0, 3);
        let r = shard.children_of(0);
        assert_eq!(shard.child_own[r.start], expect.children[0].0);
        assert_eq!(shard.child_feats[r.start], expect.children[0].1);
        assert_eq!(shard.opp_planes[0], expect.opp_plane);
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
                w.push(record(gid, 0, ply, 2));
            }
            w.finish_game(|_| 1, 0, 3).unwrap();
        }
        // Game 3 stays in the buffer until finish().
        w.push(record(3, 0, 0, 2));
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
        assert_eq!(
            recorded_game_ids(&dir).unwrap(),
            [1u32, 2, 3].into_iter().collect()
        );

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
        w.push(record(9, 0, 0, 4));
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
