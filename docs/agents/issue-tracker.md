# Issue tracker

This repo uses a **local-markdown tracker** for agent-driven planning efforts.

## Wayfinding operations

- **Map**: `wayfinder/<effort>/map.md`, frontmatter `labels: [wayfinder:map]`. The map is an index — one line per closed ticket, detail lives in the ticket.
- **Tickets**: `wayfinder/<effort>/tickets/<slug>.md`, children of the map. Frontmatter:

  ```yaml
  id: T01                # stable id, referenced by blocked-by
  title: <name>          # refer to tickets by this name in prose
  labels: [wayfinder:research|prototype|grilling|task]
  status: open | closed
  assignee:              # empty = unclaimed; set BEFORE working = the claim
  blocked-by: [T02, T03] # native blocking; unblocked when all listed ids are closed
  ```

- **Claim**: set `assignee:` to your session/dev name before any work.
- **Resolve**: append an `## Resolution` section (the answer), set `status: closed`, add the one-line pointer to the map's *Decisions so far*.
- **Frontier query** (open + unclaimed + unblocked):

  ```sh
  grep -l 'status: open' wayfinder/*/tickets/*.md | xargs grep -L 'assignee: .'
  # then drop any whose blocked-by lists an id still open
  ```
