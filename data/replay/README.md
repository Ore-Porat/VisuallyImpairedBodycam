# Replay recordings

Put recorded videos or frame folders here (gitignored) along with control scripts, e.g.
`data/replay/find_pasta.yaml`:

```yaml
fps: 10
duration: 20
list: [pasta, coffee]
events:
  - {t: 0.0, event: mode_on}
  - {t: 0.5, event: enter}
  - {t: 15.0, event: enter}
```

Run: `slr replay data/replay/find_pasta.mp4 --script data/replay/find_pasta.yaml --out timeline.json`
