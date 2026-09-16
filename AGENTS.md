# CIDenoise working instructions

- Use the `cidenoise` Conda environment explicitly. On this workstation its interpreter
  is `C:\Users\p000881\AppData\Local\miniconda3\envs\cidenoise\python.exe`.
- Keep `localdata` read-only. Never commit data, downloaded model weights or reports.
- Run `test.cmd` for relevant code changes. Actual checkpoint smoke tests are separate.
- Model adapters must work offline and must not silently substitute another checkpoint.
- Preserve code/model notices in THIRD_PARTY.md and the vendored license files.
- Docker builds and publication are explicit maintainer operations. Do not build or
  publish in response to unrelated code edits. The initial implementation task includes
  the requested container validation; subsequent edits need their own build instruction.
