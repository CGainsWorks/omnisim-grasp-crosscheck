# Security review of OmniSim v9.2.4

OmniSim was cold outreach: an email asked for a test and a star. Running
someone else's simulator means running their code, so it was read before
anything was executed. This is what was checked, what was found and how it is
run here. It is a source review of one commit, not a guarantee about any other
version.

**Reviewed:** `omnilink-tech/omnisim` tag `v9.2.4`, commit
`28e00e1f1a2d7932743911a697dc8f63ed5eb81a` (12,404 files) and the PyPI wheel
`omnilink==0.6.3`, which the runtime image installs and which is not in the
repository.

**Verdict:** nothing malicious found. There are two auto-run surfaces worth
knowing about and a few ordinary risks. None of them is reachable in the way
this repository runs OmniSim.

## What was checked

| Check | Result |
| --- | --- |
| Committed executables (`.exe`, `.dll`, `.so`, `.pyd`, `.whl`, `.jar`) | none |
| Obfuscation and staged execution (`FromBase64String`, `-EncodedCommand`, `Invoke-Expression`, `certutil`, `mshta`, `rundll32`, `DownloadString`) | none in code |
| `curl \| sh` or `wget \| sh` | none |
| Persistence (Run keys, scheduled tasks, startup folders, crontab, LaunchAgents) | none |
| Credential access (`.ssh`, browser profiles, `.git-credentials`, cloud credential files) | none |
| API keys read from the environment | LLM provider keys and `GITHUB_TOKEN`, each only by an optional benchmark, critique or release script that sends it to its own provider |
| Install hooks (`preinstall`/`postinstall`, custom build backends) | none |
| Outbound hosts in code | GitHub (asset URLs), the vendor's own `omnilink-agents.com` (chat agent, key check), an LLM provider's API (optional benchmark); everything else is `127.0.0.1` |
| Dynamic code execution | `importlib` of a sibling file in `omnisim/conformance/fingerprint.py`, reading package versions; benign |
| `omnilink==0.6.3` wheel | talks to `127.0.0.1` and `omnilink-agents.com` only; one `subprocess` call, to `openssl`, to make a localhost certificate; file tools confined to a root directory |

## Things worth knowing

- **`.mcp.json` at the repository root.** Opening the OmniSim checkout in an MCP
  client offers to start a local MCP server
  (`python -m omnisim_mcp`). It is a stdlib proxy to `127.0.0.1:6789`, so not
  harmful but it is code that runs because a folder was opened.
- **Git hooks.** `.githooks/` runs Python on checkout and push but only after
  `scripts/dev/setup_hooks.sh` sets `core.hooksPath`. Nothing sets it implicitly.
- **Agent instruction files (`AGENTS.md` and a one-line file importing it).** Instructions written for coding agents. They
  were read as documentation. No text in them asks an agent to do anything
  outside the project, such as starring, pushing or sending data.
- **589 committed `.pt` checkpoints.** PyTorch checkpoints are pickles and
  several training scripts call `torch.load` without `weights_only=True`, one
  explicitly with `weights_only=False`. Loading an untrusted checkpoint can
  execute code. Nothing here loads them.
- **The chat agent exposes local files by design.** OmniLink agents can call
  file tools on the host. Its path guard is a string prefix test, so `./data`
  would also admit `./data2`. That is a weakness, not malice and it is not
  used here.

## How it is run here

- **Built from source, not downloaded.** `scripts/build_omnisim_image.sh` clones
  the tag, refuses to build unless the commit matches the hash above and builds
  the upstream `docker/Dockerfile.runtime`. The 663 MB Windows installer and the
  published GHCR image were not used, because a binary built elsewhere cannot be
  tied to the source that was read.
- **No network.** Every simulation runs under `docker run --network none`, so
  nothing in the image can reach the vendor or anyone else during a run.
- **Read-only inputs.** The repository is mounted read only. Results come back
  through a named volume and `docker cp`.
- **No account, no key.** The OmniLink agent, its OmniKey and the hosted chat
  are not used. OmniSim runs as a plain simulator driven by a Python controller.

The image build itself does reach the network: it installs Ubuntu packages, the
pinned physics stack from PyPI (`newton`, `warp-lang`, `mujoco`,
`mujoco-warp`, `usd-core`, `omnilink` and their dependencies) and clones
`glm` and `stb` from GitHub. Those are trusted to the same degree as any `pip
install`.
