<div align="center">
  <img src="./public/reins_logo.png" alt="Reins Logo" width="360"/>
  <h1>🪢 Reins</h1>
  <p><strong>Security controls for AI agents.</strong></p>

  <p>
    <a href="https://github.com/pegasi-ai/reins">github.com/pegasi-ai/reins</a>
  </p>

  <p>
    <a href="https://www.npmjs.com/package/clawreins"><img src="https://img.shields.io/npm/v/clawreins.svg" alt="npm"></a>
    <a href="https://www.npmjs.com/package/clawreins"><img src="https://img.shields.io/npm/dt/clawreins.svg" alt="downloads"></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
    <a href="http://www.typescriptlang.org/"><img src="https://img.shields.io/badge/%3C%2F%3E-TypeScript-%23007ACC.svg" alt="TypeScript"></a>
    <img src="https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen" alt="Node.js >= 18.0.0">
  </p>
</div>

> In Greek myth, Athena gave Bellerophon the golden bridle — reins included — that let him guide Pegasus. Reins applies the same idea to AI agents: raw power is not enough — what matters is making it controllable.

Reins checks agent tool calls against configurable security policies, scans agent configs, and tracks configuration drift. Integrations include Claude Code hooks and OpenClaw. Custom tool adapters can call the policy engine directly.

## Quickstart

```bash
npm install -g @pegasi-ai/reins
reins init
```

### Try a policy check without an agent or API key

From a clone of this repository, using Node.js 22.12 or newer:

```bash
npm ci --ignore-scripts
npm run demo:karotte:policy
```

Watch Reins deny an outside recipient, allow the authorized team, and verify the saved simulated delivery. To run the same scenario through a real RL environment, install uv, start Docker, and run `npm run demo:karotte`.

### Reins + Karotte

Reins controls a scoped publishing tool; [Karotte](https://github.com/preferencemodel/karotte) runs the task and grades its protected delivery record. The keyless replay verifies both an enforced denial and authorized completion.

![Reins and Karotte scripted regression](./public/reins-karotte-demo.gif)

This GIF presents a recorded scripted regression. Delivery is simulated; no real email is sent. See the [quickstart, implementation, and boundaries](./examples/karotte/README.md).

## Claude Code Skill

Install the Reins skill to give Claude Code awareness of your security posture:

```bash
mkdir -p ~/.claude/skills/reins
curl -o ~/.claude/skills/reins/SKILL.md \
  https://raw.githubusercontent.com/pegasi-ai/reins/main/.claude/skills/reins/SKILL.md
```

Or clone the repo — the skill is included at `.claude/skills/reins/` automatically.

## Demo

![Reins demo](./public/reins-demo.gif)

An OpenClaw agent tries to bulk-delete 4,382 Gmail messages. Reins blocks it before execution.

## What Reins does

- **Prevent** — Block destructive actions before execution. Score irreversibility. Detect risky browser state.
- **Pause** — Route high-impact actions through terminal or messaging approval flows. Require explicit `CONFIRM-*` tokens for catastrophic operations.
- **Record** - Keep local decision records for inspection and debugging.

## Security scope

Controls apply to actions routed through a configured integration. Review each adapter's error handling, tool coverage, policy defaults, and approval behavior before relying on it. An alternate tool or unrestricted shell can bypass a scoped adapter.

Local JSONL records are editable by users with filesystem access. Logging alone does not prove execution or evidence completeness. The Karotte example separately records policy decisions and simulated deliveries in protected task state, then checks both during grading.

## Documentation

Full docs at [reins.sh/docs](https://reins.sh/docs):

- [Getting Started](https://reins.sh/docs/getting-started)
- [How It Works](https://reins.sh/docs/how-it-works)
- [Security Policies](https://reins.sh/docs/policies)
- [CLI Reference](https://reins.sh/docs/cli)
- [Security Scan](https://reins.sh/docs/scan)
- [Reins Cloud](https://reins.sh/docs/reins-cloud)
- [Use as a Library](https://reins.sh/docs/library)
- [Architecture](https://reins.sh/docs/architecture)

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
