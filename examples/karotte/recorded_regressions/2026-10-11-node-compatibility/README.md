# Recorded integration after the Node compatibility fix

This is the unmodified export of Docker run `reins-report-delivery-a05343553150` on code commit `58d55d76183b21453cb40ff9594a82b9c194bf76`, using the pinned Karotte and Node 22 image recorded in [evidence.json](evidence.json). The code separately passed 177 tests on Node 18.20.8 and Node 26.4.0; this Docker run is not a Node 18 run.

The real MCP tool denied `outside@example.test`, allowed `team@example.test`, and saved exactly one synthetic delivery. Native grading returned 1/1. No email was sent. This regression does not exercise channel approvals or establish model behavior.

The [run configuration](run_config.json) sets `use_fake_model: true`. The `openai/gpt-5.2` string in the [transcript](transcript.json) is a configuration label: fixed messages from `environment.fake_model` produced the two tool calls, with no model-provider request. The original October 10 live-model screen and the older recorded GIF remain separate artifacts.

Run `npm run demo:karotte` from the repository root to generate a fresh run. The source and built-file hashes in the evidence identify the package actually used. These local records support traceability; they are not independent execution attestation.
