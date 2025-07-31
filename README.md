# 🛡️ Pegasi Shield 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Python Version](https://img.shields.io/pypi/v/llm-guard)](https://pypi.org/project/guardrail-ml)
[![Downloads](https://static.pepy.tech/badge/guardrail-ml)](https://pepy.tech/project/guardrail-ml)
[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue)](https://icml.cc/)

![plot](./static/images/pegasi_shield.png)

*A lightweight safety and reliability layer for large‑language‑model (LLM) applications.*

---

## Overview

Pegasi Shield sits between your application and any LLM (OpenAI, Claude, local models, etc.).  
It inspects every prompt and response, blocks or edits unsafe content, and logs decisions for auditing—all with minimal latency and no data egress.

<a href="https://colab.research.google.com/drive/17Dq4ClbxI-AIjpQM2MxiLeoLnmfxTqrR?usp=sharing"
   target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg"
       alt="Open in Colab"/>
</a>
---

## 🔬 Research: FRED

Pegasi Shield’s hallucination module is powered by **FRED — Financial Retrieval‑Enhanced Detection & Editing**.
The method was peer‑reviewed and accepted to the *ICML 2025 Workshop*.
Code, evaluation harness and demo notebooks are in `fred/`.

![plot](./static/images/fred-v2.png)

<a href="https://pegasi-fred-demo-v1.streamlit.app/"
   target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Live%20Demo-Streamlit‑App-FF4B4B?logo=streamlit&logoColor=white"
       alt="Open ICML Streamlit Demo"/>
</a>

<a href="https://arxiv.org/abs/2507.20930"
   target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b"
       alt="Read the paper on arXiv">
</a>
---

## 🔧 Key capabilities

| Area | What Shield provides |
|------|----------------------|
| **Prompt security** | Detects and blocks prompt injections, role hijacking, system‑override attempts. |
| **Output sanitisation** | Removes personal data, hate speech, defamation and other policy violations. |
| **Hallucination controls** | Scores and rewrites ungrounded text using a 4B parameter model at performance on par with o3. |
| **Observability** | Emits structured traces and metrics (OpenTelemetry) for dashboards and alerts. |
| **Deployment** | Pure‑Python middleware, Docker image, or Helm chart for Kubernetes / VPC installs. |

---

## ⚡ Quick start

*Coming July 25th

```bash
pip install pegasi-shield
````

```python
from pegasi_shield import Shield
from openai import OpenAI

client = OpenAI()
shield = Shield()                       # uses default policy

messages = [{"role": "user", "content": "Tell me about OpenAI o3"}]
response = shield.chat_completion(
    lambda: client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
)

print(response.choices[0].message.content)
```

*`Shield.chat_completion` accepts a callable that runs your normal LLM request.
Shield returns the same response object—or raises `ShieldError` if the call is blocked.*

---

## 📚 How it works

1. **Prompt firewall** — lightweight rules (regex, AST, ML) followed by an optional LLM check.
2. **LLM request** — forwards the original or patched prompt to your provider.
3. **Output pipeline**

   * heuristics → vector similarity checks → policy LLM
   * optional “Hallucination Lens” rewrite if factuality score is below threshold.
4. **Trace** — JSON event with allow/block/edit decision and risk scores.

All stages are configurable via YAML or Python.

---

## Roadmap

* v0.5 launch (July 18th)
* LiveKit Agent Tutorial
* LangGraph Agent Tutorial
* Fine‑grained policy language 
* Streaming output inspection
* JavaScript/TypeScript SDK

---

## Contributing

Issues and pull requests are welcome. See `CONTRIBUTING.md` for details.

---

## License

Apache 2.0

```
::contentReference[oaicite:0]{index=0}
```
