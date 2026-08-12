# Positioning

## Where `qgemm-mx` Stands Out, and Who Wants These Skills

**Companion to** `PROJECT-v2.0-fp4-hopper.md`
**Date** 2026-08-11

> **A note on epistemic status.** Part II of this document reasons from *public evidence* — published papers, open-source commit histories, shipped products, and stated engineering priorities. It is not insider knowledge of anyone's hiring plans, and specific team needs change quarterly. Treat the org list as a map of *where this work is demonstrably done*, which is a durable signal, rather than a list of open positions, which is not.

---

# Part I · Where this project stands out

## 1. It closes a gap with a known price tag

Almost every kernel portfolio project is "I implemented a fast X." This one begins with the size of the prize already computed: **MXFP4 moves 1.88× fewer bytes per weight than FP8, the current Hopper software path delivers ~1.0×, and the missing factor is documented in a production stack.**

That single property does a surprising amount of work:

- The project has a **thesis that can be wrong**, not just an artifact that can be slow.
- The report is writable at any performance outcome — recovering 1.6× is a result, and demonstrating the gap is structurally unavoidable on Hopper is *also* a result.
- A reviewer can evaluate the claim in thirty seconds without trusting your benchmarks, because the byte arithmetic is checkable on paper.
- It is immediately legible to a practitioner: *"you are currently paying for 4-bit and getting 8-bit speed"* is a sentence any inference engineer understands and cares about.

Contrast with v1.0, whose headline was "faster than dequantizing to FP16 first" — true, unfalsifiable, and worth nothing.

## 2. The scarcest skill in the space, done properly

Hopper's datapath is genuinely hard in a way Ampere's is not. `wgmma` is asynchronous and reads B operands from shared memory under specific swizzle patterns; TMA imposes descriptor-level layout constraints; thread-block clusters add another coordination layer. **An Ampere weight layout does not satisfy any of it** — which is precisely why Machete exists as a distinct kernel rather than a port of Marlin.

The consequence for you: **almost everyone who claims CUDA on a résumé cannot write a correct `wgmma` + TMA pipeline over a sub-byte data type.** The population that can is small, in demand, and currently concentrated inside a handful of vendor and lab teams. Deriving the layout from the datapath's requirements — and being able to explain at a whiteboard *why* the nibbles are ordered `0,2,4,6,1,3,5,7` and why the swizzle is what it is — is the single most differentiating technical outcome available from this project.

It also happens to be the part that transfers. The specific format will churn; reasoning backwards from a datapath's fragment requirements to a memory layout is the durable skill, and it applies identically to FP8, FP6, NVFP4, whatever ships next, and to AMD's and Apple's equivalents.

## 3. The measurement apparatus is a contribution, not overhead

Nobody has a trustworthy public way to benchmark low-precision GEMM kernels. Yours will be L2-flushed, graph-captured, clock-logged, measured against empirical peak, structured-probe-verified, and multi-architecture.

Then **R0 points it at the existing published kernels and reports how much their numbers move.** That is a reproducibility contribution the field visibly lacks, and it has three unusual properties:

- **It cannot fail.** The result exists regardless of whether your own kernel ever gets fast.
- **Tools get adopted more readily than kernels**, because adopting a tool costs nothing while adopting a kernel means owning it.
- It makes every subsequent number you publish credible, which is worth more than any individual measurement.

Most portfolio projects treat methodology as a tax on the interesting work. Here it is the first deliverable and the most likely one to be used by strangers.

## 4. Preregistered predictions

Committing the roofline predictions to a timestamped commit before measuring is, as far as I can tell, **essentially unheard of in kernel engineering.** It costs an afternoon and it changes what the work *is*: a study that predicts its own headline result and then checks the prediction is doing something categorically different from one that measures and then explains.

It also converts being wrong from an embarrassment into a documented finding. "We predicted the crossover at `M`≈41, measured 28, and here is the profiler evidence for why we were wrong" reads as **more** competent than a clean sweep of wins — that author is visibly running an investigation, while the alternative might be running a demo.

## 5. Published negative results

§8.2's second prediction — that wave quantization and launch overhead, not dequantization instruction cost, dominate the current gap — is a direct partial challenge to the W4A8 literature's central claim. Publishing the outcome either way, including the version where you are wrong, is rare enough to be memorable and is specifically what a research-oriented reviewer looks for.

## 6. A frontier, not a benchmark

Three formats (MXFP4, NVFP4, INT4-g128) measured on **both** throughput and accuracy turns a single-format speed run into a decision-support artifact. MXFP4's power-of-two scale is a genuine speed advantage — an exponent add rather than a multiply — bought with a genuine accuracy cost. Nobody serving models can currently look up that trade-off on Hopper. After this project, they can.

## 7. Two architectures, with the divergence as the point

Reporting sm_86 and sm_90 for the same kernel and treating the difference as a finding rather than an inconsistency is unusual. The crossover `M` roughly doubles between them and the occupancy story inverts completely — 16 CTAs nearly fills a 20-SM laptop and starves a 132-SM H100. **The crossover is a function to characterize, not a constant to discover**, and demonstrating that is worth more than either number alone.

## 8. It lands somewhere real, on something nobody claimed

A stale, closed-as-not-planned bug with demonstrated user pain and zero competition is close to an ideal entry point: small enough to be a first PR, real enough to earn maintainer credibility, and directly on the path to the larger contribution rather than a detour from it.

**Adoption is the only dimension a third party validates for you and the only one you cannot manufacture.** Everything else in this project is your execution.

## 9. What it does *not* stand out on — own this

Honesty here is part of the positioning:

- **It is not a new algorithm.** Quantization-algorithm work (GPTQ, AWQ, MR-GPTQ) is a different contribution and is explicitly out of scope.
- **It is not a new abstraction.** Triton, CuTe, and TileLang are the abstraction-layer contributions; this uses them.
- **The core idea is not novel.** Fused dequantization-GEMM is well established. What is novel is the *measurement*, the *target architecture*, and the *decomposition*.

State this plainly in the report. A project that accurately describes its own contribution class is more credible than one that inflates it, and "engineering and measurement contribution, not an algorithmic one" is a perfectly respectable and clearly-stated category.

---

# Part II · Who wants these skills

## The skill inventory this project produces

| # | Skill | Scarcity | Notes |
|---|---|---|---|
| S1 | **Hopper `wgmma` + TMA async pipeline authoring** | **Very scarce** | The single most in-demand low-level GPU skill right now |
| S2 | **Sub-byte layout / swizzle derivation** | **Very scarce** | Reasoning backwards from datapath to memory layout |
| S3 | **Block-scaled format internals** (OCP MX, MXFP4, NVFP4, FP8) | Scarce, rising fast | Every vendor and serving stack is adopting these |
| S4 | CUTLASS 3.x / CuTe | Scarce | Table stakes for serious kernel roles |
| S5 | Triton | Moderately common, still valued | Breadth signal; the DSL-cost study is the differentiated part |
| S6 | Roofline / analytical performance modeling | Moderately scarce | Separates engineers from tuners |
| S7 | Nsight Compute deep profiling | Moderately scarce | Counter-level, not just timeline |
| S8 | **Benchmark methodology rigor** | **Very scarce, badly underrated** | The skill of not fooling yourself |
| S9 | MoE serving internals (grouped GEMM, routing, imbalance) | Scarce | Dominant architecture; least mature kernels |
| S10 | Upstream OSS contribution in vLLM / SGLang / torchao | Scarce, very high signal | Third-party validated |
| S11 | Technical writing that survives review | Scarce | Determines whether any of the above is visible |
| S12 | Cross-accelerator portability reasoning | Scarce | The bridge to TPU / Trainium / ROCm / Metal roles |

**The two to lean on in any conversation are S1 and S8.** S1 because the supply is genuinely thin; S8 because it is the thing senior people quietly screen for and almost nobody can demonstrate.

---

## Tier 1 · Frontier model labs

Organizations training and serving their own frontier models. Inference efficiency is directly load-bearing — it determines what they can afford to serve.

| Organization | Why this work is relevant | Skills that map |
|---|---|---|
| **Anthropic** | Serves Claude at large scale on a **heterogeneous fleet** — NVIDIA GPUs, Google TPUs, and AWS Trainium. Unusually, that makes the portability reasoning as valuable as the CUDA itself; Pallas/Mosaic (JAX TPU kernels) and the Neuron toolchain are live surfaces alongside GPU work | S1, S6, S8, **S12**, S11 |
| **OpenAI** | Triton originated here. **GPT-OSS shipped MXFP4 weights** — literally this project's subject matter, and the source of the 2880-dimension alignment problem | S1–S5, S3 especially, S9 |
| **Google DeepMind** | TPU-first, so XLA / Pallas / Mosaic rather than CUDA. The transferable part is the roofline, quantization, and tiling reasoning, not the syntax | S6, S8, S12, S3 |
| **Meta (GenAI + PyTorch)** | Llama serving, **and PyTorch itself** — `torchao` is where quantized kernels land, Inductor generates Triton, and both are open. Among the most directly addressable of all Tier 1 | S1–S5, S10 |
| **xAI** | Very large GPU fleet, serves Grok at scale | S1, S6, S8, S9 |
| **DeepSeek** | Arguably the most kernel-forward open lab: published **DeepGEMM** (FP8 grouped GEMM), DeepEP, and MLA. Their artifacts are references in your own §6.5 | S1, S4, **S9** |
| **Mistral AI** | European, serves its own models, small enough that kernel work is high-leverage per person | S1, S3, S6 |
| **Alibaba / Qwen** | Qwen3-MoE plus quantized releases | S3, S9 |
| **Moonshot (Kimi), Zhipu, MiniMax** | Large MoE models served at scale | S9, S1 |
| **Cohere** | Enterprise serving, efficiency-sensitive | S6, S8 |
| **Thinking Machines, SSI, Reflection** | Newer and small; small teams hire senior systems generalists rather than specialists | S1, S6, S8, S11 |

**How to be legible to Tier 1.** These organizations mostly do *not* hire from portfolio repositories — they hire from referrals, publications, and upstream contribution histories. The path in is D8 (the upstream PR) and D6 published as a preprint or well-circulated writeup, not the repository itself.

---

## Tier 2 · Inference providers and serving infrastructure

Where this skill is most *directly* monetized: for these companies, kernel efficiency is not an internal cost, it is the product's margin.

| Organization | Why this work is relevant | Skills that map |
|---|---|---|
| **Red Hat (formerly Neural Magic)** | **The single most on-the-nose employer for this project.** Marlin and Machete came out of this orbit; they maintain vLLM's quantization stack and `llm-compressor`. Fixing vLLM's stale 128-alignment bug is *literally doing their triage*, and D8 lands in their codebase | **S1–S4, S9, S10** — all of it |
| **Together AI** | Kernel-forward, publishes performance research, deep bench in this exact area | S1–S4, S6 |
| **Fireworks AI** | Inference speed is the entire product proposition | S1, S6, S8, S9 |
| **Baseten** | Substantial recent investment in performance engineering | S1, S6, S8 |
| **LMSYS / SGLang** | `sgl-kernel` is a direct integration target for D8; research-adjacent and open | S1, S4, S9, S10 |
| **Perplexity** | Serves at very large scale; publishes some kernel work | S6, S8, S9 |
| **Databricks / Mosaic** | Training and serving plus a quantization practice | S3, S6 |
| **Snowflake** | Arctic and an inference-optimization team | S3, S6, S9 |
| **Fal, Modal, Replicate, Anyscale, Deep Infra, Novita** | Varying depth; Fal in particular does real kernel work (diffusion-weighted) | S1, S5, S6 |
| **ByteDance Seed, Tencent, Baidu** | Very large internal inference organizations | S1, S9 |
| **Groq, Cerebras, SambaNova** | Custom silicon with proprietary toolchains. CUDA does not transfer; **the roofline, layout, and measurement reasoning does** | S6, S8, S12 |

**How to be legible to Tier 2.** These are the most portfolio-responsive employers on this list. A public repository with credible numbers, a released harness, and a merged upstream PR is close to a direct application. Several actively recruit from vLLM and SGLang contributor lists.

---

## Tier 3 · Hardware vendors

The largest absolute number of kernel-engineering roles, and the deepest technical ladders.

| Organization | Why this work is relevant | Skills that map |
|---|---|---|
| **NVIDIA** | CUTLASS, cuBLAS/cuBLASLt, TensorRT-LLM, Nsight, DL Algorithms, DevTech. **This project is effectively an audition for CUTLASS or TRT-LLM**, and CUTLASS is open, so contribution is itself a path in | S1–S4, S6, S7 — the full stack |
| **AMD** | ROCm, Composable Kernel, hipBLASLt, AITER. Aggressively closing a software gap, which means **high demand and materially less competition than NVIDIA.** Porting your kernel to MI300X/MI355X and reporting the delta would be an unusually strong differentiator for modest additional effort | S1, S2, S6, S8, **S12** |
| **AWS (Annapurna Labs)** | Trainium / Inferentia, Neuron SDK, and **NKI** (Neuron Kernel Interface). They explicitly need people who can author low-level kernels for non-NVIDIA silicon — a much thinner talent pool | S2, S6, S8, **S12** |
| **Google (TPU)** | XLA, Pallas, Mosaic. Same reasoning, different ISA | S6, S8, S12 |
| **Intel** | Gaudi, oneDNN, SYCL, OpenVINO | S2, S6, S12 |
| **Apple** | MLX and Metal kernels, with **on-device low-bit weight quantization as a first-order concern.** Weight-only 4-bit is arguably more central to Apple's product than to any datacenter vendor's | S2, S3, S6, S12 |
| **Qualcomm / ARM / MediaTek** | On-device NPUs with aggressive sub-8-bit quantization; block-scaled formats are directly relevant | S2, S3, S12 |
| **Tenstorrent, Etched, d-Matrix, Furiosa, Rebellions, MatX, Positron, Rivos** | Silicon startups. Small, senior-heavy, and kernel/compiler work is existential rather than optional | S2, S6, S8, S12 |

**How to be legible to Tier 3.** Vendors screen on depth and on the ability to *explain* a datapath. The D1 design note — the layout derivation from first principles — is your strongest artifact here, more than the benchmark numbers. Bring it to the whiteboard.

---

## Tier 4 · Frameworks, compilers, and open source

Often the *fastest* route into all three tiers above, because contribution is public and unmediated.

| Project / org | Relevance | Skills |
|---|---|---|
| **vLLM** (PyTorch Foundation, Red Hat-heavy) | D8's primary target; the quantization and MoE kernel surface | S1–S4, S9, S10 |
| **SGLang** (LMSYS) | `sgl-kernel`; alternative D8 target | S1, S4, S9, S10 |
| **PyTorch / torchao / Inductor** (Meta) | Where quantized kernels and codegen live | S3, S5, S10 |
| **Triton** (OpenAI + community) | The DSL twin, and upstream contribution opportunities | S5, S10 |
| **CUTLASS** (NVIDIA, open) | Contributing is a documented path into NVIDIA | S4, S2 |
| **llama.cpp / ggml** | Low-bit quantized kernels are the *entire* project | S2, S3 |
| **Microsoft Research** | BitBLAS, Ladder, T-MAC — MSR Systems does exactly this class of low-bit kernel work; also DeepSpeed, ONNX Runtime, MAIA silicon | S1–S3, S6, S11 |
| **HuggingFace** | `optimum`, `kernels`, transformers quantization integration | S3, S5, S10 |
| **Modular (Mojo / MAX)** | Explicitly building a CUDA alternative; hires kernel people directly | S2, S6, S12 |
| **JAX / XLA** (Google) | Pallas and Mosaic as the kernel surface | S12, S6 |

---

## Tier 5 · Adjacent markets where these skills pay unusually well

Worth knowing about, because the compensation and the intellectual culture can both exceed the obvious path.

| Sector | Who | Why they want it |
|---|---|---|
| **Quantitative trading / HFT** | XTX Markets, Jane Street, Citadel Securities, Two Sigma, Jump Trading, Hudson River Trading, Squarepoint | XTX in particular runs a very large GPU fleet for ML-driven trading. These firms screen hard for exactly the discipline in S8 — *measure properly, understand the machine, do not fool yourself* — and for low-latency instincts. **Compensation frequently exceeds frontier labs.** The benchmark-methodology chapter of your report is unusually persuasive here, more so than the kernel |
| **National labs / HPC** | LLNL, ORNL, Argonne, Sandia, NERSC, CSCS, Jülich, RIKEN | Roofline analysis and performance engineering are the native culture. Low-precision arithmetic is an active research area. Slower hiring, unusual intellectual freedom |
| **Robotics / autonomous systems** | Tesla, Waymo, Wayve, Figure, Physical Intelligence, Skild | On-device inference under hard latency and power budgets. Weight-only low-bit quantization is critical, not optional |
| **Scientific / industrial GPU computing** | NVIDIA Earth-2, weather and climate, computational chemistry, genomics, medical imaging | GPU performance specialists are scarce in every one of these; the skills transfer wholesale even though the kernels differ |

---

## The honest read on how this converts

Four things worth internalizing, in order of how much they matter:

**1. Problem selection is scarcer than kernel ability, and it is judged first.** Kernel skill is legible and teachable — demonstrable in an interview, verifiable from a repository. Judgment about which problem deserves eight months is much harder to fake, and v1.0 of this brief was a real signal in the wrong direction: excellent discipline aimed at a settled question. v2.0 fixes exactly that, and the *visible trail* of having revised on evidence is itself part of the artifact. Keep the review, keep the revision history, and be willing to say "I changed the target after I checked what was already solved."

**2. The upstream PR outranks the repository, by a lot.** Everything in Tier 1 and much of Tier 3 hires through referral, publication, and contribution history rather than portfolio review. One merged kernel PR in vLLM puts your name in a place those organizations actually look, and gets you a real reviewer relationship in the process. It is the highest-leverage single artifact in the plan and it is also the cheapest.

**3. S8 travels further than S1.** The `wgmma` skill gets you a kernel role. The measurement discipline gets you taken seriously in *any* performance-critical organization, including the ones paying the most, and it is what makes senior people trust your numbers without re-deriving them. Lead with the harness in conversation more often than you expect to.

**4. The single highest-leverage action remains non-technical.** Ask the vLLM / SGLang / torchao maintainers whether an MXFP4 Hopper throughput path is wanted and unowned — before building. You either get a named human confirming demand, which converts adoption from hope into commitment and hands you an expert reviewer for free, or you learn it is already being done and save two months. **Every recommendation in this document and its companions is inference; one maintainer conversation outranks all of it.**

---

*Companion documents: `PROJECT-v2.0-fp4-hopper.md` (the revised brief) · `REVIEW-fused-w4a16-gemm-v1.0.md` (the review that produced it, including sources and three corrections to earlier advice).*
