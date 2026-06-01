# Interview notes: serving architecture (Mon production project)

Scope note. This document is about the **production clinical summariser I built
and deployed at Mon**, not the Qwen3-3B PoC in this repo. Serving notes for the
PoC live in a separate document. The point of this file is to have defensible,
honest answers ready for the Heidi conversation.

Resume line this supports:

> Reduced consultation review time from ~10 mins to ~3 mins and response latency
> from ~8s to ~1.5s by rearchitecting the clinical summariser from vendor LLMs
> to locally fine-tuned LLMs, and introducing a rubric-guided HITL feedback
> system that transformed clinician corrections into data for continuous
> improvement.

______________________________________________________________________

## What was deployed

- Fine-tuned LLM, quantised to **GGUF (q4-class)**.
- Served via **llama.cpp** (`llama-server`, OpenAI-compatible HTTP).
- Dockerised and deployed as **Kubernetes Deployments** on a **GPU** node pool.
- Horizontal scaling via **K8s Horizontal Pod Autoscaler (HPA)**.
- Latency target met: ~1.5s response latency (down from ~8s on the prior
  vendor-LLM architecture).

______________________________________________________________________

## The core engineering decision: llama.cpp + GGUF vs vLLM

The honest framing is **latency-bound vs throughput-bound**, plus a
quantisation-format constraint. We were latency-bound at moderate, bursty
concurrency, so we optimised for per-request latency, small footprint, and fast
pod start rather than maximum batched throughput.

### Why llama.cpp + GGUF was the right call for our load profile

1. **We were latency-bound, not throughput-bound.** vLLM's headline wins
   (PagedAttention, continuous batching) are *throughput* gains under
   *concurrent* load. They do not make a single warm request meaningfully
   faster. We already hit ~1.5s, so the constraint was per-consultation latency,
   which we met.

2. **vLLM does not serve GGUF well.** vLLM's supported path is fp16/bf16 or
   GPU-native quantisation (AWQ, GPTQ, FP8). GGUF support is experimental and
   slow. "Switch to vLLM" was therefore not a drop-in: it would mean
   re-quantising to AWQ/GPTQ (and re-validating output quality) or serving fp16
   at ~4x the VRAM of our q4 GGUF.

3. **Footprint and cost.** The q4 GGUF is small, so we could pack more replicas
   per GPU and use cheaper GPUs. fp16/AWQ under vLLM erodes that advantage.

4. **Fast time-to-ready (see cold-start section).** Smaller image and smaller
   weights mean new pods become ready quickly, which matters for HPA scale-out.

### Where vLLM would genuinely have been better (concede these honestly)

- **Instantaneous in-pod concurrency.** If many requests land on one pod at the
  same instant, vLLM's continuous batching absorbs that far better than
  llama.cpp.
- **Tail latency under sustained high concurrency.** vLLM holds p95/p99 much
  better as concurrent slots fill.
- **Automatic prefix caching.** Our clinical templates share a long fixed system
  prompt + schema on every request. vLLM's prefix caching reuses the KV cache
  for that shared prefix, cutting per-request prompt processing. This is a real
  win for our prompt shape. (llama.cpp has prompt caching too, but it is
  per-slot and less automatic across concurrent requests.)

### One-line verdict

> We were latency-bound at moderate concurrency and chose q4 GGUF + llama.cpp
> for footprint, cost efficiency, and fast pod start on k8s, hitting the 1.5s
> target. vLLM becomes the right move when the bottleneck shifts from
> per-request latency to concurrent throughput, but it would require migrating
> off GGUF to AWQ/GPTQ or fp16, trading memory footprint and cost for higher
> batched throughput and better tail latency. Given our load profile, that trade
> was not justified yet.

______________________________________________________________________

## Anticipated question: "What did you lose by not using vLLM?"

**Honest answer (corrected version of the obvious one):**

We lost the ability to absorb **instantaneous in-pod concurrency** efficiently.
If many clinicians hit the *same* pod within the same moment, vLLM's continuous
batching would have served that burst better on a fixed pod.

We compensated on two axes, and it is important to be precise about which
problem each solves:

- **Per-pod request queuing + parallel slots** in `llama-server` (`--parallel`),
  plus running with **replica headroom** (not at 100% utilisation), absorbs
  short instantaneous bursts.
- **K8s HPA** adds pods to handle **sustained** increases in concurrency.

**Caveat to state out loud (this is the part the naive answer gets wrong):** HPA
does **not** solve a single-millisecond spike. HPA reacts on metrics over
seconds-to-minutes, then schedules + starts a pod, so it adds capacity over
time. It handles *sustained* load growth, not an instantaneous micro-burst on an
already-saturated pod. The instantaneous burst is handled by in-pod
queuing/slots and headroom, not by autoscaling. Conflating the two is the trap;
separating them is the senior signal.

Numbers to have ready: peak concurrent requests, typical and peak QPS, replica
count and headroom factor, HPA target metric and thresholds. If real load was
genuinely high and spiky, be prepared to defend why in-pod queuing + headroom
was sufficient versus moving to vLLM batching.

______________________________________________________________________

## Anticipated question: cold start / time-to-ready

**Claim:** vLLM images are large (full PyTorch + CUDA, ~8-10GB+); llama.cpp
compiles to a small binary, so the container image is much smaller and "time to
ready" is near-instant. When HPA spins up a pod, llama.cpp is serving while a
vLLM pod is still `ContainerCreating`.

**This is true and feasible, with one qualifier:**

- vLLM images really are large because they bundle PyTorch and CUDA libraries.
  Correct.
- llama.cpp GPU images still need CUDA runtime libs (so not tiny on GPU), but
  are far smaller than a full PyTorch image. CPU images are very small.
- **Qualifier:** the dominant cold-start cost is frequently **model loading**
  (pulling weights + loading into VRAM), not the image pull. llama.cpp wins here
  *too*, because our q4 GGUF (~2GB) is roughly 3x smaller than fp16 (~6GB).
- **Second qualifier:** image-layer caching hides the image-pull cost when HPA
  scales onto **already-warm nodes**. The image-size advantage bites mainly on
  **new-node scale-out** (cluster autoscaler adding a fresh node). Mitigations
  either way: pre-pulling images, node warm pools, and keeping weights small.

**Defensible framing:** "Smaller image plus smaller quantised weights gave us a
fast time-to-ready, which matters most on new-node scale-out. We reinforced it
with [pre-pulled images / warm node pool / readiness probes]. A vLLM pod's
larger image and fp16 weights would lengthen that path."

______________________________________________________________________

## Quick reference: llama.cpp vs vLLM

| Dimension                               | llama.cpp                                                | vLLM                                                  |
| --------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------- |
| Hardware                                | CPU, Metal, single/multi GPU, edge                       | NVIDIA GPU only                                       |
| Quantisation                            | GGUF (q4/q5/q8), tiny footprint                          | fp16/bf16, AWQ/GPTQ/FP8 (GGUF only experimental)      |
| Single-request latency                  | low when warm                                            | low when warm (not the differentiator)                |
| Concurrency / throughput                | OK at low/moderate; degrades sooner                      | best under load (PagedAttention, continuous batching) |
| Tail latency under load                 | degrades sooner                                          | holds p95/p99 better                                  |
| Prefix caching for shared system prompt | per-slot, less automatic                                 | automatic prefix caching                              |
| Image size / cold start                 | small image, fast ready                                  | large image, slower ready                             |
| Footprint / cost per GPU                | low (more replicas per GPU)                              | higher (fp16/AWQ memory)                              |
| Best fit                                | latency-bound, cost-sensitive, on-device, fast scale-out | throughput-bound, high concurrency, GPU-rich          |

______________________________________________________________________

## Honesty checklist before using the resume line

- Be ready to explain how **review time 10->3 min** was measured (study method,
  sample size).
- Be ready to give the **vendor baseline** for the ~8s and the **hardware** for
  the ~1.5s.
- Be ready to describe the **HITL rubric feedback loop**: how clinician
  corrections were captured, turned into training data, and fed back (and how
  often retraining ran).
- Distinguish **measured outcomes** from **design intent** without being
  prompted.

## k8s scaling questions

1. Can GGUF run on GPU? Yes. GGUF is just a model file format (weights +
   metadata + quantisation). It is format-agnostic about hardware. What runs it
   is the runtime: llama.cpp. llama.cpp can be compiled with GPU backends:

- CUDA (NVIDIA) — what you'd use on k8s GPU nodes
- Metal (Apple) — what you use locally on M2 Pro
- ROCm (AMD), Vulkan, SYCL The key knob is --n-gpu-layers (offloading). You
  choose how many transformer layers live in VRAM vs system RAM: -ngl 0 all
  layers on CPU (slow, no GPU used) -ngl 99 all layers offloaded to GPU (fast,
  full GPU) -ngl 20 partial: 20 on GPU, rest CPU (hybrid, when model > VRAM) So
  your Mon setup — quantised GGUF on GPU via llama.cpp CUDA with full offload —
  is exactly normal. The misconception is "GGUF = CPU only." Not true. GGUF's
  advantage is it can do CPU, GPU, or hybrid; fp16+vLLM is GPU-only.

______________________________________________________________________

2. Containers, Pods, and where the GPU attaches Build up the nesting first:
   ┌─────────────────────────────────────────────────────────────┐ │ NODE (a
   physical/virtual machine, e.g. a GPU VM) │ │ - has the real hardware: CPUs,
   RAM, and 1+ physical GPUs │ │ - runs the kubelet + a GPU device plugin │ │ │
   │ ┌───────────────────────────────────────────────┐ │ │ │ POD (smallest
   deployable unit in k8s) │ │ │ │ - has its own network identity (1 IP) │ │ │ │
   \- is what k8s schedules and scales │ │ │ │ │ │ │ │
   ┌─────────────────────────┐ │ │ │ │ │ CONTAINER │ │ │ │ │ │ - your llama.cpp
   binary│ │ │ │ │ │ + the GGUF file │ │ │ │ │ │ - the GPU is granted │ │ │ │ │
   │ to THIS container │ │ │ │ │ └─────────────────────────┘ │ │ │ │ (a pod can
   hold >1 container, e.g. a sidecar)│ │ │
   └───────────────────────────────────────────────┘ │ │ │ │ GPU 0 ◄──
   attached/exposed into the container above │ │ GPU 1 ◄── could go to a
   different pod │
   └─────────────────────────────────────────────────────────────┘ Where does
   the GPU attach — pod or container? Subtle but important: you request the GPU
   at the container level in the spec, but k8s schedules at the pod level.
   apiVersion: v1 kind: Pod spec: containers:

- name: llama-server image: my-llamacpp-gguf:latest resources: limits:
  nvidia.com/gpu: 1 # ← request is on the CONTAINER What actually happens:

1. You declare nvidia.com/gpu: 1 on the container.
2. The scheduler sums the pod's requests and finds a node that has a free GPU.
3. The NVIDIA device plugin on that node exposes the physical GPU's device files
   into that container's namespace.
4. The container's llama.cpp process now sees the GPU and offloads layers to it.
   So: requested per-container, scheduled per-pod, physically owned by the node.
   The GPU is "attached" to the container, but the unit that gets placed on a
   GPU-having node is the pod. Key constraint: a physical GPU is, by default,
   whole-GPU per container — not time-sliced across pods unless you enable MIG
   (A100/H100 partitioning) or time-slicing. So normally: 1 pod → 1 container →
   1 whole GPU. A node with 4 GPUs can run 4 such pods. This is also why your q4
   GGUF footprint matters: a smaller model means you could fit it on a
   smaller/cheaper GPU, or (with time-slicing/MIG) pack more replicas per
   physical GPU.

______________________________________________________________________

3. How scaling works in k8s Two layers stack on top of each other: Layer A —
   Horizontal Pod Autoscaler (HPA): add/remove pods metric crosses threshold
   (e.g. GPU util > 70%, or queue depth) │ ▼ ┌────────────┐ HPA decides
   ┌────────────┐ ┌────────────┐ │ Pod 1 │ ───────────────────► │ Pod 1 │ │ Pod
   2 │ │ (1 GPU) │ scale 1 → 3 │ (1 GPU) │ │ (1 GPU) │ └────────────┘
   └────────────┘ └────────────┘ ┌────────────┐ │ Pod 3 │ │ (1 GPU) │
   └────────────┘ A Service / load balancer spreads incoming requests across the
   pods.

- HPA watches a metric (CPU, GPU utilisation via custom metrics, or a custom
  signal like request-queue length).
- When it crosses a target, HPA changes the Deployment's replica count.
- New pods need a free GPU on some node. If every GPU is taken, the new pod is
  Pending until capacity appears → which triggers Layer B. Layer B — Cluster
  Autoscaler: add/remove nodes HPA wants Pod 4, but all GPUs on existing nodes
  are full │ ▼ Cluster Autoscaler adds a NEW GPU NODE (cloud VM provisioning) │
  ▼ New node boots → joins cluster → Pod 4 schedules → pulls image → loads GGUF
  → Ready The timing reality (ties back to your interview answer) instantaneous
  burst ──► absorbed by: in-pod queue + parallel slots + replica HEADROOM
  (milliseconds) sustained load rise ──► absorbed by: HPA adds pods on warm
  nodes (seconds — if free GPUs exist) load beyond cluster ──► absorbed by:
  Cluster Autoscaler adds nodes (minutes — VM boot + image pull + weight load)
  This is exactly why the "HPA handled the same-millisecond spike" claim is
  shaky: HPA lives in the seconds-to-minutes band, not the millisecond band. And
  it's why your small image + small GGUF weights matter — they shrink the
  "minutes" band on new-node scale-out, making time-to-ready faster than a vLLM
  pod stuck pulling a 10GB image and loading fp16 weights.
