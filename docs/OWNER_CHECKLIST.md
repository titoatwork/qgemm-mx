# What the project owner needs to provide

Agents/collaborators can build almost everything else **locally**.

## Required for peak claims (Hopper)

- [ ] **DGX / H100 (sm_90) access**  
  - Run: `make ARCH=sm_90 run-probe`  
  - Paste/save output into `PREREGISTRATION.md` sm_90 table  
  - Prefer exclusive node; log co-tenants  

- [ ] **Decide ncu**  
  - Laptop: fix Windows GPU Performance Counters **or** accept profiler-free method  
  - DGX: ask admin for counter access  

## Strongly recommended

- [ ] **Upstream interest** (one Discord/GitHub question):  
  “Is an MXFP4 Hopper throughput path / 2880-align fix wanted and unowned?”  

## Not required for daily progress

- GitHub pushes  
- Continuous micro-commits  
- Product UI / full vLLM clone  

## When you reply to the agent

Useful short messages:

- `ncu fixed` / `skip ncu`  
- `DGX available now` + how you connect  
- `pause local` / `focus on D1 only`  
