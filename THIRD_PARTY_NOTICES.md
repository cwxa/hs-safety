# Third-party resources and licenses

The MIT license in this repository applies to original project code. Dataset records, pretrained models, external packages and derived artifacts remain subject to their respective terms. No upstream model or dataset files are bundled in the public source release.

| Resource | Upstream license declared in the retrieved card | Use |
| --- | --- | --- |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | CC BY-NC 4.0 | Response-level safety annotations; pinned revision in `configs/main.yaml` |
| [StreamGuardBench](https://huggingface.co/datasets/Alibaba-AAIG/StreamGuardBench) | Apache-2.0 at the pinned original revision | External response-source evaluation |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 | Frozen monitoring encoder |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache-2.0 | Optional backbone-scale validation |
| [IEEEtran](https://ctan.org/pkg/ieeetran) | LaTeX Project Public License | Conference manuscript class, obtained through the user's TeX distribution |

Research references are in `papers/refs.bib`, including SCM and Kelp. They inform the experimental design; the public release does not bundle their code or the installed Supervisor-Skills package. Runtime dependencies are installed separately through `pyproject.toml` or the recorded environment lock and retain their own licenses.

Aggregate measurements, vector figures and their numeric source tables are included to document the reported research. Reusing upstream data or training derivative artifacts requires checking the applicable upstream terms, including BeaverTails' noncommercial condition. The project's MIT license does not relicense these resources.
