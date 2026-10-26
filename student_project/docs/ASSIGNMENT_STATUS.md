# Assignment audit — 2026-10-26

Source: professor's AIH_LiteMedSAM_MedNet_Student_Handout.pdf.

| Requirement | Status | Evidence / remaining work |
|---|---|---|
| Fork LiteMedSAM | Complete | KrishnachaithanyaThummala/MedSAM, LiteMedSAM branch |
| Task 1: obtain both implementations and provenance | Complete locally | Both source checkouts; commits recorded below and in run report |
| Task 2: LiteMedSAM Colab demo | Complete for one demo | Browser review: Tesla T4, CUDA check, official checkpoint, ultrasound box, displayed/saved mask, 4990 foreground pixels |
| Task 3: MedNet forward pass | Local complete; Colab pending | Actual Darma architecture, finite 1x2 output; 02_mednet_test.ipynb prepared, GPU execution not yet verified |
| Task 4: integration | Local complete; Colab pending | Image + box -> mask -> three crops -> two scores each; outputs/integrated_local |
| Task 5: modular source organization | Implemented | Four src modules, three assignment notebooks, README, requirements and tests |
| Task 6: full demonstration | Partial | Saved local outputs and timing; full Colab demonstration and student explanation still needed |
| Student understanding | In progress | Models-and-code lessons available; exercises and interactive walkthrough still required |

LiteMedSAM source commit: b0fab476e54e631dd412b25e0db9fdf2a2b0f54c.
MedNet source commit (main): 6adb072630811b1d39568a87732aed96dc39c9ae.
Original sources: https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM and https://github.com/Md-Ferdous/MedNet.

Local standalone Darma test: [1,3,224,224] zeros -> scores [[0.04124801978468895,0.02989237755537033]], five feature maps, random seed 42, no trained classifier checkpoint.
Local integrated tight crop scores: [[0.04322659969329834,0.025680024176836014]]. These are untrained numerical outputs, not class probabilities or diagnostic predictions.

Local pipeline wall time: 3.555555 seconds in one run, including loading and I/O. Colab LiteMedSAM case time previously observed: 4.367787 seconds. These are not comparable pure-model benchmarks.

Four automated checks passed: exact per-side margin bounds, boundary clipping, explicit empty-mask failure, and RGB-to-BGR normalization behavior. Original model implementations are unchanged. The adapter omits only unrelated MedNet imports/training utilities while compiling original architecture class bodies.

Reading a document does not certify mastery. Begin with MODELS_AND_CODE_LESSONS.md, then the actual source and executed notebooks. Full research data, training and clinical performance evaluation remain future work.
