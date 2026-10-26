# Assignment audit — 2026-10-26

Source: professor's AIH_LiteMedSAM_MedNet_Student_Handout.pdf.

| Requirement | Status | Evidence / remaining work |
|---|---|---|
| Fork LiteMedSAM | Complete | KrishnachaithanyaThummala/MedSAM, LiteMedSAM branch |
| Task 1: obtain both implementations and provenance | Complete locally | Both source checkouts; commits recorded below and in run report |
| Task 2: LiteMedSAM Colab demo | Complete for one demo | Browser review: Tesla T4, CUDA check, official checkpoint, ultrasound box, displayed/saved mask, 4990 foreground pixels |
| Task 3: MedNet forward pass | Local and Colab GPU verified | Actual Darma architecture, finite 1x2 output on Tesla T4; verification appended to existing live Colab notebook; separate 02 notebook prepared |
| Task 4: integration | Local and Colab GPU verified | Image + box -> mask -> three crops -> two scores each; saved GPU report confirmed in live Colab |
| Task 5: modular source organization | Implemented | Four src modules, three assignment notebooks, README, requirements and tests |
| Task 6: full demonstration | Engineering demo verified; student explanation pending | Live Colab model/pipeline outputs, local saved evidence and published modular source; standalone new notebooks still need their own saved execution outputs |
| Student understanding | In progress | Models-and-code lessons available; exercises and interactive walkthrough still required |

LiteMedSAM source commit: b0fab476e54e631dd412b25e0db9fdf2a2b0f54c.
MedNet source commit (main): 6adb072630811b1d39568a87732aed96dc39c9ae.
Original sources: https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM and https://github.com/Md-Ferdous/MedNet.

Local standalone Darma test: [1,3,224,224] zeros -> scores [[0.04124801978468895,0.02989237755537033]], five feature maps, random seed 42, no trained classifier checkpoint.
Local integrated tight crop scores: [[0.04322659969329834,0.025680024176836014]]. These are untrained numerical outputs, not class probabilities or diagnostic predictions.

Local pipeline wall time: 3.555555 seconds in one run, including loading and I/O. Colab LiteMedSAM case time previously observed: 4.367787 seconds. These are not comparable pure-model benchmarks.

Four automated checks passed: exact per-side margin bounds, boundary clipping, explicit empty-mask failure, and RGB-to-BGR normalization behavior. Original model implementations are unchanged. The adapter omits only unrelated MedNet imports/training utilities while compiling original architecture class bodies.

Reading a document does not certify mastery. Begin with MODELS_AND_CODE_LESSONS.md, then the actual source and executed notebooks. Full research data, training and clinical performance evaluation remain future work.

## Colab verification added during this session
Existing notebook: https://colab.research.google.com/drive/1OmhWWMlCFhMEOPHVzz1ZvWSd-C9xyG85
MedNet zero-input GPU logits: [[0.04124801605939865, 0.02989237755537033]].
Compact verification printed GPU Tesla T4, REPORT_EXISTS True, SEGMENTATION_DEVICE cuda:0, and three [1,2] crop outputs. GPU total wall time was 10.925201574998937 seconds, including loading/I/O. Results archive generation/download was requested; local ZIP arrival has not been confirmed. New standalone notebooks are prepared; GPU verification was performed in appended cells of the existing notebook.
Published source: https://github.com/KrishnachaithanyaThummala/MedSAM/tree/codex/student-foundation/student_project
