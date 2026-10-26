# Technical issue log

| Observation | Resolution / status |
|---|---|
| MedNet README names MedNet; source class is Darma | Use actual Darma(num_classes=2), unpack scores and activations |
| MedNet.py imports training dependencies for a forward-only use case | AST adapter compiles seven original class bodies unchanged; no training loop or training-only imports run |
| MedNet source preprocessing converts RGB to BGR then normalizes | Adapter explicitly matches this source behavior; color convention must be reconsidered for future trained checkpoints |
| No trained MedNet checkpoint in inspected repository | Reproducible random initialization; untrained outputs clearly labeled; no clinical labels assigned |
| Local OpenMP shared-memory initialization blocked in sandbox | Same integrated demo succeeded outside restricted sandbox; failure log is retained separately |
| Prepared Colab notebooks are not GPU evidence | LiteMedSAM GPU output verified previously; MedNet/integration now verified in appended cells of the existing Colab notebook; separate notebook executions remain a distinct reproducibility check |
| Full research dataset/labels not present | Demo-only testing; no training, dataset collection or accuracy claims |
| Readme initially described only LiteMedSAM scope | Extended project and assignment audit now distinguish both models and integration |
| Colab rich-output JavaScript failed to load after successful GPU execution | Colab suggested expired login or third-party-cookie restrictions; notebook changes saved and compact text confirms results. Reload attempted; archive download to computer not yet confirmed. No browser security settings changed. |
