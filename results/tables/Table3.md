| Model                         | acc  | kappa | macro-AUC |
| ----------------------------- | ---- | ----- | --------- |
| XGB static-only (full cohort) | 0.49 | 0.37  | 0.814     |
| XGB static-only (ICU subset)  | 0.51 | 0.38  | 0.818     |
| XGB static+TS (ICU subset)    | 0.53 | 0.40  | 0.826     |
| GRU seq-only                  | 0.42 | 0.30  | 0.777     |
| GRU+static                    | 0.47 | 0.37  | 0.814     |
| GRU+TPR                       | 0.47 | 0.37  | 0.815     |
| GRU+static+TS                 | 0.48 | 0.37  | 0.812     |
| DG-GRU                        | 0.48 | 0.37  | 0.814     |
| DG-GRU+TPR                    | 0.47 | 0.36  | 0.815     |
| GRU long (60ep)               | 0.46 | 0.35  | 0.804     |
| GRU-D                         | 0.52 | 0.39  | 0.815     |
| Transformer                   | 0.47 | 0.35  | 0.784     |
| STGCN                         | 0.50 | 0.37  | 0.801     |
| Stack (proposed)              | 0.54 | 0.41  | 0.830     |
