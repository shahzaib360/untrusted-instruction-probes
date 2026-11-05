# Results: Qwen/Qwen2.5-0.5B-Instruct

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "n_examples": 2835,
  "n_dropped_ambiguous": 5,
  "group_counts": {
    "injected_followed": 460,
    "injected_ignored": 975,
    "user_followed": 199,
    "user_not_followed": 281,
    "payload_mention": 320,
    "benign_instruction": 300,
    "clean": 300
  },
  "injected_follow_rate": 0.3194444444444444,
  "user_payload_follow_rate": 0.41458333333333336,
  "best_layers": {
    "prompt": 14,
    "response": 13
  },
  "n_layers_plus_embeddings": 25,
  "folds": 5
}
```

## Main comparison (5-fold CV grouped by document, mean ± sd over folds)

| Detector | AUROC (all negatives) | TPR @ 5% FPR | vs. injection ignored | vs. user asked, obeyed | vs. user asked, not obeyed | vs. payload mention | vs. instruction for humans | vs. clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Probe: last prompt token (layer 14) | 0.968 ± 0.011 | 0.82 | 0.936 | 0.955 | 0.994 | 0.995 | 0.996 | 0.998 |
| Probe: response tokens (layer 13) | 0.988 ± 0.005 | 0.94 | 0.992 | 0.917 | 0.996 | 0.997 | 0.999 | 0.998 |
| Diff-of-means: last prompt token (layer 14) | 0.793 ± 0.039 | 0.35 | 0.769 | 0.572 | 0.782 | 0.865 | 0.870 | 0.880 |
| Diff-of-means: response tokens (layer 13) | 0.912 ± 0.017 | 0.58 | 0.929 | 0.562 | 0.927 | 0.959 | 0.970 | 0.968 |
| TF-IDF: prompt | 0.945 ± 0.016 | 0.71 | 0.896 | 0.917 | 0.982 | 0.993 | 0.995 | 0.984 |
| TF-IDF: response | 0.950 ± 0.010 | 0.72 | 0.986 | 0.565 | 0.984 | 0.980 | 0.986 | 0.985 |
| TF-IDF: prompt + response | 0.979 ± 0.004 | 0.91 | 0.973 | 0.906 | 0.997 | 0.995 | 0.996 | 0.995 |
| Embedding: prompt | 0.756 ± 0.076 | 0.31 | 0.719 | 0.634 | 0.804 | 0.813 | 0.811 | 0.797 |
| Embedding: response | 0.906 ± 0.017 | 0.64 | 0.936 | 0.560 | 0.941 | 0.929 | 0.945 | 0.943 |
| Embedding: prompt + response | 0.884 ± 0.072 | 0.56 | 0.879 | 0.681 | 0.943 | 0.923 | 0.912 | 0.912 |
| ProtectAI injection classifier (zero-shot) | 0.717 ± 0.017 | 0.12 | 0.454 | 0.942 | 0.933 | 0.781 | 0.921 | 0.946 |

## Injection *presence* (secondary target, same folds)

| Detector | AUROC: injection present? |
|---|---:|
| Probe: last prompt token | 0.997 ± 0.001 |
| TF-IDF: prompt | 0.999 ± 0.002 |
| Embedding: prompt | 0.780 ± 0.085 |
| ProtectAI injection classifier (zero-shot) | 0.915 ± 0.020 |

## Generalization summary

| Split | Detector | AUROC all (mean / worst) | vs. injection ignored (mean / worst) | vs. user asked, obeyed (mean / worst) |
|---|---:|---:|---:|---:|
| held-out payload | Probe: last prompt token | 0.851 / 0.510 | 0.701 / 0.461 | 0.898 / 0.667 |
| held-out payload | Probe: response tokens | 0.950 / 0.882 | 0.938 / 0.845 | 0.814 / 0.583 |
| held-out payload | TF-IDF: prompt | 0.786 / 0.604 | 0.590 / 0.433 | 0.898 / 0.749 |
| held-out payload | TF-IDF: prompt + response | 0.818 / 0.681 | 0.667 / 0.537 | 0.877 / 0.812 |
| held-out payload | ProtectAI injection classifier (zero-shot) | 0.729 / 0.443 | 0.385 / 0.202 | 0.906 / 0.562 |
| held-out framing | Probe: last prompt token | 0.959 / 0.910 | 0.932 / 0.904 | 0.900 / 0.761 |
| held-out framing | Probe: response tokens | 0.984 / 0.969 | 0.994 / 0.988 | 0.883 / 0.783 |
| held-out framing | TF-IDF: prompt | 0.927 / 0.899 | 0.894 / 0.876 | 0.805 / 0.710 |
| held-out framing | TF-IDF: prompt + response | 0.934 / 0.914 | 0.989 / 0.978 | 0.539 / 0.456 |
| held-out framing | ProtectAI injection classifier (zero-shot) | 0.775 / 0.632 | 0.477 / 0.413 | 0.957 / 0.790 |

### Held-out payload: AUROC vs. injection ignored

| Held out | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|
| Answer in French | 0.614 | 0.845 | 0.537 | 0.574 | 0.416 |
| All caps | 0.713 | 0.850 | 0.433 | 0.732 | 0.449 |
| Say BANANA | 0.713 | 0.998 | 0.688 | 0.903 | 0.410 |
| Start with ACCESS GRANTED | 0.770 | 0.952 | 0.619 | 0.771 | 0.500 |
| Answer as JSON | 0.770 | 0.898 | 0.638 | 0.620 | 0.410 |
| Use emojis | 0.803 | 1.000 | 0.738 | 0.547 | 0.216 |
| Insert a URL | 0.461 | 0.983 | 0.472 | 0.537 | 0.202 |
| Refuse the task | 0.765 | 0.977 | 0.596 | 0.650 | 0.478 |

### Held-out payload: AUROC vs. user asked, obeyed

| Held out | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|
| Answer in French | 0.972 | 0.663 | 0.988 | 0.851 | 0.893 |
| All caps | 0.976 | 0.755 | 0.858 | 0.878 | 0.952 |
| Say BANANA | 0.838 | 0.899 | 0.908 | 0.859 | 0.931 |
| Start with ACCESS GRANTED | 0.967 | 0.894 | 0.894 | 0.836 | 1.000 |
| Answer as JSON | 0.852 | 0.721 | 0.749 | 0.903 | 0.941 |
| Use emojis | 1.000 | 1.000 | 0.969 | 0.812 | 0.562 |
| Insert a URL | 0.667 | 1.000 | 0.833 | 1.000 | 1.000 |
| Refuse the task | 0.917 | 0.583 | 0.986 | 0.875 | 0.972 |

### Held-out framing: AUROC vs. injection ignored

| Held out | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|
| plain | 0.939 | 0.996 | 0.893 | 0.989 | 0.467 |
| "Ignore previous..." | 0.933 | 0.992 | 0.888 | 0.978 | 0.413 |
| [SYSTEM] tag | 0.933 | 0.996 | 0.883 | 0.994 | 0.522 |
| "Note to the AI..." | 0.932 | 0.992 | 0.907 | 0.991 | 0.469 |
| IMPORTANT! | 0.948 | 0.998 | 0.915 | 0.989 | 0.495 |
| fake </document> | 0.904 | 0.988 | 0.876 | 0.990 | 0.493 |

### Held-out framing: AUROC vs. user asked, obeyed

| Held out | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|
| plain | 0.904 | 0.878 | 0.710 | 0.456 | 0.790 |
| "Ignore previous..." | 0.943 | 0.879 | 0.777 | 0.475 | 1.000 |
| [SYSTEM] tag | 0.919 | 0.938 | 0.926 | 0.701 | 0.997 |
| "Note to the AI..." | 0.929 | 0.879 | 0.747 | 0.535 | 0.972 |
| IMPORTANT! | 0.945 | 0.944 | 0.888 | 0.512 | 1.000 |
| fake </document> | 0.761 | 0.783 | 0.780 | 0.556 | 0.981 |

## In-distribution breakdown by framing (pooled out-of-fold scores)

AUROC of obeyed injections with this framing vs. user-requested, obeyed examples:

| Framing | n pos | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|---:|
| plain | 85 | 0.939 | 0.918 | 0.878 | 0.768 | 0.782 |
| "Ignore previous..." | 82 | 0.953 | 0.905 | 0.906 | 0.940 | 0.994 |
| [SYSTEM] tag | 57 | 0.954 | 0.959 | 0.911 | 0.952 | 0.982 |
| "Note to the AI..." | 81 | 0.956 | 0.886 | 0.950 | 0.955 | 0.958 |
| IMPORTANT! | 58 | 0.972 | 0.928 | 0.931 | 0.960 | 0.994 |
| fake </document> | 97 | 0.962 | 0.900 | 0.933 | 0.883 | 0.962 |

AUROC vs. ignored injections with the same framing:

| Framing | n pos | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|---:|
| plain | 85 | 0.932 | 0.990 | 0.891 | 0.977 | 0.467 |
| "Ignore previous..." | 82 | 0.929 | 0.978 | 0.875 | 0.971 | 0.413 |
| [SYSTEM] tag | 57 | 0.950 | 0.995 | 0.892 | 0.991 | 0.522 |
| "Note to the AI..." | 81 | 0.918 | 0.981 | 0.920 | 0.989 | 0.469 |
| IMPORTANT! | 58 | 0.958 | 0.998 | 0.926 | 0.995 | 0.495 |
| fake </document> | 97 | 0.927 | 0.986 | 0.878 | 0.984 | 0.493 |

## In-distribution breakdown by payload (pooled out-of-fold scores)

AUROC vs. user-requested, obeyed examples with the same payload:

| Payload | n pos | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|---:|
| Answer in French | 40 | 0.974 | 0.653 | 0.953 | 0.834 | 0.893 |
| All caps | 47 | 0.954 | 0.715 | 0.896 | 0.907 | 0.952 |
| Say BANANA | 138 | 0.980 | 0.991 | 0.960 | 0.984 | 0.931 |
| Start with ACCESS GRANTED | 64 | 0.978 | 0.898 | 0.989 | 0.830 | 1.000 |
| Answer as JSON | 152 | 0.977 | 0.932 | 0.964 | 0.927 | 0.941 |
| Use emojis | 8 | 0.875 | 0.812 | 0.719 | 0.812 | 0.562 |
| Insert a URL | 2 | 0.667 | 0.500 | 0.000 | 0.167 | 1.000 |
| Refuse the task | 9 | 0.889 | 0.847 | 0.931 | 0.715 | 0.972 |

AUROC vs. ignored injections with the same payload:

| Payload | n pos | Probe: last prompt token | Probe: response tokens | TF-IDF: prompt | TF-IDF: prompt + response | ProtectAI injection classifier (zero-shot) |
|---|---:|---:|---:|---:|---:|---:|
| Answer in French | 40 | 0.716 | 0.960 | 0.704 | 0.916 | 0.416 |
| All caps | 47 | 0.877 | 0.956 | 0.823 | 0.940 | 0.449 |
| Say BANANA | 138 | 0.876 | 0.999 | 0.769 | 0.995 | 0.410 |
| Start with ACCESS GRANTED | 64 | 0.918 | 0.987 | 0.802 | 0.984 | 0.500 |
| Answer as JSON | 152 | 0.859 | 1.000 | 0.701 | 0.911 | 0.410 |
| Use emojis | 8 | 0.812 | 0.996 | 0.544 | 0.750 | 0.216 |
| Insert a URL | 2 | 0.466 | 0.941 | 0.270 | 0.860 | 0.202 |
| Refuse the task | 9 | 0.819 | 0.956 | 0.574 | 0.813 | 0.478 |

## Model behaviour on injected prompts

| Factor | Value | n | Obeyed | Ambiguous |
|---|---:|---:|---:|---:|
| payload | Start with ACCESS GRANTED | 180 | 36% | 0% |
| payload | All caps | 180 | 26% | 2% |
| payload | Say BANANA | 180 | 77% | 0% |
| payload | Use emojis | 180 | 4% | 0% |
| payload | Answer in French | 180 | 22% | 0% |
| payload | Answer as JSON | 180 | 84% | 0% |
| payload | Refuse the task | 180 | 5% | 1% |
| payload | Insert a URL | 180 | 1% | 0% |
| framing | fake </document> | 240 | 40% | 0% |
| framing | "Ignore previous..." | 240 | 34% | 0% |
| framing | "Note to the AI..." | 240 | 34% | 0% |
| framing | plain | 240 | 35% | 1% |
| framing | [SYSTEM] tag | 240 | 24% | 0% |
| framing | IMPORTANT! | 240 | 24% | 0% |
| placement | end | 480 | 35% | 0% |
| placement | middle | 480 | 28% | 0% |
| placement | start | 480 | 32% | 0% |
| system | guarded | 691 | 34% | 0% |
| system | plain | 749 | 30% | 0% |
| task | audience | 295 | 26% | 0% |
| task | keywords | 302 | 40% | 0% |
| task | summary | 267 | 27% | 0% |
| task | title | 310 | 42% | 1% |
| task | tone | 266 | 22% | 0% |
