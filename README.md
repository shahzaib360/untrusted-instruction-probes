# Probing for obedience to untrusted instructions

Linear probes on a language model's internal activations that detect when the model is **obeying an instruction that was injected into untrusted data** (a document it was asked to process), compared against text-only classifiers.

Status: pipeline complete; results are being computed and will be added here.

## Quick start

```bash
pip install -r requirements.txt
scripts/run_all.sh Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b
```

## License

MIT
