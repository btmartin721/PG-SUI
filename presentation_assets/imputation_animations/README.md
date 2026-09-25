# PG-SUI workflow animations

Run from the repository root:

```bash
python presentation_assets/imputation_animations/make_animations.py
```

The script writes nine looping GIFs to `img/`: one for each of the eight public
imputers and one comparing all five simulated-missingness strategies. Use
`--output-dir` to write them elsewhere.

All input genotypes are synthetic. The neural and supervised predictions are
scripted illustrations; no model is trained or evaluated to generate them. The
reference-allele and most-frequent outputs follow their respective rules on
the synthetic matrix. Teal cells denote originally missing calls, while amber
cells are observed calls temporarily hidden for validation. Only the latter
have known ground truth for scoring.

The masking comparison includes the five implemented strategy names:
`random`, `random_weighted`, `random_weighted_inv`, `nonrandom`, and
`nonrandom_weighted`. The cell selections illustrate their tendencies and are
not samples produced by `SimMissingTransformer`. Tree-based strategies require
a supplied genotype tree. `random_weighted_inv` is the configuration and CLI
name for inverse-frequency weighting.

The older VAE slide animation is generated separately by
`presentation_assets/vae_imputation/make_animation.py` and is also stored in
`img/vae_imputation.gif`.
