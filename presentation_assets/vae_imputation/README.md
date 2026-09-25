# VAE genotype imputation animation

- `../../img/vae_imputation.gif`: looping 16:9 animation for a PowerPoint slide (1600 × 900; about 10 seconds).
- `vae_imputation_poster.png`: static final frame for a thumbnail or non-animated export.
- `make_animation.py`: source used to regenerate both files with Pillow.

The alignment and predictions are illustrative. No VAE was trained or run to
produce the displayed calls or the 7/8 validation count. Teal cells indicate
originally missing calls. Amber cells are observed calls hidden by a simulated
validation mask; only those have stored truth and receive a score.

To regenerate the assets from the repository root:

```bash
python presentation_assets/vae_imputation/make_animation.py
```
