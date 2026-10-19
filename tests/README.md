# Fast local checks

Run from the repository root:

```sh
python3 -m unittest discover -s tests -v
```

These checks exercise package imports and Modal wiring without a Modal account,
simulator install, or GPU. Cloud image installation and EGL rendering still need
the separate onboarding and simulator smoke runs.
