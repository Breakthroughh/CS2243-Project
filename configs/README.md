# Experiment configurations

Store complete, versioned experiment configurations here. Once used for a reported
run, preserve that configuration and add a new file for changes. Record seeds,
task suite and initial-state IDs, episode limits, model/data revisions, precision,
resources, and checkpoint source as applicable. Never include credentials.

The current smoke entrypoint accepts CLI arguments; it does not load configuration
files from this directory. Modal includes this directory at `/root/configs` for
future experiment entrypoints.
