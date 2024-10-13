Add badges -> same as in flow-judge
https://img.shields.io/badge/mypy-checked-blue

Copy the dataspeech repo from HF to this repo and credit them

Add better GPU usage setting, just per one GPU and test
-> setting job to just one gpu and running three in parallel
-> running one job with 3 gpus
-> give some benchmark numbers


Add more notebooks
- custom step usage
- using the filter
- multi-lingual
- usage on single gpu or multiple gpus
- CLI usage

Checking the results using parquet tool

Adding full testing coverage

Add instructions how to add espeak path -> test on macos or linux
Add e2e testing on the build server


Caveats:
- this is for single speakers only, diarization is out of scope, but you could contribute a custom step for it


- Finetuning step for
    - StyleTTSV2
    WITH YAML config format that fits speechnoodle



- COMPARE ORIGINAL ENRICHMENTS AND TEST THE NEW ONES OUT
    - IMPROVE THEM
