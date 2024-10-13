# speechnoodle

<p align="center">
  <img src="img/speechnoodle_banner.png" alt="Speechnoodle Banner">
</p>

<p align="center" style="font-family: 'Courier New', Courier, monospace;">
  <strong>
    <a href="https://github.com/yourusername/speechnoodle/tree/main/examples">Examples</a> |
    <a href="https://github.com/yourusername/speechnoodle/blob/main/CONTRIBUTING.md">Contributing</a> |
    <a href="https://github.com/yourusername/speechnoodle/blob/main/LICENSE">License</a>
  </strong>
</p>

<p align="center" style="font-family: 'Courier New', Courier, monospace;">
  <code>speechnoodle</code> is a flexible library for creating and processing audio datasets using customizable pipelines.
</p>

<p align="center">
<a href="https://github.com/yourusername/speechnoodle/stargazers/" target="_blank">
    <img src="https://img.shields.io/github/stars/yourusername/speechnoodle?style=social&label=Star&maxAge=3600" alt="GitHub stars">
</a>
<a href="https://github.com/yourusername/speechnoodle/releases" target="_blank">
    <img src="https://img.shields.io/github/v/release/yourusername/speechnoodle?color=blue" alt="Release">
</a>
<a href="https://github.com/yourusername/speechnoodle/actions/workflows/test-and-lint.yml" target="_blank">
    <img src="https://github.com/yourusername/speechnoodle/actions/workflows/test-and-lint.yml/badge.svg" alt="Build">
</a>
<a href="https://codecov.io/gh/yourusername/speechnoodle" target="_blank">
    <img src="https://codecov.io/gh/yourusername/speechnoodle/branch/main/graph/badge.svg" alt="Code coverage">
</a>
<a href="https://github.com/yourusername/speechnoodle/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=Apache%202.0&color=blue" alt="License">
</a>
</p>

## Features

- Flexible audio dataset creation pipeline
- Integration with Huggingface's dataspeech utility scripts
- Support for custom processing steps
- Efficient batch processing and GPU utilization
- Easy-to-use CLI for dataset creation
- Extensible architecture for adding new features

## Installation

Install speechnoodle using pip:

```bash
pip install -e ".[dev,extra]"
```

Extras available:
- `dev` to install development dependencies
- `extra` to install extra dependencies

## Quick Start

Here's a simple example to get you started:

```python
from speechnoodle import AudioDatasetCreator, AudioDatasetCreatorConfig
from speechnoodle.processors import AudioProcessor, DatasetCreator, DatasetSplitter, DatasetFilter

# Configure the dataset creator
config = AudioDatasetCreatorConfig()

# Create a custom pipeline
pipeline = [
    AudioProcessor(),
    DatasetCreator(),
    DatasetSplitter(config.split_ratios),
    DatasetFilter(config.filter_conditions)
]

# Initialize the dataset creator
creator = AudioDatasetCreator(config=config, pipeline=pipeline)

# Create the dataset
creator.create_dataset("path/to/audio/files", "path/to/output/dataset")
```

## Usage

### Command Line Interface

You can use speechnoodle from the command line:

```bash
python -m speechnoodle.cli source_dir dest_dir --train_ratio 0.8 --validation_ratio 0.1 --test_ratio 0.1
```

### Custom Processing Steps

You can create custom processing steps by subclassing `PipelineStep`:

```python
startLine: 17
endLine: 37
```

### GPU Utilization

speechnoodle supports efficient GPU usage. You can control GPU memory utilization:

```python
from speechnoodle import AudioDatasetCreator, AudioDatasetCreatorConfig

config = AudioDatasetCreatorConfig()
creator = AudioDatasetCreator(config=config)
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/speechnoodle.git
   cd speechnoodle
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev,extra]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

Run the tests using pytest:

```bash
pytest tests/
```

## Contributing

Contributions to speechnoodle are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

speechnoodle uses Huggingface's dataspeech utility scripts. We appreciate their contributions to the open-source community.

## Caveats

- This library is designed for processing single-speaker audio datasets. Multi-speaker diarization is out of scope, but you can contribute a custom step for it if needed.
- Ensure you have sufficient disk space and GPU memory when processing large datasets.
```
