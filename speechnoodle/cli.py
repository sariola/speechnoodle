# audio_dataset_creator/cli.py

import argparse
import asyncio

from speechnoodle.speechnoodle import (
    AudioDatasetCreator,
    AudioDatasetCreatorConfig,
    FilterConditions,
    SplitRatios,
)


def parse_args() -> argparse.Namespace:
    default_config = AudioDatasetCreatorConfig()

    parser = argparse.ArgumentParser(
        description="Create an audio dataset from a directory of audio files."
    )
    parser.add_argument("source_dir", help="Source directory containing audio files")
    parser.add_argument("dest_dir", help="Destination directory for the created dataset")
    parser.add_argument("--num_dirs", type=int, help="Number of directories to process (optional)")

    # Use default values from AudioDatasetCreatorConfig
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=default_config.split_ratios.train,
        help="Ratio of training data",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=default_config.split_ratios.validation,
        help="Ratio of validation data",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=default_config.split_ratios.test,
        help="Ratio of test data",
    )
    parser.add_argument(
        "--speech_duration",
        type=float,
        default=default_config.filter_conditions.speech_duration,
        help="Minimum speech duration",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=default_config.filter_conditions.snr,
        help="Minimum signal-to-noise ratio",
    )
    parser.add_argument(
        "--c50", type=float, default=default_config.filter_conditions.c50, help="Minimum C50 value"
    )
    parser.add_argument(
        "--stoi",
        type=float,
        default=default_config.filter_conditions.stoi,
        help="Minimum STOI value",
    )
    parser.add_argument(
        "--pesq",
        type=float,
        default=default_config.filter_conditions.pesq,
        help="Minimum PESQ value",
    )
    parser.add_argument(
        "--sdr", type=float, default=default_config.filter_conditions.sdr, help="Minimum SDR value"
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    config = AudioDatasetCreatorConfig(
        split_ratios=SplitRatios(
            train=args.train_ratio, validation=args.validation_ratio, test=args.test_ratio
        ),
        filter_conditions=FilterConditions(
            speech_duration=args.speech_duration,
            snr=args.snr,
            c50=args.c50,
            stoi=args.stoi,
            pesq=args.pesq,
            sdr=args.sdr,
        ),
    )

    creator = AudioDatasetCreator(config)
    await creator.create_dataset(args.source_dir, args.dest_dir, args.num_dirs)


if __name__ == "__main__":
    asyncio.run(main())
