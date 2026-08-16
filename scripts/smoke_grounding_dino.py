from __future__ import annotations

import argparse
import json

import cv2

from baseline.inference import Baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Load GroundingDINO and optionally infer one image")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image")
    parser.add_argument("--query")
    args = parser.parse_args()

    predictor = Baseline(
        config_path=args.model_config,
        weight_path=args.weights,
        device=args.device,
    )
    print(f"model_loaded device={predictor.model.device}")
    if args.image is None:
        return
    if not args.query:
        parser.error("--query is required with --image")
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(json.dumps(predictor.predict(image, args.query), ensure_ascii=False))


if __name__ == "__main__":
    main()
