"""
CLI entry point for HiSVG inference.

Usage:
    # Batch inference (ShareGPT format, default)
    python -m hivg_infer.cli \
        --model_path /path/to/model \
        --dataset test.json \
        --output_dir ./outputs

    # Batch inference (Alpaca format)
    python -m hivg_infer.cli \
        --model_path /path/to/model \
        --dataset test.json \
        --format alpaca \
        --output_dir ./outputs

    # Single text2svg
    python -m hivg_infer.cli \
        --model_path /path/to/model \
        --prompt "Draw a simple icon"

    # Single img2svg
    python -m hivg_infer.cli \
        --model_path /path/to/model \
        --image /path/to/image.png

    # Interactive mode
    python -m hivg_infer.cli \
        --model_path /path/to/model \
        --interactive
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="HiSVG Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_path", type=str, required=True,
                             help="Path to model checkpoint or HuggingFace hub ID")
    model_group.add_argument("--adapter_path", type=str, default=None,
                             help="Path to LoRA adapter (optional, requires peft)")
    model_group.add_argument("--torch_dtype", type=str, default="bfloat16",
                             choices=["bfloat16", "float16", "float32"],
                             help="Model weight dtype")
    model_group.add_argument("--device_map", type=str, default="auto",
                             help="HuggingFace device map strategy")

    svg_group = parser.add_argument_group("SVG")
    svg_group.add_argument("--coord_range", type=int, default=234,
                           help="SVG coordinate range (canvas size)")
    svg_group.add_argument("--relative_range", type=int, default=None,
                           help="Relative coordinate range")

    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument("--temperature", type=float, default=0.7)
    gen_group.add_argument("--top_p", type=float, default=0.9)
    gen_group.add_argument("--top_k", type=int, default=50)
    gen_group.add_argument("--max_new_tokens", type=int, default=4096)
    gen_group.add_argument("--repetition_penalty", type=float, default=1.0)

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--dataset", type=str, default=None,
                          help="Path to dataset JSON for batch inference")
    io_group.add_argument("--format", type=str, default="sharegpt",
                          choices=["sharegpt", "alpaca"],
                          help="Dataset format")
    io_group.add_argument("--max_samples", type=int, default=None,
                          help="Maximum samples to process (default: all)")
    io_group.add_argument("--output_dir", type=str, default="./svg_outputs",
                          help="Output directory for generated SVGs")
    io_group.add_argument("--save_tokens", action="store_true",
                          help="Also save raw token files")

    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument("--interactive", action="store_true",
                            help="Run in interactive mode")
    mode_group.add_argument("--image", type=str, default=None,
                            help="Image path for single img2svg inference")
    mode_group.add_argument("--prompt", type=str, default=None,
                            help="Text description for text2svg (plain text; "
                                 "training instruction prefix is added automatically)")

    return parser.parse_args()


def _create_pipeline(args):
    from .pipeline import HiSVGInferencePipeline

    return HiSVGInferencePipeline(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        coord_range=args.coord_range,
        relative_range=args.relative_range,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )


def _run_batch(pipeline, args):
    print(f"Batch inference  | dataset: {args.dataset} | format: {args.format}")
    print(f"Output directory : {args.output_dir}")
    if args.max_samples:
        print(f"Max samples      : {args.max_samples}")

    results = pipeline.generate_batch(
        dataset=args.dataset,
        output_dir=args.output_dir,
        save_svg=True,
        save_tokens=args.save_tokens,
        max_samples=args.max_samples,
        fmt=args.format,
    )
    success = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {success}/{len(results)} successful")
    return results


def _run_single(pipeline, args):
    if args.image:
        print(f"img2svg | image: {args.image}")
        result = pipeline.img2svg(args.image)
    else:
        description = args.prompt or "A simple icon."
        print(f"text2svg | description: {description}")
        result = pipeline.text2svg(description)

    if result["success"]:
        print("\n--- Generated SVG ---")
        print(result["svg"])
    else:
        print(f"\nError: {result['error']}")
    return result


def _run_interactive(pipeline):
    import os

    print("Interactive SVG generation mode.")
    print("Commands: 'quit' to exit, 'img:<path>' for img2svg")
    print("For text2svg, enter a plain description — the [T2ST] prefix is added automatically.")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        if user_input.startswith("img:"):
            image_path = user_input[4:].strip()
            if os.path.exists(image_path):
                result = pipeline.img2svg(image_path)
            else:
                print(f"Image not found: {image_path}")
                continue
        else:
            result = pipeline.text2svg(user_input)

        if result["success"]:
            print("\n--- Generated SVG ---")
            preview = result["svg"]
            if len(preview) > 500:
                preview = preview[:500] + "..."
            print(preview)
        else:
            print(f"Error: {result['error']}")


def main():
    args = parse_args()

    print("Initializing HiSVG inference pipeline...")
    pipeline = _create_pipeline(args)
    print("Pipeline ready.\n")

    if args.dataset:
        _run_batch(pipeline, args)
    elif args.interactive:
        _run_interactive(pipeline)
    elif args.prompt or args.image:
        _run_single(pipeline, args)
    else:
        print("No input specified. Use --dataset, --interactive, --prompt, or --image")
        print("Run with --help for usage.")


if __name__ == "__main__":
    main()
