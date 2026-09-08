import csv
from pathlib import Path

from object_detection.ml.classifier import classify_image_timed


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_DIRECTORY = PROJECT_ROOT / "evaluation_data"
RESULTS_FILE = PROJECT_ROOT / "model_comparison_results.csv"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODELS = ("mobilenetv2", "efficientnetb0")


def get_evaluation_images():
    evaluation_images = []

    for actual_label, folder_name in (("Cat", "cats"), ("Dog", "dogs")):
        folder_path = EVALUATION_DIRECTORY / folder_name

        for image_path in sorted(folder_path.iterdir()):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                evaluation_images.append((image_path, actual_label))

    return evaluation_images


def evaluate_model(model_name, evaluation_images):
    results = []

    print(f"\nTesting {model_name}...")

    for image_path, actual_label in evaluation_images:
        predicted_label, confidence, elapsed_seconds = classify_image_timed(
            str(image_path),
            model_name,
        )

        is_correct = predicted_label.lower() == actual_label.lower()

        result = {
            "model": model_name,
            "image": image_path.name,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "confidence": round(float(confidence) * 100, 2),
            "time_seconds": round(float(elapsed_seconds), 4),
            "correct": is_correct,
        }

        results.append(result)

        status = "Correct" if is_correct else "Incorrect"

        print(
            f"{image_path.name}: "
            f"actual={actual_label}, "
            f"predicted={predicted_label}, "
            f"confidence={result['confidence']}%, "
            f"time={result['time_seconds']}s, "
            f"{status}"
        )

    return results


def print_summary(all_results):
    print("\nModel comparison summary")
    print("-" * 70)

    for model_name in MODELS:
        model_results = [
            result
            for result in all_results
            if result["model"] == model_name
        ]

        correct_count = sum(
            1 for result in model_results if result["correct"]
        )
        incorrect_count = len(model_results) - correct_count

        accuracy = (
            correct_count / len(model_results) * 100
            if model_results
            else 0
        )

        average_time = (
            sum(result["time_seconds"] for result in model_results)
            / len(model_results)
            if model_results
            else 0
        )

        print(f"\nModel: {model_name}")
        print(f"Images tested: {len(model_results)}")
        print(f"Correct predictions: {correct_count}")
        print(f"Incorrect predictions: {incorrect_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average prediction time: {average_time:.4f} seconds")


def save_results(all_results):
    fieldnames = [
        "model",
        "image",
        "actual_label",
        "predicted_label",
        "confidence",
        "time_seconds",
        "correct",
    ]

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nDetailed results saved to: {RESULTS_FILE}")


def main():
    evaluation_images = get_evaluation_images()

    if not evaluation_images:
        print("No evaluation images were found.")
        return

    cat_count = sum(
        1 for _, label in evaluation_images if label == "Cat"
    )
    dog_count = sum(
        1 for _, label in evaluation_images if label == "Dog"
    )

    print(f"Cat images found: {cat_count}")
    print(f"Dog images found: {dog_count}")
    print(f"Total images found: {len(evaluation_images)}")

    all_results = []

    for model_name in MODELS:
        model_results = evaluate_model(model_name, evaluation_images)
        all_results.extend(model_results)

    save_results(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()