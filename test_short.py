import json
from model import FakeNewsDetector

def main():
    detector = FakeNewsDetector()

    if not detector.is_loaded:
        print("Failed to load model.")
        return

    test_cases = [
        "The earth is round.",
        "Water boils at 100 degrees Celsius.",
        "The sun rises in the east.",
        "Lizard people rule the government secretly.",
        "The moon landing was faked on a soundstage.",
        "Aliens built the pyramids.",
        "It is what it is.",
        "This is a fact."
    ]

    results = []
    for t in test_cases:
        res = detector.predict(t)
        results.append({"text": t, "result": res})
        
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main()
