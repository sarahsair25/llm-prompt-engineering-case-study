import json
import os


def analyze_openai_results(folder="openai_improved"):
    jsonl_path = os.path.join(folder, "eval_results.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            case = data['case']
            result = data['result']

            print(f"\n{'=' * 80}")
            print(f"CASE: {case['id']}")
            print(f"PROMPT: {case['prompt']}")
            print(f"{'-' * 80}")
            print(f"RAW RESPONSE:")
            print(result['raw_output'])
            print(f"{'-' * 80}")

            if data.get('parsed_json'):
                print(f"PARSED JSON KEYS: {list(data['parsed_json'].keys())}")

            print(f"SCORE: {result['score']}")
            print(f"NOTES: {result['notes']}")
            print(f"{'=' * 80}\n")


if __name__ == "__main__":
    analyze_openai_results()