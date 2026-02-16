import sys
import os
import time
import json
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reasoning_layer.llm import LLMReasoner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_tests():
    print("=== WalkSense Reasoning Benchmark ===")
    print("Connecting to LLM Backend (Ollama/LM Studio)...")
    
    try:
        # Try to connect to Ollama (default from logs)
        llm = LLMReasoner(backend="ollama", api_url="http://localhost:11434", model_name="gemma3:270m")
        # Or try connection to check if alive? LLMReasoner init doesn't connect immediately usually, but let's assume it works.
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    resources_path = os.path.join(os.path.dirname(__file__), 'resources', 'benchmark_cases.json')
    try:
        with open(resources_path, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Test cases file not found at {resources_path}")
        return

    results = []
    
    print(f"\nRunning {len(test_cases)} test cases...")
    
    for case in test_cases:
        print(f"\nTest Case {case['id']}: {case['query']}")
        start_t = time.time()
        try:
            answer = llm.answer_query(
                user_query=case['query'],
                spatial_context=case['spatial'],
                scene_description=case['vlm']
            )
            duration = time.time() - start_t
            
            # Check correctness
            passed = any(k.lower() in answer.lower() for k in case['expected_keywords'])
            status = "PASS" if passed else "FAIL"
            
            print(f"Answer: {answer}")
            print(f"Status: {status} ({duration:.2f}s)")
            
            results.append({
                "id": case['id'],
                "category": case['category'],
                "passed": passed,
                "duration": duration,
                "answer": answer
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "id": case['id'],
                "category": case['category'],
                "passed": False,
                "duration": 0,
                "answer": str(e)
            })

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    accuracy = (passed / total) * 100 if total > 0 else 0
    avg_latency = sum(r['duration'] for r in results) / total if total > 0 else 0
    
    print("\n=== Benchmark Results ===")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Avg Latency: {avg_latency:.2f}s")
    
    # Save results
    output_path = r'd:\Github\WalkSense\docs\benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": time.time(),
            "accuracy": accuracy,
            "avg_latency": avg_latency,
            "details": results
        }, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_tests()
