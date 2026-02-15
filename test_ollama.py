#!/usr/bin/env python3
"""
Test script to verify Ollama is running and accessible.
"""
import requests

def test_ollama_connection():
    """Test if Ollama is running and list available models."""
    base_url = "http://localhost:11434"
    
    print("Testing Ollama connection...")
    print(f"Base URL: {base_url}\n")
    
    # Test 1: Check if Ollama is running
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running and accessible")
            models = response.json().get("models", [])
            if models:
                print(f"\n✓ Available models ({len(models)}):")
                for model in models:
                    print(f"  - {model.get('name', 'unknown')}")
            else:
                print("\n⚠ No models found. Please pull a model first:")
                print("  Example: ollama pull llama2")
            return True
        else:
            print(f"✗ Unexpected response: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama")
        print("\nPlease ensure Ollama is running:")
        print("  1. Check if Ollama is installed: ollama --version")
        print("  2. Start Ollama service if not running")
        print("  3. Pull a model: ollama pull llama2")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    test_ollama_connection()
