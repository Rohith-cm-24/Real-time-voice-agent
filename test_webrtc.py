#!/usr/bin/env python3
"""
Test script to verify WebRTC backend is working
"""
import requests
import json

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        data = response.json()
        print(f"✅ Health check: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_webrtc_offer():
    """Test WebRTC offer endpoint"""
    print("\nTesting WebRTC offer endpoint...")
    try:
        # Create a dummy SDP offer
        offer = {
            "sdp": "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n",
            "type": "offer"
        }
        
        response = requests.post("http://localhost:8000/webrtc/offer", json=offer)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ WebRTC offer accepted")
            print(f"   Session ID: {data.get('session_id', 'N/A')}")
            print(f"   Has SDP: {bool(data.get('sdp'))}")
            return True
        else:
            print(f"❌ WebRTC offer failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ WebRTC offer test failed: {e}")
        return False

def test_env():
    """Check environment variables"""
    print("\nChecking environment variables...")
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    print(f"DEEPGRAM_API_KEY: {'✅ Set' if deepgram_key else '❌ Not set'}")
    print(f"GROQ_API_KEY: {'✅ Set' if groq_key else '❌ Not set'}")
    
    return bool(deepgram_key and groq_key)

if __name__ == "__main__":
    print("=" * 60)
    print("WebRTC Backend Test")
    print("=" * 60)
    
    results = []
    results.append(test_env())
    results.append(test_health())
    results.append(test_webrtc_offer())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")
    print("=" * 60)

