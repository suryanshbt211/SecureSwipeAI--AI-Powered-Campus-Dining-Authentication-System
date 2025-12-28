#!/usr/bin/env python
"""
Quick start script for SecureSwipe AI
Run this file to start the application.
"""

if __name__ == '__main__':
    print("=" * 60)
    print("🍽️  SecureSwipe AI - Starting Server...")
    print("=" * 60)
    print("\nMake sure you have installed all dependencies:")
    print("  pip install -r requirements.txt")
    print("\nThe application will be available at:")
    print("  http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 60)
    
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

