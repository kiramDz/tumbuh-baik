from functools import wraps
from flask import request, jsonify
from pymongo import MongoClient
from datetime import datetime
import urllib.parse
import os

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get session token from cookie
        raw_token = request.cookies.get('better-auth.session_token')
        
        if not raw_token:
            return jsonify({"success": False, "error": {"code": "UNAUTHORIZED", "message": "Session required. Please login."}}), 401

        # URL Decode cookie (%3D menjadi =) jika ada, lalu ambil string sebelum tanda titik (.)
        decoded_token = urllib.parse.unquote(raw_token)
        session_token = decoded_token.split('.')[0]
        
        # Validate token in MongoDB better-auth session table
        session = db.session.find_one({
            "token": session_token,
            "expiresAt": {"$gt": datetime.utcnow()}
        })
        
        if not session:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_SESSION",
                    "message": "Invalid or expired session. Please login."
                }
            }), 401
        
        # Attach to request for later use
        request.session = session
        
        return f(*args, **kwargs)
    return decorated_function