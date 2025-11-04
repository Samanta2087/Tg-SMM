"""
SVXHUB - Smart Digital SMM Reseller Bot
A production-ready Telegram bot that connects to Xmedia SMM API
"""

import os
import sqlite3
import logging
import requests
import razorpay
import random
import string
import re
import io
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from dotenv import load_dotenv
from PIL import Image
import easyocr

from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    ParseMode
)
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext
)
from apscheduler.schedulers.background import BackgroundScheduler
from telegram.error import BadRequest

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
BOT_USERNAME = os.getenv('BOT_USERNAME', 'svxhub_bot')  # Your bot username without @
SUPPLIER_API_KEY = os.getenv('SUPPLIER_API_KEY', 'YOUR_API_KEY_HERE')
ADMIN_ID = int(os.getenv('ADMIN_ID', '0'))
SUPPLIER_API_URL = 'https://xmediasmm.in/api/v2'
PROFIT_MARGIN = 1.5  # 50% profit margin

# Virtual Number Service Configuration
# Prefer environment variables; fall back to sensible defaults
VIRTUAL_NUMBER_API_KEY = os.getenv('VIRTUAL_NUMBER_API_KEY', 'UTGKmFqiCYaIPExxMpWE9wIpsfTzNT1P')
# Note: Provider appears to serve API under /api/handler_api.php (not /stubs)
VIRTUAL_NUMBER_API_URL = os.getenv('VIRTUAL_NUMBER_API_URL', "https://otpx.in/api/handler_api.php")
VIRTUAL_NUMBER_DEFAULT_PRICE_INR = float(os.getenv('VIRTUAL_NUMBER_DEFAULT_PRICE_INR', '50'))
VIRTUAL_NUMBER_OPERATOR = os.getenv('VIRTUAL_NUMBER_OPERATOR', '').strip()  # optional, e.g., 'any'
VIRTUAL_NUMBER_SHOW_PROVIDER_BALANCE = os.getenv('VIRTUAL_NUMBER_SHOW_PROVIDER_BALANCE', 'true').lower() == 'true'
VIRTUAL_NUMBER_ENABLED = os.getenv('VIRTUAL_NUMBER_ENABLED', 'true').lower() == 'true'

# VN provider URL resolution cache and candidates
vn_resolved_url: Optional[str] = None
VN_URL_CANDIDATES = [
    # Preferred env URL first
    VIRTUAL_NUMBER_API_URL,
    # Documented and potential alternates
    "https://otpx.in/stubs/handler_api.php",
    "https://otpx.in/handler_api.php",
    "http://otpx.in/stubs/handler_api.php",
    "http://otpx.in/handler_api.php",
    "https://api.otpx.in/handler_api.php",
    "https://api.otpx.in/stubs/handler_api.php",
]

def vn_request(params: Dict[str, Any]) -> requests.Response:
    """Make a VN API request with fallback URL discovery.
    Returns a Response that is HTTP 200 and not an HTML page; raises on failure.
    Caches the working URL in-memory for subsequent calls.
    """
    global vn_resolved_url
    tried = []
    # Build attempt list: resolved first if present, then unique candidates
    attempt_urls = []
    if vn_resolved_url:
        attempt_urls.append(vn_resolved_url)
    for u in VN_URL_CANDIDATES:
        if u and u not in attempt_urls:
            attempt_urls.append(u)

    last_exc: Optional[Exception] = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    for u in attempt_urls:
        tried.append(u)
        try:
            resp = requests.get(u, params=params, headers=headers, timeout=12)
            text = (resp.text or "").strip()
            if resp.status_code == 200 and not text.lower().startswith('<!doctype') and '<html' not in text[:200].lower():
                vn_resolved_url = u
                return resp
            logger.warning(f"VN request bad candidate {u} HTTP {resp.status_code}: {text[:120]}")
        except Exception as e:
            last_exc = e
            logger.warning(f"VN request exception for {u}: {e}")

    raise RuntimeError(f"VN provider unreachable or invalid URL. Tried: {', '.join(tried)}. Last error: {last_exc}")

# Conversation states
AWAITING_PAYMENT_PROOF = 1
AWAITING_PAYMENT_AMOUNT = 2
AWAITING_SUPPORT_MESSAGE = 3
AWAITING_ORDER_QUANTITY = 4
AWAITING_ORDER_LINK = 5
AWAITING_QR_CODE = 6
AWAITING_UTR_NUMBER = 7
AWAITING_RAZORPAY_AMOUNT = 8
AWAITING_PAYMENT_VERIFICATION = 9
AWAITING_RAZORPAY_KEY_ID = 10
AWAITING_RAZORPAY_KEY_SECRET = 11
AWAITING_WALLET_ADD_AMOUNT = 12
AWAITING_WALLET_DEDUCT_AMOUNT = 13
AWAITING_USER_ID_FOR_WALLET = 14
AWAITING_BROADCAST_MESSAGE = 15
AWAITING_VIRTUAL_NUMBER_COUNTRY = 16
AWAITING_VIRTUAL_NUMBER_SERVICE = 17

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def safe_edit_message_text(query, text, reply_markup=None, parse_mode='Markdown') -> None:
    """Safely edit a message; ignore 'Message is not modified' errors and soft-handle others."""
    try:
        query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except Exception as e:
        if isinstance(e, BadRequest) and 'Message is not modified' in str(e):
            # Ignore benign no-op edits
            try:
                query.answer()
            except Exception:
                pass
        else:
            logger.error(f"edit_message_text failed: {e}")
            try:
                query.answer("⚠️ Please try again.", show_alert=False)
            except Exception:
                pass


# ======================== OCR PAYMENT VERIFICATION ========================

# Initialize EasyOCR reader (supports English and numbers)
ocr_reader = None

def init_ocr_reader():
    """Initialize OCR reader (lazy loading)"""
    global ocr_reader
    if ocr_reader is None:
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {e}")
            raise
    return ocr_reader


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract all text from image using OCR - OPTIMIZED FOR SPEED
    
    Args:
        image_bytes: Image file as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        # Initialize reader if needed
        reader = init_ocr_reader()
        
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # SPEED OPTIMIZATION: Resize image if too large (max 1024px width)
        max_width = 1024
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {max_width}x{new_height} for faster OCR")
        
        # SPEED OPTIMIZATION: Use faster OCR settings
        results = reader.readtext(
            image, 
            detail=0, 
            paragraph=False,
            batch_size=1,
            contrast_ths=0.3,
            adjust_contrast=0.7
        )
        
        # Combine all text
        full_text = ' '.join(results)
        
        logger.info(f"OCR extracted text (length: {len(full_text)} chars) in optimized mode")
        logger.debug(f"OCR extracted text: {full_text[:500]}...")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""


def find_payment_code_in_text(text: str, expected_code: str = None) -> Tuple[bool, Optional[str], str]:
    """
    Search for payment code in extracted text - OPTIMIZED FOR SPEED
    
    Args:
        text: Extracted text from OCR
        expected_code: Optional expected code to validate against
        
    Returns:
        Tuple of (found, extracted_code, reason)
    """
    text = text.upper().strip()
    
    # SPEED OPTIMIZATION: Quick check if expected code exists in text
    if expected_code and expected_code.upper() in text:
        logger.info(f"Quick match: Expected code '{expected_code}' found in text")
        return True, expected_code.upper(), "Code found (quick match)"
    
    # Keywords that typically precede payment notes/codes
    keywords = [
        r'NOTE[S]?\s*:?\s*',
        r'REMARK[S]?\s*:?\s*',
        r'MESSAGE\s*:?\s*',
        r'DESCRIPTION\s*:?\s*',
        r'MEMO\s*:?\s*',
        r'COMMENT\s*:?\s*',
        r'REF\s*:?\s*',
        r'REFERENCE\s*:?\s*'
    ]
    
    # Try to find code after keywords
    for keyword in keywords:
        pattern = keyword + r'([A-Z0-9\s]{6,30})'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if matches:
            for match in matches:
                # Clean up the extracted code
                code = re.sub(r'\s+', '', match.strip())
                
                # Validate code format (alphanumeric, 6-20 chars)
                if 6 <= len(code) <= 20 and code.isalnum():
                    if expected_code:
                        if code == expected_code.upper():
                            return True, code, f"Valid code found after keyword"
                        else:
                            logger.info(f"Found code '{code}' but doesn't match expected '{expected_code}'")
                    else:
                        return True, code, f"Payment code found"
    
    # If no code found after keywords, search for standalone alphanumeric codes
    standalone_pattern = r'\b([A-Z]{3,10}[0-9]{3,10})\b'
    standalone_matches = re.findall(standalone_pattern, text)
    
    if standalone_matches:
        for code in standalone_matches:
            if 6 <= len(code) <= 20:
                if expected_code:
                    if code == expected_code.upper():
                        return True, code, f"Valid code found (standalone)"
                else:
                    return True, code, f"Payment code found (standalone)"
    
    # No valid code found
    if expected_code:
        return False, None, f"Expected code '{expected_code}' not found in receipt"
    else:
        return False, None, "No payment code found in receipt"


def verify_payment_receipt(image_bytes: bytes, expected_code: str = None, 
                          min_amount: float = None, utr_number: str = None) -> Dict:
    """
    Main verification function - scans receipt and validates payment
    
    Args:
        image_bytes: Payment receipt image as bytes
        expected_code: Expected payment verification code (optional)
        min_amount: Minimum amount that should be in receipt (optional)
        utr_number: UTR number to search for (optional)
        
    Returns:
        Dictionary with verification results
    """
    try:
        # Extract text from image
        extracted_text = extract_text_from_image(image_bytes)
        
        if not extracted_text:
            return {
                'verified': False,
                'code_found': None,
                'amount_found': None,
                'utr_found': False,
                'reason': 'Failed to extract text from image',
                'extracted_text': ''
            }
        
        # Search for payment code
        code_verified, found_code, code_reason = find_payment_code_in_text(extracted_text, expected_code)
        
        # Search for amount (optional)
        amount_found = None
        if min_amount:
            amount_pattern = r'[\u20B9]?\s*(\d{1,5}(?:\.\d{2})?)'
            amounts = re.findall(amount_pattern, extracted_text)
            
            if amounts:
                # Convert to floats and find largest
                float_amounts = [float(a) for a in amounts if float(a) >= min_amount]
                if float_amounts:
                    amount_found = max(float_amounts)
        
        # Search for UTR (optional)
        utr_verified = False
        if utr_number:
            utr_pattern = re.escape(utr_number.upper())
            if re.search(utr_pattern, extracted_text.upper()):
                utr_verified = True
        
        # Overall verification
        verified = code_verified
        if min_amount and not amount_found:
            verified = False
            code_reason += f" | Amount ₹{min_amount} not found"
        
        if utr_number and not utr_verified:
            verified = False
            code_reason += f" | UTR {utr_number} not found"
        
        return {
            'verified': verified,
            'code_found': found_code,
            'amount_found': amount_found,
            'utr_found': utr_verified if utr_number else None,
            'reason': code_reason,
            'extracted_text': extracted_text[:1000]
        }
        
    except Exception as e:
        logger.error(f"Error verifying payment receipt: {e}")
        return {
            'verified': False,
            'code_found': None,
            'amount_found': None,
            'utr_found': False,
            'reason': f'Verification error: {str(e)}',
            'extracted_text': ''
        }


# ======================== RATE LIMITING ========================

from collections import defaultdict
from time import time

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # Time window in seconds
RATE_LIMIT_MAX_REQUESTS = 20  # Max requests per window
RATE_LIMIT_COOLDOWN = 300  # Cooldown period in seconds after hitting limit

# Store user request timestamps
user_requests = defaultdict(list)
user_cooldowns = {}


def check_rate_limit(user_id: int) -> tuple[bool, Optional[str]]:
    """
    Check if user has exceeded rate limit
    Returns: (is_allowed, error_message)
    """
    current_time = time()
    
    # Check if user is in cooldown
    if user_id in user_cooldowns:
        cooldown_until = user_cooldowns[user_id]
        if current_time < cooldown_until:
            remaining = int(cooldown_until - current_time)
            minutes = remaining // 60
            seconds = remaining % 60
            return False, f"⚠️ You're sending requests too quickly!\n\nPlease wait {minutes}m {seconds}s before trying again."
        else:
            # Cooldown expired
            del user_cooldowns[user_id]
    
    # Clean old requests outside the time window
    user_requests[user_id] = [
        req_time for req_time in user_requests[user_id]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if user exceeded the limit
    if len(user_requests[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        # Put user in cooldown
        user_cooldowns[user_id] = current_time + RATE_LIMIT_COOLDOWN
        minutes = RATE_LIMIT_COOLDOWN // 60
        return False, f"⚠️ Rate limit exceeded!\n\nYou've made too many requests. Please wait {minutes} minutes before trying again."
    
    # Add current request
    user_requests[user_id].append(current_time)
    return True, None


def is_admin(user_id: int) -> bool:
    """Check if user is admin (admins bypass rate limiting)"""
    return user_id == ADMIN_ID


# ======================== DATABASE SETUP ========================

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('fastxhub.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            wallet REAL DEFAULT 0.0,
            join_date TEXT,
            is_blocked INTEGER DEFAULT 0,
            block_reason TEXT,
            referred_by INTEGER,
            referral_earnings REAL DEFAULT 0.0
        )
    ''')
    
    # Add is_blocked column if it doesn't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN is_blocked INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add block_reason column if it doesn't exist
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN block_reason TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add referral columns if they don't exist
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN referred_by INTEGER')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN referral_earnings REAL DEFAULT 0.0')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Services table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS services (
            id INTEGER PRIMARY KEY,
            name TEXT,
            supplier_rate REAL,
            sell_rate REAL,
            category TEXT,
            status TEXT,
            min_qty INTEGER DEFAULT 1,
            max_qty INTEGER DEFAULT 100000,
            service_status TEXT DEFAULT 'normal',
            price_category TEXT DEFAULT 'standard'
        )
    ''')
    
    # Orders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            service_id INTEGER,
            qty INTEGER,
            link TEXT,
            sell_price REAL,
            supplier_order_id INTEGER,
            status TEXT DEFAULT 'Pending',
            created TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (service_id) REFERENCES services(id)
        )
    ''')
    
    # Pending funds table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_funds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount REAL,
            photo_file_id TEXT,
            utr_number TEXT,
            payment_code TEXT,
            status TEXT DEFAULT 'pending',
            requested_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Add utr_number column if it doesn't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE pending_funds ADD COLUMN utr_number TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add payment_code column if it doesn't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE pending_funds ADD COLUMN payment_code TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add service_status column to services table if it doesn't exist
    try:
        cursor.execute('ALTER TABLE services ADD COLUMN service_status TEXT DEFAULT "normal"')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add price_category column to services table if it doesn't exist
    try:
        cursor.execute('ALTER TABLE services ADD COLUMN price_category TEXT DEFAULT "standard"')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # UTR tracking table to prevent duplicate UTR usage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS utr_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            utr_number TEXT UNIQUE,
            user_id INTEGER,
            used_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Razorpay payments tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS razorpay_payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            payment_link_id TEXT,
            payment_id TEXT,
            amount REAL,
            status TEXT DEFAULT 'pending',
            created_date TEXT,
            completed_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Support tickets tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS support_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            sent_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Settings table for UPI and profit margin
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Insert default UPI settings if not exists
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('upi_id', 'your-upi@bank')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('upi_name', 'Your Name')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('profit_margin', '1.5')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('referral_commission', '10')")  # 10% commission on orders
    
    # Razorpay settings
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('razorpay_enabled', 'false')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('razorpay_key_id', '')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('razorpay_key_secret', '')")
    
    # Service snapshots table for tracking service changes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS service_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_id INTEGER,
            name TEXT,
            supplier_rate REAL,
            category TEXT,
            snapshot_date TEXT,
            FOREIGN KEY (service_id) REFERENCES services(id)
        )
    ''')
    
    # User service notifications table to avoid duplicate alerts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_service_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            service_id INTEGER,
            notified_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (service_id) REFERENCES services(id),
            UNIQUE(user_id, service_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


# ======================== DATABASE HELPERS ========================

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect('fastxhub.db')


def register_user(user_id: int, username: str, referred_by: int = None) -> bool:
    """Register a new user or update existing one. Returns True if new user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
    if cursor.fetchone() is None:
        cursor.execute(
            'INSERT INTO users (id, username, wallet, join_date, is_blocked, referred_by) VALUES (?, ?, ?, ?, ?, ?)',
            (user_id, username, 0.0, datetime.now().isoformat(), 0, referred_by)
        )
        conn.commit()
        logger.info(f"New user registered: {user_id} (@{username}), referred by: {referred_by}")
        conn.close()
        return True  # New user
    
    conn.close()
    return False  # Existing user


def is_user_blocked(user_id: int) -> bool:
    """Check if user is blocked"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT is_blocked FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] == 1 if result else False


def block_user(user_id: int, reason: str = "Violation of terms") -> bool:
    """Block a user with reason"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE users SET is_blocked = 1, block_reason = ? WHERE id = ?', (reason, user_id))
        conn.commit()
        conn.close()
        logger.info(f"User blocked: {user_id} - Reason: {reason}")
        return True
    except Exception as e:
        logger.error(f"Failed to block user {user_id}: {e}")
        conn.close()
        return False


def unblock_user(user_id: int) -> bool:
    """Unblock a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE users SET is_blocked = 0, block_reason = NULL WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        logger.info(f"User unblocked: {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to unblock user {user_id}: {e}")
        conn.close()
        return False


def get_block_reason(user_id: int) -> Optional[str]:
    """Get block reason for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT block_reason FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else None


def get_user_wallet(user_id: int) -> float:
    """Get user's wallet balance"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT wallet FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0.0


def update_user_wallet(user_id: int, amount: float) -> None:
    """Update user's wallet balance"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE users SET wallet = wallet + ? WHERE id = ?',
        (amount, user_id)
    )
    conn.commit()
    conn.close()
    logger.info(f"Wallet updated for user {user_id}: {amount:+.2f}")


def check_utr_exists(utr_number: str) -> Optional[Dict[str, Any]]:
    """Check if UTR number already exists and return user info"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT ut.user_id, ut.used_date, u.username 
        FROM utr_tracking ut
        JOIN users u ON ut.user_id = u.id
        WHERE ut.utr_number = ?
    ''', (utr_number,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'user_id': result[0],
            'used_date': result[1],
            'username': result[2]
        }
    return None


def add_utr_tracking(utr_number: str, user_id: int) -> bool:
    """Add UTR number to tracking"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO utr_tracking (utr_number, user_id, used_date)
            VALUES (?, ?, ?)
        ''', (utr_number, user_id, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def check_support_ticket_limit(user_id: int) -> tuple[bool, int]:
    """
    Check if user has exceeded daily support ticket limit
    Returns: (can_send, tickets_sent_today)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get today's date
    today = datetime.now().date().isoformat()
    
    # Count tickets sent today
    cursor.execute('''
        SELECT COUNT(*) FROM support_tickets 
        WHERE user_id = ? AND DATE(sent_date) = ?
    ''', (user_id, today))
    
    count = cursor.fetchone()[0]
    conn.close()
    
    # Blocked users limited to 3 tickets per day
    max_tickets = 3
    return count < max_tickets, count


def add_support_ticket(user_id: int, message: str) -> bool:
    """Record a support ticket"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO support_tickets (user_id, message, sent_date)
            VALUES (?, ?, ?)
        ''', (user_id, message, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to record support ticket: {e}")
        conn.close()
        return False


def get_setting(key: str, default: str = '') -> str:
    """Get a setting value"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else default


def set_setting(key: str, value: str) -> None:
    """Set a setting value"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()
    logger.info(f"Setting updated: {key} = {value}")


# ======================== RAZORPAY HELPERS ========================

def get_razorpay_client():
    """Get Razorpay client instance"""
    key_id = get_setting('razorpay_key_id', '')
    key_secret = get_setting('razorpay_key_secret', '')
    
    if not key_id or not key_secret:
        return None
    
    return razorpay.Client(auth=(key_id, key_secret))


def is_razorpay_enabled() -> bool:
    """Check if Razorpay is enabled"""
    return get_setting('razorpay_enabled', 'false').lower() == 'true'


def create_razorpay_payment_link(user_id: int, amount: float, user_name: str) -> Optional[Dict]:
    """Create a Razorpay payment link"""
    try:
        client = get_razorpay_client()
        if not client:
            logger.error("Razorpay client not configured")
            return None
        
        # Amount must be in paise (1 rupee = 100 paise)
        amount_paise = int(amount * 100)
        
        # Create payment link
        payment_link = client.payment_link.create({
            "amount": amount_paise,
            "currency": "INR",
            "description": f"Add Funds - SVXHUB",
            "customer": {
                "name": user_name,
                "contact": "",
                "email": ""
            },
            "notify": {
                "sms": False,
                "email": False
            },
            "reminder_enable": False,
            "callback_url": f"https://t.me/{BOT_USERNAME}",
            "callback_method": "get"
        })
        
        logger.info(f"Created Razorpay payment link for user {user_id}: {payment_link['short_url']}")
        return payment_link
        
    except Exception as e:
        logger.error(f"Failed to create Razorpay payment link: {e}")
        return None


def verify_razorpay_payment(payment_id: str) -> Optional[Dict]:
    """Verify Razorpay payment status"""
    try:
        client = get_razorpay_client()
        if not client:
            return None
        
        payment = client.payment.fetch(payment_id)
        return payment
        
    except Exception as e:
        logger.error(f"Failed to verify Razorpay payment: {e}")
        return None


def generate_payment_code(user_id: int) -> str:
    """
    Generate unique payment verification code for user
    Format: SVXHUB + random 6 digits
    """
    code = f"SVXHUB{random.randint(100000, 999999)}"
    logger.info(f"Generated payment code for user {user_id}: {code}")
    return code


def get_user_info(user_id: int) -> Optional[Dict]:
    """Get user information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'username': row[1],
            'wallet': row[2],
            'join_date': row[3]
        }
    return None


# ======================== API HELPERS ========================

def call_supplier_api(action: str, params: Dict = None) -> Dict[str, Any]:
    """Call Xmedia SMM API"""
    try:
        data = {
            'key': SUPPLIER_API_KEY,
            'action': action
        }
        if params:
            data.update(params)
        
        response = requests.post(SUPPLIER_API_URL, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {'error': str(e)}


def get_supplier_balance() -> float:
    """Get supplier account balance"""
    result = call_supplier_api('balance')
    try:
        return float(result.get('balance', 0))
    except (ValueError, TypeError):
        return 0.0


def sync_services() -> int:
    """Sync services from supplier API and apply profit margin"""
    logger.info("Starting service sync...")
    result = call_supplier_api('services')
    
    if 'error' in result or not isinstance(result, list):
        logger.error(f"Failed to fetch services: {result}")
        return 0
    
    # Get profit margin from settings
    profit_margin = float(get_setting('profit_margin', '1.5'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    synced_count = 0
    
    for service in result:
        try:
            service_id = int(service['service'])
            name = service['name']
            supplier_rate = float(service['rate'])
            sell_rate = round(supplier_rate * profit_margin, 2)
            category = service.get('category', 'Other')
            min_qty = int(service.get('min', 1))
            max_qty = int(service.get('max', 100000))
            
            cursor.execute('''
                INSERT OR REPLACE INTO services 
                (id, name, supplier_rate, sell_rate, category, status, min_qty, max_qty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (service_id, name, supplier_rate, sell_rate, category, 'active', min_qty, max_qty))
            synced_count += 1
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to sync service: {e}")
            continue
    
    conn.commit()
    
    # Auto-categorize services by price (Cheapest/Premium)
    auto_categorize_services_by_price(cursor)
    conn.commit()
    
    conn.close()
    logger.info(f"Service sync completed: {synced_count} services synced")
    
    return synced_count


def auto_categorize_services_by_price(cursor) -> None:
    """Auto-categorize services into Cheapest, Premium, or Standard based on price within each category"""
    try:
        # Get all distinct categories
        cursor.execute('SELECT DISTINCT category FROM services WHERE status = "active"')
        categories = [cat[0] for cat in cursor.fetchall()]
        
        for category in categories:
            # Get all services in this category sorted by price
            cursor.execute('''
                SELECT id, sell_rate FROM services 
                WHERE category = ? AND status = "active" 
                ORDER BY sell_rate ASC
            ''', (category,))
            
            services = cursor.fetchall()
            
            if not services or len(services) < 3:
                # Not enough services to categorize, mark all as standard
                for service_id, _ in services:
                    cursor.execute('UPDATE services SET price_category = "standard" WHERE id = ?', (service_id,))
                continue
            
            # Calculate thresholds (bottom 30% = cheapest, top 30% = premium)
            total_count = len(services)
            cheapest_count = max(1, int(total_count * 0.3))
            premium_count = max(1, int(total_count * 0.3))
            
            # Categorize services
            for idx, (service_id, _) in enumerate(services):
                if idx < cheapest_count:
                    price_cat = 'cheapest'
                elif idx >= (total_count - premium_count):
                    price_cat = 'premium'
                else:
                    price_cat = 'standard'
                
                cursor.execute('UPDATE services SET price_category = ? WHERE id = ?', (price_cat, service_id))
        
        logger.info("Services auto-categorized by price")
        
    except Exception as e:
        logger.error(f"Error in auto_categorize_services_by_price: {e}")


def detect_new_services() -> List[int]:
    """
    Detect new services by comparing current services with the latest snapshot.
    Returns list of new service IDs.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the date of the most recent snapshot
        cursor.execute('SELECT MAX(snapshot_date) FROM service_snapshots')
        last_snapshot_date = cursor.fetchone()[0]
        
        # Get all service IDs from the latest snapshot
        if last_snapshot_date:
            cursor.execute('''
                SELECT DISTINCT service_id FROM service_snapshots 
                WHERE snapshot_date = ?
            ''', (last_snapshot_date,))
            snapshot_service_ids = set(row[0] for row in cursor.fetchall())
        else:
            snapshot_service_ids = set()
        
        # Get all current active service IDs
        cursor.execute('SELECT id FROM services WHERE status = "active"')
        current_service_ids = set(row[0] for row in cursor.fetchall())
        
        # Find new services (in current but not in snapshot)
        new_service_ids = current_service_ids - snapshot_service_ids
        
        # Create a new snapshot for all current services
        snapshot_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for service_id in current_service_ids:
            cursor.execute('''
                SELECT name, supplier_rate, category FROM services WHERE id = ?
            ''', (service_id,))
            row = cursor.fetchone()
            if row:
                cursor.execute('''
                    INSERT INTO service_snapshots 
                    (service_id, name, supplier_rate, category, snapshot_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (service_id, row[0], row[1], row[2], snapshot_date))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Service detection completed: {len(new_service_ids)} new services found")
        return list(new_service_ids)
        
    except Exception as e:
        logger.error(f"Error in detect_new_services: {e}")
        return []


def notify_users_about_new_service(bot, service_id: int) -> int:
    """
    Notify ALL users about a new service.
    Returns the number of users notified.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the new service details
        cursor.execute('''
            SELECT id, name, supplier_rate, sell_rate, category, min_qty, max_qty 
            FROM services WHERE id = ?
        ''', (service_id,))
        service_row = cursor.fetchone()
        
        if not service_row:
            conn.close()
            return 0
        
        service = {
            'id': service_row[0],
            'name': service_row[1],
            'supplier_rate': service_row[2],
            'sell_rate': service_row[3],
            'category': service_row[4],
            'min_qty': service_row[5],
            'max_qty': service_row[6]
        }
        
        # Get ALL users (excluding admin)
        cursor.execute('''
            SELECT user_id FROM users WHERE user_id != ?
        ''', (ADMIN_ID,))
        
        all_users = cursor.fetchall()
        notified_count = 0
        
        for (user_id,) in all_users:
            # Check if we already notified this user about this service
            cursor.execute('''
                SELECT id FROM user_service_notifications 
                WHERE user_id = ? AND service_id = ?
            ''', (user_id, service_id))
            
            if cursor.fetchone():
                continue  # Already notified
            
            # Prepare notification message for all users
            message = f"""
🆕 <b>New Service Available!</b>

We've added a new service in the <b>{service['category']}</b> category:

📦 <b>Service:</b> {service['name']}
💰 <b>Price:</b> ₹{service['sell_rate']:.2f} per 1000
📊 <b>Min/Max:</b> {service['min_qty']}/{service['max_qty']}
🏷️ <b>Category:</b> {service['category']}

🚀 Order this service now and grow your social media!
            """
            
            try:
                # Send notification to user
                bot.send_message(
                    chat_id=user_id,
                    text=message.strip(),
                    parse_mode='HTML',
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("🛒 Order Now", callback_data=f"select_service_{service_id}")
                    ]])
                )
                
                # Record the notification
                notification_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('''
                    INSERT INTO user_service_notifications 
                    (user_id, service_id, notified_date)
                    VALUES (?, ?, ?)
                ''', (user_id, service_id, notification_date))
                
                notified_count += 1
                logger.info(f"Notified user {user_id} about new service {service_id}")
                
            except Exception as e:
                logger.error(f"Failed to notify user {user_id} about service {service_id}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"New service notification completed: {notified_count} users notified about service {service_id}")
        return notified_count
        
    except Exception as e:
        logger.error(f"Error in notify_users_about_new_service: {e}")
        return 0


def check_and_notify_new_services(bot):
    """
    Main function to check for new services and notify users.
    Called by the scheduler.
    """
    try:
        logger.info("Starting new service detection and notification...")
        
        # First sync services from supplier
        sync_services()
        
        # Detect new services
        new_service_ids = detect_new_services()
        
        if not new_service_ids:
            logger.info("No new services detected")
            return
        
        # Notify users about each new service
        total_notified = 0
        for service_id in new_service_ids:
            notified = notify_users_about_new_service(bot, service_id)
            total_notified += notified
        
        # Send summary to admin
        try:
            bot.send_message(
                chat_id=ADMIN_ID,
                text=f"🆕 <b>New Service Alert</b>\n\n{len(new_service_ids)} new service(s) detected\n{total_notified} user(s) notified",
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send summary to admin: {e}")
        
        logger.info(f"New service check completed: {len(new_service_ids)} new services, {total_notified} users notified")
        
    except Exception as e:
        logger.error(f"Error in check_and_notify_new_services: {e}")


def get_service_by_id(service_id: int) -> Optional[Dict]:
    """Get service details by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM services WHERE id = ?', (service_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'name': row[1],
            'supplier_rate': row[2],
            'sell_rate': row[3],
            'category': row[4],
            'status': row[5],
            'min_qty': row[6],
            'max_qty': row[7]
        }
    return None


def create_order(user_id: int, service_id: int, qty: int, link: str) -> Dict[str, Any]:
    """Create a new order"""
    # Check if service exists
    service = get_service_by_id(service_id)
    if not service:
        return {'success': False, 'error': 'Service not found'}
    
    # Check quantity limits
    if qty < service['min_qty'] or qty > service['max_qty']:
        return {
            'success': False,
            'error': f"Quantity must be between {service['min_qty']} and {service['max_qty']}"
        }
    
    # Calculate total price
    total_price = service['sell_rate'] * qty
    
    # Check user wallet
    user_wallet = get_user_wallet(user_id)
    if user_wallet < total_price:
        return {
            'success': False,
            'error': f"Insufficient balance. Required: ₹{total_price:.2f}, Available: ₹{user_wallet:.2f}"
        }
    
    # Check supplier balance
    supplier_balance = get_supplier_balance()
    supplier_cost = service['supplier_rate'] * qty
    if supplier_balance < supplier_cost:
        return {
            'success': False,
            'error': 'Supplier balance is low. Please contact admin.'
        }
    
    # Place order with supplier
    api_result = call_supplier_api('add', {
        'service': service_id,
        'link': link,
        'quantity': qty
    })
    
    if 'error' in api_result:
        return {'success': False, 'error': api_result['error']}
    
    supplier_order_id = api_result.get('order')
    if not supplier_order_id:
        return {'success': False, 'error': 'Failed to place order with supplier'}
    
    # Deduct from user wallet
    update_user_wallet(user_id, -total_price)
    
    # Save order to database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO orders (user_id, service_id, qty, link, sell_price, supplier_order_id, status, created)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, service_id, qty, link, total_price, supplier_order_id, 'Pending', datetime.now().isoformat()))
    
    order_id = cursor.lastrowid
    
    # Check if user was referred and give commission to referrer
    cursor.execute('SELECT referred_by FROM users WHERE id = ?', (user_id,))
    referrer_row = cursor.fetchone()
    if referrer_row and referrer_row[0]:
        referrer_id = referrer_row[0]
        referral_commission = float(get_setting('referral_commission', '10'))
        commission_amount = total_price * (referral_commission / 100)
        
        # Add commission to referrer's wallet and earnings
        cursor.execute('UPDATE users SET wallet = wallet + ?, referral_earnings = referral_earnings + ? WHERE id = ?',
                      (commission_amount, commission_amount, referrer_id))
        
        logger.info(f"Referral commission: ₹{commission_amount:.2f} to user {referrer_id} for order {order_id}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Order created: {order_id} for user {user_id}")
    
    # Notify admin about new order
    try:
        # Get user info
        user_info = get_user_info(user_id)
        username = user_info.get('username', 'Unknown') if user_info else 'Unknown'
        
        # Create admin notification message
        admin_message = f"""
🆕 <b>NEW ORDER PLACED</b>

━━━━━━━━━━━━━━━━━━━━
👤 <b>Customer:</b> @{username} (ID: {user_id})
🆔 <b>Order ID:</b> #{order_id}
📦 <b>Service:</b> {service['name'][:50]}
🏷️ <b>Category:</b> {service['category']}
📊 <b>Quantity:</b> {qty:,}
💰 <b>Amount:</b> ₹{total_price:.2f}
🔗 <b>Supplier Order ID:</b> {supplier_order_id}
━━━━━━━━━━━━━━━━━━━━

✅ Order successfully placed!
        """
        
        # Send notification to admin
        from telegram import Bot
        bot = Bot(token=BOT_TOKEN)
        bot.send_message(
            chat_id=ADMIN_ID,
            text=admin_message.strip(),
            parse_mode='HTML'
        )
        logger.info(f"Admin notified about order {order_id}")
    except Exception as e:
        logger.error(f"Failed to notify admin about order {order_id}: {e}")
    
    return {
        'success': True,
        'order_id': order_id,
        'supplier_order_id': supplier_order_id,
        'total_price': total_price
    }


def check_order_status(order_id: int) -> Dict[str, Any]:
    """Check order status from database and update from API"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM orders WHERE id = ?', (order_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return {'success': False, 'error': 'Order not found'}
    
    order = {
        'id': row[0],
        'user_id': row[1],
        'service_id': row[2],
        'qty': row[3],
        'link': row[4],
        'sell_price': row[5],
        'supplier_order_id': row[6],
        'status': row[7],
        'created': row[8]
    }
    
    # Check status from API
    api_result = call_supplier_api('status', {'order': order['supplier_order_id']})
    
    if 'error' not in api_result:
        new_status = api_result.get('status', 'Pending')
        
        # Update status in database
        if new_status != order['status']:
            cursor.execute('UPDATE orders SET status = ? WHERE id = ?', (new_status, order_id))
            conn.commit()
            order['status'] = new_status
            
            # Handle refund cases
            if new_status.lower() in ['canceled', 'refunded', 'cancelled']:
                # Refund to user wallet
                cursor.execute('SELECT user_id, sell_price FROM orders WHERE id = ?', (order_id,))
                user_id, sell_price = cursor.fetchone()
                update_user_wallet(user_id, sell_price)
                logger.info(f"Order {order_id} refunded: ₹{sell_price:.2f} to user {user_id}")
        
        # Add additional info from API
        order['charge'] = api_result.get('charge', 0)
        order['start_count'] = api_result.get('start_count', 0)
        order['remains'] = api_result.get('remains', 0)
    
    conn.close()
    order['success'] = True
    return order


def automated_order_status_checker(bot):
    """Automatically check and update status for all pending orders"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all orders that are not completed
        cursor.execute('''
            SELECT id, user_id, supplier_order_id, status, sell_price 
            FROM orders 
            WHERE status IN ('Pending', 'Processing', 'In progress', 'Partial')
            ORDER BY created DESC
        ''')
        
        pending_orders = cursor.fetchall()
        
        if not pending_orders:
            logger.info("No pending orders to check")
            conn.close()
            return
        
        logger.info(f"Checking status for {len(pending_orders)} pending orders...")
        
        updated_count = 0
        refunded_count = 0
        
        for order in pending_orders:
            order_id, user_id, supplier_order_id, old_status, sell_price = order
            
            try:
                # Check status from supplier API
                api_result = call_supplier_api('status', {'order': supplier_order_id})
                
                if 'error' in api_result:
                    logger.warning(f"Error checking order {order_id}: {api_result.get('error')}")
                    continue
                
                new_status = api_result.get('status', 'Pending')
                
                # Update status if changed
                if new_status != old_status:
                    cursor.execute('UPDATE orders SET status = ? WHERE id = ?', (new_status, order_id))
                    conn.commit()
                    updated_count += 1
                    
                    logger.info(f"Order {order_id} status updated: {old_status} -> {new_status}")
                    
                    # Notify user about status change
                    try:
                        status_message = f"📦 *Order Update*\n\n"
                        status_message += f"Order ID: `{order_id}`\n"
                        status_message += f"Status: *{new_status}*\n"
                        
                        if new_status.lower() == 'completed':
                            status_message += "\n✅ Your order has been completed successfully!"
                        elif new_status.lower() in ['canceled', 'cancelled', 'refunded']:
                            status_message += f"\n❌ Your order has been {new_status.lower()}.\n"
                            status_message += f"₹{sell_price:.2f} has been refunded to your wallet."
                        elif new_status.lower() == 'partial':
                            status_message += "\n⚠️ Your order was partially completed."
                            
                        bot.send_message(
                            chat_id=user_id,
                            text=status_message,
                            parse_mode='Markdown'
                        )
                    except Exception as e:
                        logger.error(f"Failed to notify user {user_id} about order {order_id}: {str(e)}")
                    
                    # Handle refund cases
                    if new_status.lower() in ['canceled', 'refunded', 'cancelled']:
                        update_user_wallet(user_id, sell_price)
                        refunded_count += 1
                        logger.info(f"Order {order_id} auto-refunded: ₹{sell_price:.2f} to user {user_id}")
                    
                    # Handle partial refunds
                    elif new_status.lower() == 'partial':
                        charge = api_result.get('charge', 0)
                        remains = api_result.get('remains', 0)
                        
                        # Calculate partial refund (for remaining quantity)
                        if remains > 0 and charge > 0:
                            # Get service rate to calculate refund
                            cursor.execute('SELECT qty FROM orders WHERE id = ?', (order_id,))
                            original_qty = cursor.fetchone()[0]
                            
                            if original_qty > 0:
                                refund_per_unit = sell_price / original_qty
                                partial_refund = refund_per_unit * remains
                                
                                update_user_wallet(user_id, partial_refund)
                                logger.info(f"Order {order_id} partial refund: ₹{partial_refund:.2f} to user {user_id}")
                
                # Small delay to avoid API rate limiting
                __import__('time').sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing order {order_id}: {str(e)}")
                continue
        
        conn.close()
        
        logger.info(f"Order status check complete: {updated_count} updated, {refunded_count} refunded")
        
    except Exception as e:
        logger.error(f"Error in automated order status checker: {str(e)}")


def refund_order(order_id: int) -> Dict[str, Any]:
    """Manually refund an order"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, sell_price, status FROM orders WHERE id = ?', (order_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return {'success': False, 'error': 'Order not found'}
    
    user_id, sell_price, status = row
    
    if status.lower() in ['refunded', 'cancelled', 'canceled']:
        conn.close()
        return {'success': False, 'error': 'Order already refunded'}
    
    # Update order status
    cursor.execute('UPDATE orders SET status = ? WHERE id = ?', ('Refunded', order_id))
    conn.commit()
    conn.close()
    
    # Refund to wallet
    update_user_wallet(user_id, sell_price)
    
    logger.info(f"Manual refund: Order {order_id}, ₹{sell_price:.2f} to user {user_id}")
    
    return {'success': True, 'amount': sell_price}


# ======================== BOT HANDLERS ========================

def start(update: Update, context: CallbackContext) -> None:
    """Handle /start command"""
    user = update.effective_user
    
    # Extract referral code from deep link
    referrer_id = None
    if context.args and len(context.args) > 0:
        try:
            referrer_id = int(context.args[0])
        except ValueError:
            pass
    
    # Check if user is blocked - show limited menu with only support
    if is_user_blocked(user.id):
        register_user(user.id, user.username or user.first_name, referrer_id)
        
        keyboard = [
            [InlineKeyboardButton("🧾 Support", callback_data='menu_support')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Check ticket limit
        can_send, tickets_sent = check_support_ticket_limit(user.id)
        tickets_remaining = 3 - tickets_sent
        
        # Get block reason
        block_reason = get_block_reason(user.id)
        reason_text = f"\n⚠️ Reason: {escape_markdown(block_reason)}\n" if block_reason else ""
        
        blocked_text = f"""
╔══════════════════════╗
   🚫 ACCOUNT BLOCKED
╚══════════════════════╝

Your account has been suspended.
{reason_text}
━━━━━━━━━━━━━━━━━━━━

📊 Daily Support Access:
   • Used: {tickets_sent}/3
   • Available: {tickets_remaining}

You can still reach our support team!

👇 Click below to contact us:
"""
        
        update.message.reply_text(
            blocked_text,
            reply_markup=reply_markup
        )
        return
    
    # Apply rate limiting (except for admins)
    if not is_admin(user.id):
        allowed, error_msg = check_rate_limit(user.id)
        if not allowed:
            update.message.reply_text(error_msg)
            return
    
    is_new_user = register_user(user.id, user.username or user.first_name, referrer_id)
    
    # Notify admin about new user
    if is_new_user:
        try:
            referral_text = ""
            if referrer_id:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT username FROM users WHERE id = ?', (referrer_id,))
                referrer = cursor.fetchone()
                conn.close()
                if referrer:
                    referral_text = f"\n👥 Referred by: @{referrer[0]} ({referrer_id})"
            
            context.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"""
🎉 NEW USER JOINED!

👤 User: @{user.username or user.first_name}
🆔 ID: {user.id}
📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{referral_text}
"""
            )
        except Exception as e:
            logger.error(f"Failed to notify admin about new user: {e}")
        
        # No welcome bonus - only commission on orders
        if referrer_id and referrer_id != user.id:
            # Just notify referrer about new referral
            try:
                context.bot.send_message(
                    chat_id=referrer_id,
                    text=f"""
🎉 NEW REFERRAL!

Someone joined using your referral link!

💰 You'll earn 10% commission on all their orders!

Keep sharing to earn more! 🚀
"""
                )
            except:
                pass
    
    keyboard = [
        [
            InlineKeyboardButton("🛍 New Order", callback_data='menu_order'),
            InlineKeyboardButton("📋 Services", callback_data='menu_services')
        ],
        [
            InlineKeyboardButton("🔥 Cheapest Services", callback_data='menu_cheapest_services')
        ],
        [
            InlineKeyboardButton("💰 Add Funds", callback_data='menu_addfund'),
            InlineKeyboardButton("🎁 Refer & Earn", callback_data='menu_referral')
        ],
        [
            InlineKeyboardButton("📦 My Orders", callback_data='menu_myorders'),
            InlineKeyboardButton("🧾 Support", callback_data='menu_support')
        ],
        [
            InlineKeyboardButton("👤 Profile", callback_data='menu_profile')
        ]
    ]
    
    if user.id == ADMIN_ID:
        keyboard.append([InlineKeyboardButton("⚙️ Admin Panel", callback_data='admin_panel')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = f"""
╔══════════════════════╗
    🌟 SVXHUB 🌟
╚══════════════════════╝

Welcome back, {user.first_name}! ✨

━━━━━━━━━━━━━━━━━━━━
💎 Premium SMM Services
━━━━━━━━━━━━━━━━━━━━

✓ Lightning Fast Delivery
✓ Competitive Pricing  
✓ 24/7 Customer Support
✓ 100% Secure & Reliable

👇 Choose an option to continue:
"""
    
    update.message.reply_text(
        welcome_text,
        reply_markup=reply_markup
    )


def button_handler(update: Update, context: CallbackContext) -> Optional[int]:
    """Handle inline keyboard button callbacks"""
    query = update.callback_query
    query.answer()
    
    user_id = query.from_user.id
    data = query.data
    
    logger.info(f"button_handler called: user_id={user_id}, data={data}")
    
    # Check if user is blocked (except admins and support access)
    if not is_admin(user_id) and is_user_blocked(user_id):
        # Allow blocked users to access support menu and send support messages
        if data not in ['menu_support', 'send_support', 'back_to_main']:
            try:
                # Get block reason
                block_reason = get_block_reason(user_id)
                reason_text = f"\nReason: {escape_markdown(block_reason)}\n" if block_reason else ""
                
                keyboard = [[InlineKeyboardButton("🧾 Support", callback_data='menu_support')]]
                query.message.reply_text(
                    f"🚫 Access Denied\n\nYour account has been blocked.{reason_text}\nYou can only access Support.",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            except:
                pass
            return None
    
    # Apply rate limiting (except for admins)
    if not is_admin(user_id):
        allowed, error_msg = check_rate_limit(user_id)
        if not allowed:
            try:
                query.message.reply_text(error_msg)
            except:
                pass
            return None
    
    # Main menu handlers
    if data == 'menu_order':
        show_order_menu(query)
    elif data == 'menu_services':
        show_services(query, context)
    elif data == 'menu_cheapest_services':
        show_cheapest_services(query, context)
    elif data == 'menu_addfund':
        show_addfund_menu(query)
    elif data == 'menu_referral':
        show_referral_menu(query, user_id, context)
    elif data == 'upload_payment':
        # Generate unique payment verification code
        payment_code = generate_payment_code(user_id)
        
        # Store payment code in context for later use
        context.user_data['payment_code'] = payment_code
        
        # Get UPI details from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM settings WHERE key = ?', ('upi_id',))
        upi_id_row = cursor.fetchone()
        cursor.execute('SELECT value FROM settings WHERE key = ?', ('upi_name',))
        upi_name_row = cursor.fetchone()
        cursor.execute('SELECT value FROM settings WHERE key = ?', ('qr_code_file_id',))
        qr_code_row = cursor.fetchone()
        conn.close()
        
        upi_id = upi_id_row[0] if upi_id_row else "Not Set"
        upi_name = upi_name_row[0] if upi_name_row else "Not Set"
        qr_code_file_id = qr_code_row[0] if qr_code_row else None
        
        text = f"""
💰 Manual Payment - Step 1/3

━━━━━━━━━━━━━━━━━━━━

🔐 **IMPORTANT - Your Verification Code:**

`{payment_code}` (tap to copy)

⚠️ You MUST include this code in the payment notes/remarks field when making the UPI payment. This is required for automatic verification.

━━━━━━━━━━━━━━━━━━━━

📱 UPI Payment Details:
UPI ID: `{upi_id}` (tap to copy)
Name: {upi_name}

━━━━━━━━━━━━━━━━━━━━

Please enter the amount you want to add:

💵 Minimum: ₹1
💵 Maximum: ₹1000
✅ No extra charges

Example: 500

(Enter amount in numbers only)
After entering amount, you'll upload payment screenshot with the verification code visible.
"""
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='back_to_main')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # If QR code exists, send it as a photo with caption
        if qr_code_file_id:
            try:
                query.message.reply_photo(
                    photo=qr_code_file_id,
                    caption=text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                try:
                    query.message.delete()
                except:
                    pass
            except Exception as e:
                logger.error(f"Error sending QR code: {e}")
                # Fallback to text if QR code fails
                try:
                    query.message.delete()
                except:
                    pass
                query.message.reply_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            # No QR code, send text only
            try:
                query.message.delete()
            except:
                pass
            query.message.reply_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        
        return AWAITING_PAYMENT_AMOUNT
    elif data == 'razorpay_payment':
        # Handle Razorpay payment option
        text = """
💳 Razorpay Payment - Instant Credit

━━━━━━━━━━━━━━━━━━━━

Please enter the amount you want to pay:

💵 Minimum: ₹1
💵 Maximum: ₹1000

⚠️ 5% Convenience Fee Applied

Examples:
• Pay ₹100 → Get ₹95 credited
• Pay ₹500 → Get ₹475 credited
• Pay ₹1000 → Get ₹950 credited

(Enter amount in numbers only)

✨ You'll get a secure payment link instantly!
"""
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='back_to_main')]]
        try:
            query.message.delete()
        except Exception:
            pass
        query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        return AWAITING_RAZORPAY_AMOUNT
    elif data == 'menu_myorders':
        show_my_orders(query, user_id)
    elif data == 'menu_support':
        show_support_menu(query)
    elif data == 'send_support':
        text = (
            "📝 **Send Support Message**\n\n"
            "You can send:\n"
            "• 💬 Text message\n"
            "• 📷 Photo with caption\n"
            "• 📷 Photo only\n\n"
            "Example: I need help with my order"
        )
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='back_to_main')]]
        query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        return AWAITING_SUPPORT_MESSAGE
    elif data == 'menu_profile':
        show_profile(query, user_id)
    # VN disabled gate: block VN menu and sub-actions when feature is off
    elif (data == 'menu_virtual_number' or data.startswith('vn_')) and not VIRTUAL_NUMBER_ENABLED:
        try:
            query.answer("📵 Virtual Number service is temporarily unavailable.", show_alert=True)
        except Exception:
            pass
        return ConversationHandler.END
    elif data == 'menu_virtual_number':
        show_virtual_number_menu(query)
    elif data == 'vn_select_country':
        show_virtual_number_countries(query)
    elif data.startswith('vn_country_'):
        country_code = data.split('_')[2]
        show_virtual_number_services_menu(query, country_code)
    elif data.startswith('vn_service_'):
        parts = data.split('_')
        country_code = parts[2]
        service_code = parts[3]
        # Give immediate feedback to the user
        try:
            query.answer("⏳ Processing...", show_alert=False)
        except Exception:
            pass
        purchase_virtual_number(query, country_code, service_code, user_id)
    elif data.startswith('vn_status_'):
        activation_id = data.split('_')[2]
        check_virtual_number_status(query, activation_id)
    elif data.startswith('vn_cancel_'):
        activation_id = data.split('_')[2]
        cancel_virtual_number(query, activation_id)
    elif data == 'vn_my_numbers':
        show_my_virtual_numbers(query, user_id)
    elif data == 'menu_refill':
        show_refill_orders(query, user_id)
    elif data.startswith('refill_order_'):
        order_id = int(data.split('_')[2])
        process_order_refill(query, order_id, user_id)
    elif data.startswith('check_razorpay_'):
        check_razorpay_payment_status(query, data)
    
    # Admin wallet control handlers
    elif data.startswith('wallet_add_') and user_id == ADMIN_ID:
        target_user_id = int(data.replace('wallet_add_', ''))
        context.user_data['wallet_target_user'] = target_user_id
        query.message.reply_text(
            f"💰 **Add Money to Wallet**\n\n"
            f"User ID: {target_user_id}\n\n"
            f"Enter amount to add (in ₹):",
            parse_mode='Markdown'
        )
        return AWAITING_WALLET_ADD_AMOUNT
    
    elif data.startswith('wallet_deduct_') and user_id == ADMIN_ID:
        target_user_id = int(data.replace('wallet_deduct_', ''))
        context.user_data['wallet_target_user'] = target_user_id
        query.message.reply_text(
            f"💰 **Deduct Money from Wallet**\n\n"
            f"User ID: {target_user_id}\n\n"
            f"Enter amount to deduct (in ₹):",
            parse_mode='Markdown'
        )
        return AWAITING_WALLET_DEDUCT_AMOUNT
    
    elif data == 'back_to_main':
        show_main_menu(query)
    
    # Admin fund approval/rejection handlers
    elif data.startswith('approve_fund_') and user_id == ADMIN_ID:
        approve_fund_request(query, data)
    elif data.startswith('reject_fund_') and user_id == ADMIN_ID:
        reject_fund_request(query, data)
    
    # User list pagination
    elif data.startswith('userlist_page_') and user_id == ADMIN_ID:
        try:
            page = int(data.replace('userlist_page_', ''))
            show_user_list(query, page)
        except ValueError:
            query.answer("Invalid page number")
    
    # Admin panel handlers
    elif data == 'admin_panel' and user_id == ADMIN_ID:
        show_admin_panel(query)
    elif data == 'admin_check_orders' and user_id == ADMIN_ID:
        # Handle check orders button
        query.edit_message_text("🔄 Checking all pending orders...\n\nPlease wait...")
        try:
            automated_order_status_checker(query.bot)
            query.edit_message_text(
                "✅ Order status check completed!\n\nAll pending orders have been checked and updated.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Back to Admin Panel", callback_data='admin_panel')
                ]])
            )
        except Exception as e:
            logger.error(f"Error in manual order check: {e}")
            query.edit_message_text(
                f"❌ Error checking orders: {str(e)}",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Back to Admin Panel", callback_data='admin_panel')
                ]])
            )
    elif data.startswith('admin_') and user_id == ADMIN_ID:
        result = handle_admin_action(query, data)
        if result:
            return result
    elif data in ['set_razorpay_key_id', 'set_razorpay_key_secret', 'razorpay_toggle', 'upload_qr_code'] and user_id == ADMIN_ID:
        result = handle_admin_action(query, data)
        if result:
            return result
    # Route analytics and service status callbacks to admin handler
    elif data in ['analytics_dashboard', 'analytics_top_services', 'analytics_top_users', 'export_users', 'export_orders', 'service_status_manager'] and user_id == ADMIN_ID:
        result = handle_admin_action(query, data)
        if result:
            return result
    
    # Service category handlers using index
    # Platform selection handlers
    elif data.startswith('platform_'):
        platform = data.replace('platform_', '')
        show_platform_subcategories(query, platform, context)
    
    # Subcategory selection handlers (short format: sc_ins_0)
    elif data.startswith('sc_'):
        try:
            subcat_map = context.user_data.get('subcat_map', {})
            if data in subcat_map:
                platform, subcategory = subcat_map[data]
                show_subcategory_services(query, platform, subcategory, context)
            else:
                query.answer("Please select platform again")
        except Exception as e:
            query.answer("Error loading services")
    
    # Price filter handlers (filter_cheapest_instagram_followers, filter_premium_facebook_likes, etc.)
    elif data.startswith('filter_'):
        try:
            parts = data.replace('filter_', '').split('_', 2)  # Split into: [filter_type, platform, subcategory]
            if len(parts) >= 3:
                filter_type = parts[0]  # cheapest, premium, or none
                platform = parts[1]
                subcategory = parts[2]
                
                price_filter = None if filter_type == 'none' else filter_type
                show_subcategory_services(query, platform, subcategory, context, price_filter)
        except Exception as e:
            query.answer("Error applying filter")
    
    # Old subcategory handler for backward compatibility
    elif data.startswith('subcat_'):
        try:
            parts = data.replace('subcat_', '').split('_')
            if len(parts) >= 2:
                platform = parts[0]
                subcategory = '_'.join(parts[1:])  # Handle subcategories with underscores
                show_subcategory_services(query, platform, subcategory, context)
        except Exception as e:
            query.answer("Error loading services")
    
    # Old handlers for backward compatibility
    elif data.startswith('maincat_'):
        try:
            idx = int(data.replace('maincat_', ''))
            platforms = ['instagram', 'facebook', 'youtube', 'telegram']
            if idx < len(platforms):
                show_platform_subcategories(query, platforms[idx], context)
        except (ValueError, IndexError):
            query.answer("Error loading category")
    
    elif data == 'cat_other':
        show_platform_subcategories(query, 'other', context)
    
    elif data.startswith('othercat_'):
        try:
            idx = int(data.replace('othercat_', ''))
            category = context.user_data.get('other_category_map', {}).get(idx)
            if category:
                show_category_services(query, category)
            else:
                show_services(query, context)
        except (ValueError, KeyError):
            query.answer("Error loading category")
    
    elif data.startswith('cat_'):
        try:
            idx = int(data.replace('cat_', ''))
            category = context.user_data.get('category_map', {}).get(idx)
            if category:
                show_category_services(query, category)
            else:
                show_services(query, context)
        except (ValueError, KeyError):
            query.answer("Error loading category")
    
    # Service details handler (new shorter format)
    elif data.startswith('srv_'):
        service_id = int(data.replace('srv_', ''))
        show_service_details(query, service_id, context)
    
    # Cheapest services pagination handler
    elif data.startswith('cheapest_page_'):
        try:
            page = int(data.replace('cheapest_page_', ''))
            show_cheapest_services(query, context, page)
        except ValueError:
            query.answer("Error loading page")
    
    # Place order handler
    elif data.startswith('placeorder_'):
        service_id = int(data.replace('placeorder_', ''))
        context.user_data['order_service_id'] = service_id
        service = get_service_by_id(service_id)
        if service:
            text = f"""
╔══════════════════════╗
    🛒 PLACE ORDER
╚══════════════════════╝

📱 {service['name'][:60]}

━━━━━ Order Details ━━━━━

💰 Price: ₹{service['sell_rate']:.2f} per 1K
📊 Min: {service['min_qty']:,}
📈 Max: {service['max_qty']:,}

━━━━━━━━━━━━━━━━━━━━

📝 Enter quantity you want:
   (Numbers only)
"""
            keyboard = [[InlineKeyboardButton("❌ Cancel Order", callback_data='back_to_main')]]
            query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
            return AWAITING_ORDER_QUANTITY
    
    # Old service format (for backwards compatibility)
    # BUT exclude service_status_ which is handled elsewhere for admin
    elif data.startswith('service_') and not data.startswith('service_status_') and not data.startswith('set_status_'):
        service_id = int(data.replace('service_', ''))
        show_service_details(query, service_id, context)
    
    return ConversationHandler.END


def show_main_menu(query) -> None:
    """Show main menu"""
    keyboard = [
        [
            InlineKeyboardButton("🛍 New Order", callback_data='menu_order'),
            InlineKeyboardButton("📋 Services", callback_data='menu_services')
        ],
        [
            InlineKeyboardButton("🔥 Cheapest Services", callback_data='menu_cheapest_services')
        ],
        [
            InlineKeyboardButton("💰 Add Funds", callback_data='menu_addfund'),
            InlineKeyboardButton("🔄 Refill Order", callback_data='menu_refill')
        ]
    ]

    # Orders + Virtual Number row (conditionally include VN)
    row = [InlineKeyboardButton("📦 My Orders", callback_data='menu_myorders')]
    if VIRTUAL_NUMBER_ENABLED:
        row.append(InlineKeyboardButton("📱 Virtual Number", callback_data='menu_virtual_number'))
    keyboard.append(row)

    keyboard.append([
        InlineKeyboardButton("🧾 Support", callback_data='menu_support'),
        InlineKeyboardButton("👤 Profile", callback_data='menu_profile')
    ])
    
    if query.from_user.id == ADMIN_ID:
        keyboard.append([InlineKeyboardButton("⚙️ Admin Panel", callback_data='admin_panel')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    menu_text = """
╔══════════════════════╗
      🏠 MAIN MENU
╚══════════════════════╝

Choose your desired service:
        """
    
    # Check if message has photo (QR code) - delete it and send new message
    try:
        if query.message.photo:
            query.message.delete()
            query.message.reply_text(menu_text, reply_markup=reply_markup)
        else:
            query.edit_message_text(menu_text, reply_markup=reply_markup)
    except Exception as e:
        # Fallback: just send a new message
        try:
            query.message.reply_text(menu_text, reply_markup=reply_markup)
        except:
            pass


def show_order_menu(query) -> None:
    """Show order placement instructions"""
    text = """
╔══════════════════════╗
    🛍️ NEW ORDER
╚══════════════════════╝

📱 Browse our premium services
💫 Select your desired category
⚡ Fast & reliable delivery

📂 Choose a service category:
"""
    
    keyboard = [[InlineKeyboardButton("📋 View All Services", callback_data='menu_services')],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_services(query, context=None) -> None:
    """Show main platform categories"""
    
    # Define main platforms
    main_platforms = {
        'Instagram': '📸',
        'Facebook': '📘', 
        'Youtube': '📺',
        'Telegram': '✈️',
        'Other': '📁'
    }
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT category FROM services WHERE status = "active" ORDER BY category')
    all_categories = [cat[0] for cat in cursor.fetchall()]
    conn.close()
    
    if not all_categories:
        query.edit_message_text(
            "❌ No services available.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data='back_to_main')]])
        )
        return
    
    # Group categories by platform
    platform_groups = {platform: [] for platform in main_platforms.keys()}
    
    for cat in all_categories:
        grouped = False
        for platform in ['Instagram', 'Facebook', 'Youtube', 'Telegram']:
            if cat.lower().startswith(platform.lower()):
                platform_groups[platform].append(cat)
                grouped = True
                break
        if not grouped:
            platform_groups['Other'].append(cat)
    
    # Store in context
    if context:
        context.user_data['platform_groups'] = platform_groups
    
    # Build keyboard
    keyboard = []
    for platform, emoji in main_platforms.items():
        if platform_groups[platform]:
            keyboard.append([InlineKeyboardButton(
                f"{emoji} {platform}", 
                callback_data=f'platform_{platform.lower()}'
            )])
    
    keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(
        """
╔══════════════════════╗
   🌐 SELECT PLATFORM
╚══════════════════════╝

Choose your social media platform:

━━━━━━━━━━━━━━━━━━━━
💎 All platforms available
⚡ Premium quality services
        """,
        reply_markup=reply_markup
    )


def show_cheapest_services(query, context=None, page: int = 1) -> None:
    """Show only the cheapest services across all categories"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all cheapest services (price_category = 'cheapest')
    cursor.execute('''
        SELECT id, name, sell_rate, category, min_qty, max_qty 
        FROM services 
        WHERE status = "active" AND price_category = "cheapest"
        ORDER BY sell_rate ASC
    ''')
    
    cheapest_services = cursor.fetchall()
    conn.close()
    
    if not cheapest_services:
        query.edit_message_text(
            "❌ No cheapest services available at the moment.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data='back_to_main')]])
        )
        return
    
    # Pagination
    services_per_page = 10
    total_services = len(cheapest_services)
    total_pages = (total_services + services_per_page - 1) // services_per_page
    
    start_idx = (page - 1) * services_per_page
    end_idx = start_idx + services_per_page
    page_services = cheapest_services[start_idx:end_idx]
    
    # Build text
    text = f"""
╔══════════════════════╗
  🔥 CHEAPEST SERVICES
╚══════════════════════╝

💰 Best Prices Guaranteed!
📊 Page {page}/{total_pages} ({total_services} services)

━━━━━━━━━━━━━━━━━━━━

"""
    
    for idx, service in enumerate(page_services, start=start_idx + 1):
        service_id, name, sell_rate, category, min_qty, max_qty = service
        
        # Truncate long names
        display_name = name[:40] + "..." if len(name) > 40 else name
        
        text += f"""
{idx}. {display_name}
   💰 ₹{sell_rate:.2f}/1k | 📁 {category}
   Min: {min_qty} | Max: {max_qty}

"""
    
    text += "\n━━━━━━━━━━━━━━━━━━━━\n👇 Select a service to order:"
    
    # Build keyboard with service buttons
    keyboard = []
    for service in page_services:
        service_id, name, sell_rate, category, min_qty, max_qty = service
        display_name = name[:35] + "..." if len(name) > 35 else name
        keyboard.append([InlineKeyboardButton(
            f"💰 {display_name} - ₹{sell_rate:.2f}",
            callback_data=f'srv_{service_id}'
        )])
    
    # Pagination buttons
    pagination_row = []
    if page > 1:
        pagination_row.append(InlineKeyboardButton("⬅️ Previous", callback_data=f'cheapest_page_{page-1}'))
    if page < total_pages:
        pagination_row.append(InlineKeyboardButton("Next ➡️", callback_data=f'cheapest_page_{page+1}'))
    
    if pagination_row:
        keyboard.append(pagination_row)
    
    keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    safe_edit_message_text(query, text, reply_markup=reply_markup)


def show_platform_subcategories(query, platform: str, context=None) -> None:
    """Show subcategories for a platform (e.g., Instagram Followers, Instagram Likes)"""
    
    # Try to get from context first, if not available, reload from database
    platform_groups = context.user_data.get('platform_groups', {}) if context else {}
    
    # If not in context, reload from database
    if not platform_groups or platform.capitalize() not in platform_groups:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT category FROM services WHERE status = "active" ORDER BY category')
        all_categories = [cat[0] for cat in cursor.fetchall()]
        conn.close()
        
        # Rebuild platform groups
        platform_groups = {'Instagram': [], 'Facebook': [], 'Youtube': [], 'Telegram': [], 'Other': []}
        for cat in all_categories:
            grouped = False
            for plat in ['Instagram', 'Facebook', 'Youtube', 'Telegram']:
                if cat.lower().startswith(plat.lower()):
                    platform_groups[plat].append(cat)
                    grouped = True
                    break
            if not grouped:
                platform_groups['Other'].append(cat)
        
        # Store in context for future use
        if context:
            context.user_data['platform_groups'] = platform_groups
    
    categories = platform_groups.get(platform.capitalize(), [])
    
    if not categories:
        query.answer("No services available for this platform")
        return
    
    # Extract subcategories (e.g., "Instagram Views" -> "Views", "Instagram Likes" -> "Likes")
    subcategories = {}
    for cat in categories:
        # Remove platform name to get the rest
        cat_without_platform = cat.replace(f"{platform.capitalize()} ", "", 1).strip()
        
        # Extract first word as subcategory (before space, », [, or other separator)
        import re
        match = re.match(r'^([A-Za-z]+)', cat_without_platform)
        if match:
            subcat = match.group(1).capitalize()
            if subcat not in subcategories:
                subcategories[subcat] = []
            subcategories[subcat].append(cat)
    
    # Store in context
    if context:
        context.user_data['current_platform'] = platform
        context.user_data['current_subcategories'] = subcategories
    
    # Emoji mapping for subcategories
    subcat_emojis = {
        'Followers': '👥',
        'Likes': '❤️',
        'Views': '👁️',
        'Comments': '💬',
        'Subscribers': '📊',
        'Members': '👤',
        'Shares': '🔄',
        'Saves': '💾',
        'Story': '📖',
        'Reel': '🎬',
        'Live': '🔴',
        'IGTV': '📺',
        'Posts': '📸',
        'Impressions': '📈',
        'Reach': '🌍'
    }
    
    # Build keyboard with short callback data
    keyboard = []
    if context:
        if 'subcat_map' not in context.user_data:
            context.user_data['subcat_map'] = {}
    
    for idx, subcat in enumerate(sorted(subcategories.keys())):
        emoji = subcat_emojis.get(subcat, '💫')
        # Use short callback data with index
        callback_key = f'sc_{platform[:3]}_{idx}'
        if context:
            context.user_data['subcat_map'][callback_key] = (platform, subcat)
        keyboard.append([InlineKeyboardButton(
            f"{emoji} {subcat}",
            callback_data=callback_key
        )])
    
    keyboard.append([InlineKeyboardButton("◀️ Back to Platforms", callback_data='menu_services')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    platform_emojis = {
        'instagram': '📷',  # Camera emoji
        'facebook': '📘',
        'youtube': '📺',
        'telegram': '✈️',
        'other': '📁'
    }
    
    emoji = platform_emojis.get(platform.lower(), '💫')
    
    query.edit_message_text(
        f"""
╔══════════════════════╗
   {emoji} {platform.upper()}
╚══════════════════════╝

Select service type:

━━━━━━━━━━━━━━━━━━━━
🎯 Choose what you need
        """,
        reply_markup=reply_markup)


def show_subcategory_services(query, platform: str, subcategory: str, context=None, price_filter: str = None) -> None:
    """Show all services in a subcategory with price filter option"""
    
    subcategories = context.user_data.get('current_subcategories', {}) if context else {}

    categories_to_show = subcategories.get(subcategory.capitalize(), [])
    
    # If not in context, reload from database
    if not categories_to_show:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT category FROM services WHERE status = "active" ORDER BY category')
        all_categories = [cat[0] for cat in cursor.fetchall()]
        conn.close()
        
        # Rebuild subcategories for this platform
        platform_categories = [cat for cat in all_categories if cat.lower().startswith(platform.lower())]
        
        # Extract subcategories - categories are like "Instagram Views", "Instagram Likes", etc.
        subcategories = {}
        for cat in platform_categories:
            # Remove platform name and extract first word as subcategory
            # e.g. "Instagram Views » Video" -> "Views"
            # e.g. "Instagram Likes Indian" -> "Likes"
            cat_without_platform = cat.replace(f"{platform.capitalize()} ", "", 1).strip()
            
            # Get first word as subcategory (before space, », [, or other separator)
            import re
            match = re.match(r'^([A-Za-z]+)', cat_without_platform)
            if match:
                subcat = match.group(1).capitalize()
                if subcat not in subcategories:
                    subcategories[subcat] = []
                subcategories[subcat].append(cat)
        
        # Store in context for future use
        if context:
            context.user_data['current_subcategories'] = subcategories
        
        # Try again to get categories
        categories_to_show = subcategories.get(subcategory.capitalize(), [])
    
    if not categories_to_show:
        query.answer("No services found in this category")
        return
    
    # Get all services from these categories with optional price filter
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in categories_to_show])
    
    if price_filter:
        query_sql = f'SELECT id, name, sell_rate, category, price_category FROM services WHERE category IN ({placeholders}) AND status = "active" AND price_category = ? ORDER BY sell_rate ASC'
        cursor.execute(query_sql, categories_to_show + [price_filter])
    else:
        query_sql = f'SELECT id, name, sell_rate, category, price_category FROM services WHERE category IN ({placeholders}) AND status = "active" ORDER BY sell_rate ASC'
        cursor.execute(query_sql, categories_to_show)
    
    services = cursor.fetchall()
    conn.close()
    
    if not services:
        query.answer("No services available with this filter")
        return
    
    # Store current view info for filter callbacks
    if context:
        context.user_data['current_view'] = {
            'platform': platform,
            'subcategory': subcategory,
            'filter': price_filter
        }
    
    # Add filter buttons at the top
    filter_keyboard = []
    if not price_filter:
        filter_keyboard.append([
            InlineKeyboardButton("💰 Cheapest", callback_data=f'filter_cheapest_{platform}_{subcategory}'),
            InlineKeyboardButton("💎 Premium", callback_data=f'filter_premium_{platform}_{subcategory}')
        ])
    else:
        filter_text = "💰 Cheapest" if price_filter == "cheapest" else "💎 Premium"
        filter_keyboard.append([
            InlineKeyboardButton(f"✅ {filter_text}", callback_data=f'filter_none_{platform}_{subcategory}')
        ])
    
    # Limit to 15 services (to fit with filter buttons)
    services_to_show = services[:15]
    
    # Build service buttons
    service_keyboard = []
    for service in services_to_show:
        service_id, name, rate, category, price_cat = service
        
        # Add emoji based on price category
        if price_cat == 'cheapest':
            prefix = "💰"
        elif price_cat == 'premium':
            prefix = "💎"
        else:
            prefix = "💫"
        
        button_text = f"{prefix} {name[:28]} • ₹{rate:.2f}"
        service_keyboard.append([InlineKeyboardButton(button_text, callback_data=f'srv_{service_id}')])
    
    # Combine keyboards
    keyboard = filter_keyboard + service_keyboard
    keyboard.append([InlineKeyboardButton("◀️ Back", callback_data=f'platform_{platform.lower()}')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Display text based on filter
    filter_info = ""
    if price_filter == "cheapest":
        filter_info = "\n💰 **Showing: CHEAPEST Services**"
    elif price_filter == "premium":
        filter_info = "\n💎 **Showing: PREMIUM Services**"
    
    query.edit_message_text(
        f"""
╔══════════════════════╗
   💫 {subcategory.upper()}
╚══════════════════════╝
{filter_info}

🌟 Quality Services Available
⚡ Instant Start • 💎 Best Prices

━━━ Services ({len(services_to_show)}/{len(services)}) ━━━
        """,
        reply_markup=reply_markup,
        parse_mode='Markdown')


def escape_markdown(text: str) -> str:
    """Escape markdown special characters"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def show_category_services(query, category: str) -> None:
    """Show services in a category sorted by demand"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, name, sell_rate FROM services WHERE category = ? AND status = "active"',
        (category,)
    )
    services = cursor.fetchall()
    conn.close()
    
    if not services:
        query.answer("No services in this category")
        return
    
    # Enhanced sorting by demand/popularity with better matching
    def get_priority(service_name):
        """Get priority score for sorting (lower = higher priority)"""
        name_lower = service_name.lower()
        
        # Priority 1: Followers (most popular)
        if any(word in name_lower for word in ['follower', 'متابع', 'متابعين']):
            return 1
        
        # Priority 2: Likes
        if any(word in name_lower for word in ['like', 'لايك', 'إعجاب']):
            return 2
        
        # Priority 3: Views
        if any(word in name_lower for word in ['view', 'مشاهد', 'مشاهدة']):
            return 3
        
        # Priority 4: Members/Subscribers
        if any(word in name_lower for word in ['member', 'subscriber', 'sub', 'عضو', 'مشترك']):
            return 4
        
        # Priority 5: Comments
        if any(word in name_lower for word in ['comment', 'تعليق']):
            return 5
        
        # Priority 6: Shares
        if any(word in name_lower for word in ['share', 'مشاركة']):
            return 6
        
        # Priority 7: Story/Reel
        if any(word in name_lower for word in ['story', 'reel', 'ستوري', 'قصة']):
            return 7
        
        # Priority 8: Impressions/Reach
        if any(word in name_lower for word in ['impression', 'reach', 'وصول']):
            return 8
        
        # Priority 9: Saves
        if any(word in name_lower for word in ['save', 'حفظ']):
            return 9
        
        # Priority 10: Watch Time
        if any(word in name_lower for word in ['watch', 'وقت']):
            return 10
        
        # Priority 11: Everything else
        return 999
    
    # Sort services by priority, then by name
    sorted_services = sorted(services, key=lambda x: (get_priority(x[1]), x[1].lower()))
    
    # Limit to 20 services
    sorted_services = sorted_services[:20]
    
    # Escape category name for markdown
    safe_category = escape_markdown(category[:50])
    text = f"""
╔══════════════════════╗
   🏷️ {safe_category.upper()}
╚══════════════════════╝

🌟 Premium Quality Services
⚡ Instant Start Available
💎 Best Market Prices

━━━━ Select Service ━━━━
"""
    
    keyboard = []
    for service in sorted_services:
        service_id, name, rate = service
        # Create cleaner button with emoji and price
        button_text = f"💫 {name[:30]} • ₹{rate:.2f}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f'srv_{service_id}')])
    
    keyboard.append([InlineKeyboardButton("◀️ Back to Categories", callback_data='menu_services')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_service_details(query, service_id: int, context=None) -> None:
    """Show detailed service information"""
    service = get_service_by_id(service_id)
    
    if not service:
        query.answer("Service not found")
        return
    
    safe_name = escape_markdown(service['name'][:80])
    safe_category = escape_markdown(service['category'][:50])
    
    text = f"""
╔══════════════════════╗
    📦 SERVICE INFO
╚══════════════════════╝

{safe_name}

━━━━━ Details ━━━━━

🆔 Service ID: {service['id']}
📂 Category: {safe_category}

━━━━━ Pricing ━━━━━

💰 Price: ₹{service['sell_rate']:.2f} per 1K
📊 Min Order: {service['min_qty']:,}
📈 Max Order: {service['max_qty']:,}

━━━━━━━━━━━━━━━━━━━━
⚡ Ready to order?
"""
    
    keyboard = [
        [InlineKeyboardButton("🛒 Place Order Now", callback_data=f'placeorder_{service_id}')],
        [InlineKeyboardButton("◀️ Back to Services", callback_data='menu_services')],
        [InlineKeyboardButton("🏠 Main Menu", callback_data='back_to_main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_addfund_menu(query) -> None:
    """Show add funds menu with both automatic and manual options"""
    
    # Get UPI details from database for manual option
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM settings WHERE key = ?', ('upi_id',))
    upi_id_row = cursor.fetchone()
    cursor.execute('SELECT value FROM settings WHERE key = ?', ('upi_name',))
    upi_name_row = cursor.fetchone()
    conn.close()
    
    upi_id = upi_id_row[0] if upi_id_row else "Not Set"
    upi_name = upi_name_row[0] if upi_name_row else "Not Set"
    
    # Check if Razorpay is enabled
    razorpay_enabled = is_razorpay_enabled()
    
    text = f"""
╔══════════════════════╗
    💰 ADD FUNDS
╚══════════════════════╝

Choose your payment method:

━━━━━ Option 1: Manual Payment ━━━━━

📱 UPI ID: {upi_id}
👤 Name: {upi_name}

✅ No extra charges
⏱️ Manual verification (few minutes)
💰 Max ₹1000 per transaction

Steps:
1️⃣ Pay via UPI
2️⃣ Enter amount
3️⃣ Upload screenshot

━━━━━ Option 2: Automatic Payment ━━━━━

💳 Pay with Razorpay
{"✅ Available" if razorpay_enabled else "❌ Currently Unavailable"}

✅ Instant auto-credit
⚡ Multiple payment methods
💰 5% convenience fee
💰 Max ₹1000 per transaction

Example: Pay ₹100 → Get ₹95 credited
         Pay ₹500 → Get ₹475 credited

👇 Choose your payment method:
"""
    
    keyboard = [
        [InlineKeyboardButton("📤 Manual Payment (No Fee)", callback_data='upload_payment')]
    ]
    
    if razorpay_enabled:
        keyboard.append([InlineKeyboardButton("💳 Razorpay (Instant + 5% Fee)", callback_data='razorpay_payment')])
    
    keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        query.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error showing add funds menu: {e}")
        try:
            query.message.reply_text(text, reply_markup=reply_markup)
        except:
            pass


def show_my_orders(query, user_id: int) -> None:
    """Show user's recent orders"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, service_id, qty, sell_price, status, created FROM orders WHERE user_id = ? ORDER BY created DESC LIMIT 10',
        (user_id,)
    )
    orders = cursor.fetchall()
    conn.close()
    
    if not orders:
        text = "📦 My Orders\n\nYou haven't placed any orders yet."
    else:
        text = "📦 My Orders (Latest 10)\n\n"
        for order in orders:
            order_id, service_id, qty, price, status, created = order
            text += f"🆔 {order_id} | Service: {service_id} | Qty: {qty}\n"
            text += f"💰 ₹{price:.2f} | Status: *{status}*\n"
            text += f"📅 {created[:10]}\n\n"
        
        text += "\nUse /status <order_id> to check order status."
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='back_to_main')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_refill_orders(query, user_id: int) -> None:
    """Show user's orders for refill selection"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, service_id, qty, sell_price, status, created FROM orders WHERE user_id = ? ORDER BY created DESC LIMIT 20',
        (user_id,)
    )
    orders = cursor.fetchall()
    conn.close()
    
    if not orders:
        text = "🔄 Refill Orders\n\nYou haven't placed any orders yet."
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='menu_profile')]]
    else:
        text = "🔄 Select Order to Refill\n\nClick on an order to refill it:\n\n"
        keyboard = []
        
        for order in orders:
            order_id, service_id, qty, price, status, created = order
            # Show order info with refill button
            btn_text = f"🆔 {order_id} | {status[:15]} | ₹{price:.2f}"
            keyboard.append([InlineKeyboardButton(btn_text, callback_data=f'refill_order_{order_id}')])
        
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='menu_profile')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    query.edit_message_text(text, reply_markup=reply_markup)


def process_order_refill(query, order_id: int, user_id: int) -> None:
    """Process order refill request"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get original order details
    cursor.execute(
        'SELECT service_id, qty, link, sell_price FROM orders WHERE id = ? AND user_id = ?',
        (order_id, user_id)
    )
    order = cursor.fetchone()
    
    if not order:
        query.answer("❌ Order not found!")
        return
    
    service_id, qty, link, sell_price = order
    
    # Check user balance
    cursor.execute('SELECT wallet FROM users WHERE id = ?', (user_id,))
    balance = cursor.fetchone()[0]
    
    if balance < sell_price:
        query.answer("❌ Insufficient balance!")
        conn.close()
        return
    
    # Deduct balance
    cursor.execute('UPDATE users SET wallet = wallet - ? WHERE id = ?', (sell_price, user_id))
    
    # Create refill order
    cursor.execute(
        'INSERT INTO orders (user_id, service_id, qty, link, sell_price, status) VALUES (?, ?, ?, ?, ?, ?)',
        (user_id, service_id, qty, link, sell_price, 'Pending')
    )
    new_order_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    # Try to place order with supplier API
    try:
        api_key = os.getenv('SUPPLIER_API_KEY')
        url = 'https://xmediasmm.in/api/v2'
        payload = {
            'key': api_key,
            'action': 'add',
            'service': service_id,
            'link': link,
            'quantity': qty
        }
        response = requests.post(url, data=payload, timeout=10)
        data = response.json()
        
        if 'order' in data:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE orders SET order_id = ? WHERE id = ?', (data['order'], new_order_id))
            conn.commit()
            conn.close()
            
            text = f"✅ Refill Order Placed!\n\n🆔 Order ID: {new_order_id}\n📦 Supplier ID: {data['order']}\n💰 Amount: ₹{sell_price:.2f}\n\n✨ Your order is being processed!"
        else:
            text = f"⚠️ Order Placed (Pending)\n\n🆔 Order ID: {new_order_id}\n💰 Amount: ₹{sell_price:.2f}\n\nWaiting for supplier confirmation..."
    except Exception as e:
        text = f"⚠️ Order Created\n\n🆔 Order ID: {new_order_id}\n💰 Amount: ₹{sell_price:.2f}\n\nProcessing may take a moment..."
    
    keyboard = [[InlineKeyboardButton("🔙 Back to Profile", callback_data='menu_profile')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_support_menu(query) -> None:
    """Show support instructions"""
    user_id = query.from_user.id
    
    # Check if user is blocked and show ticket limit
    if is_user_blocked(user_id):
        can_send, tickets_sent = check_support_ticket_limit(user_id)
        tickets_remaining = 3 - tickets_sent
        
        text = f"""
╔══════════════════════╗
    🧾 SUPPORT CENTER
╚══════════════════════╝

⚠️ Account Status: Blocked

━━━━━ Daily Ticket Limit ━━━━━

📊 Sent Today: {tickets_sent}/3
✅ Remaining: {tickets_remaining}

You can still contact support!
"""
        
        if not can_send:
            text += "\n🚫 Daily limit reached\n   Try again tomorrow"
            keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]]
        else:
            keyboard = [
                [InlineKeyboardButton("📝 Send Support Message", callback_data='send_support')],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]
            ]
    else:
        text = """
╔══════════════════════╗
    🧾 SUPPORT CENTER
╚══════════════════════╝

Need assistance? We're here to help!

━━━━━━━━━━━━━━━━━━━━

💬 Click below to send message
⚡ Fast response guaranteed
🔒 100% confidential

Our team will respond ASAP!
"""
        keyboard = [
            [InlineKeyboardButton("📝 Send Support Message", callback_data='send_support')],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]
        ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        query.edit_message_text(text, reply_markup=reply_markup)
    except:
        pass


def show_profile(query, user_id: int) -> None:
    """Show user profile"""
    user_info = get_user_info(user_id)
    
    if not user_info:
        query.answer("User not found")
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM orders WHERE user_id = ?', (user_id,))
    total_orders = cursor.fetchone()[0]
    cursor.execute('SELECT SUM(sell_price) FROM orders WHERE user_id = ?', (user_id,))
    total_spent = cursor.fetchone()[0] or 0
    conn.close()
    
    text = f"""
╔══════════════════════╗
     👤 YOUR PROFILE
╚══════════════════════╝

━━━━━ Account Details ━━━━━

🆔 User ID: {user_info['id']}
👤 Username: @{user_info['username']}
💰 Wallet: ₹{user_info['wallet']:.2f}
📅 Member Since: {user_info['join_date'][:10]}

━━━━━ Activity Stats ━━━━━

📦 Total Orders: {total_orders}
💳 Total Spent: ₹{total_spent:.2f}
"""
    
    keyboard = [
        [InlineKeyboardButton("🔄 Refill Order", callback_data='menu_refill')],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


# ======================== VIRTUAL NUMBER FUNCTIONS ========================

def get_virtual_number_balance() -> Optional[float]:
    """Get balance from virtual number API"""
    try:
        params = {
            'api_key': VIRTUAL_NUMBER_API_KEY,
            'action': 'getBalance'
        }
        response = vn_request(params)
        resp_text = response.text.strip()

        # Guard against HTML/404 or unexpected responses
        if response.status_code != 200 or resp_text.lower().startswith('<!doctype') or '<html' in resp_text[:200].lower():
            logger.error(f"Virtual number balance unexpected HTTP {response.status_code}: {resp_text[:200]}")
            return None

        if resp_text.startswith('ACCESS_BALANCE'):
            balance = float(resp_text.split(':')[1])
            return balance
        else:
            logger.error(f"Virtual number balance error: {resp_text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Error getting virtual number balance: {e}")
        return None


def get_virtual_number_countries() -> Dict:
    """Get available countries for virtual numbers"""
    return {
        '0': '🇷🇺 Russia',
        '1': '🇺🇦 Ukraine',
        '2': '🇰🇿 Kazakhstan',
        '3': '🇨🇳 China',
        '4': '🇵🇭 Philippines',
        '6': '🇮🇩 Indonesia',
        '7': '🇲🇾 Malaysia',
        '10': '🇻🇳 Vietnam',
        '16': '🇺🇸 USA',
        '22': '🇬🇧 UK',
        '36': '🇨🇦 Canada',
        '44': '🇮🇳 India',
        '77': '🇹🇭 Thailand',
    }


def get_virtual_number_services() -> Dict:
    """Get available services for virtual numbers"""
    return {
        'wa': '💚 WhatsApp',
        'tg': '✈️ Telegram',
        'ig': '📸 Instagram',
        'fb': '📘 Facebook',
        'tw': '🐦 Twitter',
        'go': '🔍 Google',
    }


def show_virtual_number_menu(query) -> None:
    """Show virtual number service menu"""
    # Try to fetch provider balance for visibility
    balance_text = ""
    if VIRTUAL_NUMBER_SHOW_PROVIDER_BALANCE:
        try:
            bal = get_virtual_number_balance()
            if bal is not None:
                balance_text = f"\n🏦 Provider Balance: ${bal:.2f}"
            else:
                balance_text = "\n🏦 Provider Balance: unavailable"
        except Exception:
            balance_text = "\n🏦 Provider Balance: error"

    text = f"""
📱 **Virtual Number Service**
━━━━━━━━━━━━━━━━━━━━

🌍 Get temporary phone numbers for verification

**How it works:**
1️⃣ Select country
2️⃣ Choose service (WhatsApp, Telegram, etc.)
3️⃣ Receive your virtual number
4️⃣ Get SMS code instantly

💰 **Pricing:** ₹10 - ₹500 per number
⚡ **Delivery:** Instant activation

━━━━━━━━━━━━━━━━━━━━
Click below to start:
{balance_text}
"""
    
    keyboard = [
        [InlineKeyboardButton("🌍 Select Country", callback_data='vn_select_country')],
        [InlineKeyboardButton("📱 My Active Numbers", callback_data='vn_my_numbers')],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')


def show_virtual_number_countries(query) -> None:
    """Show available countries for virtual numbers"""
    countries = get_virtual_number_countries()
    
    text = """
🌍 **Select Country**
━━━━━━━━━━━━━━━━━━━━

Choose a country for your virtual number:
"""
    
    keyboard = []
    # Show countries in rows of 2
    country_items = list(countries.items())
    for i in range(0, len(country_items), 2):
        row = []
        for code, name in country_items[i:i+2]:
            row.append(InlineKeyboardButton(name, callback_data=f'vn_country_{code}'))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='menu_virtual_number')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')


def show_virtual_number_services_menu(query, country_code: str) -> None:
    """Show available services for selected country"""
    services = get_virtual_number_services()
    countries = get_virtual_number_countries()
    country_name = countries.get(country_code, 'Unknown')
    
    text = f"""
📱 **Select Service**
━━━━━━━━━━━━━━━━━━━━

Country: {country_name}

Choose the service you need:
"""
    
    keyboard = []
    # Show services in rows of 2
    service_items = list(services.items())
    for i in range(0, len(service_items), 2):
        row = []
        for code, name in service_items[i:i+2]:
            row.append(InlineKeyboardButton(name, callback_data=f'vn_service_{country_code}_{code}'))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='vn_select_country')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')


def purchase_virtual_number(query, country_code: str, service_code: str, user_id: int) -> None:
    """Purchase a virtual number"""
    try:
        # Check user balance
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT wallet FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()

        if not result:
            query.answer("❌ User not found!", show_alert=True)
            conn.close()
            return

        user_balance = result[0]
        price_inr = VIRTUAL_NUMBER_DEFAULT_PRICE_INR  # Configurable via .env

        if user_balance < price_inr:
            query.answer(f"❌ Insufficient balance! Need ₹{price_inr:.2f}", show_alert=True)
            conn.close()
            return

        # Purchase number
        params = {
            'api_key': VIRTUAL_NUMBER_API_KEY,
            'action': 'getNumber',
            'service': service_code,
            'country': country_code
        }
        if VIRTUAL_NUMBER_OPERATOR:
            params['operator'] = VIRTUAL_NUMBER_OPERATOR

        response = vn_request(params)
        resp_text = response.text.strip()
        # Clamp logs to avoid spamming with large HTML bodies
        logger.info(f"VN getNumber response (HTTP {response.status_code}): {resp_text[:200]}")

        # Handle unexpected HTML/404 or non-OK
        if response.status_code != 200 or resp_text.lower().startswith('<!doctype') or '<html' in resp_text[:200].lower():
            query.answer("❌ Provider error (unexpected response). Please verify VIRTUAL_NUMBER_API_URL/API key.", show_alert=True)
            conn.close()
            return

        if resp_text.startswith('ACCESS_NUMBER'):
            # Format: ACCESS_NUMBER:activation_id:phone_number
            parts = resp_text.split(':')
            activation_id = parts[1]
            phone_number = parts[2]

            # Deduct from wallet
            new_balance = user_balance - price_inr
            cursor.execute('UPDATE users SET wallet = ? WHERE id = ?', (new_balance, user_id))

            # Store in database (match existing orders schema)
            # orders(user_id, service_id, qty, link, sell_price, supplier_order_id, status, created)
            cursor.execute('''INSERT INTO orders 
                (user_id, service_id, qty, link, sell_price, supplier_order_id, status, created)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    user_id,
                    None,  # virtual number not mapped to a service_id
                    1,
                    f'VN:{activation_id}',  # link used to reference activation
                    price_inr,
                    activation_id,
                    'Active',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))

            conn.commit()
            conn.close()

            countries = get_virtual_number_countries()
            services = get_virtual_number_services()

            text = f"""
✅ **Number Activated!**
━━━━━━━━━━━━━━━━━━━━

📱 **Number:** `+{phone_number}`
🌍 **Country:** {countries.get(country_code, 'Unknown')}
📲 **Service:** {services.get(service_code, 'Unknown')}
🆔 **Activation ID:** `{activation_id}`

💰 **Cost:** ₹{price_inr:.2f}
💵 **New Balance:** ₹{new_balance:.2f}

⏳ **Waiting for SMS...**
SMS will arrive within 20 minutes.
"""

            keyboard = [
                [InlineKeyboardButton("🔄 Check Status", callback_data=f'vn_status_{activation_id}')],
                [InlineKeyboardButton("❌ Cancel Number", callback_data=f'vn_cancel_{activation_id}')],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')

        else:
            conn.close()
            # Map common provider errors to friendly messages
            error_map = {
                'NO_BALANCE': 'Provider is out of balance. Please try later or contact admin.',
                'NO_NUMBERS': 'No numbers available for the selected country/service.',
                'BAD_KEY': 'Invalid API key. Admin should set VIRTUAL_NUMBER_API_KEY.',
                'BAD_ACTION': 'Provider rejected the request (bad action).',
                'BAD_SERVICE': 'Unsupported service code for this provider.',
                'BAD_COUNTRY': 'Unsupported country code for this provider.',
                'REQUEST_NOT_SUCCESS': 'Provider request failed. Please try again later.',
            }
            # Extract base error code up to first colon if present
            base_err = resp_text.split(':', 1)[0]
            friendly = error_map.get(base_err, "Provider error.")
            query.answer(f"❌ {friendly}", show_alert=True)

    except Exception as e:
        logger.error(f"Error purchasing virtual number: {e}")
        try:
            query.answer("❌ Error processing request. Please try again.", show_alert=True)
        except Exception:
            pass


def check_virtual_number_status(query, activation_id: str) -> None:
    """Check status of virtual number activation"""
    try:
        params = {
            'api_key': VIRTUAL_NUMBER_API_KEY,
            'action': 'getStatus',
            'id': activation_id
        }
        
        response = vn_request(params)
        resp_text = response.text.strip()
        logger.info(f"VN getStatus response (HTTP {response.status_code}): {resp_text[:200]}")

        # Handle unexpected HTML/404 or non-OK
        if response.status_code != 200 or resp_text.lower().startswith('<!doctype') or '<html' in resp_text[:200].lower():
            query.answer("❌ Provider error (unexpected response). Please try again later.", show_alert=True)
            return
        
        if resp_text.startswith('STATUS_OK'):
            # Format: STATUS_OK:sms_code
            sms_code = resp_text.split(':')[1]
            # Mark order as completed
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("UPDATE orders SET status = 'Completed' WHERE supplier_order_id = ? AND link LIKE 'VN:%'", (activation_id,))
                conn.commit()
                conn.close()
            except Exception as db_e:
                logger.error(f"Failed to update VN order status to Completed: {db_e}")
            
            text = f"""
✅ **SMS Received!**
━━━━━━━━━━━━━━━━━━━━

🔐 **Verification Code:** `{sms_code}`
🆔 **Activation ID:** `{activation_id}`

✨ Copy the code above and use it for verification!
"""
            
            keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')
            
        elif resp_text == 'STATUS_WAIT_CODE':
            query.answer("⏳ Waiting for SMS... Please try again in a moment.", show_alert=True)
        elif resp_text == 'STATUS_CANCEL':
            query.answer("❌ Activation was cancelled.", show_alert=True)
        else:
            query.answer(f"Status: {resp_text}", show_alert=True)
            
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        query.answer(f"❌ Error: {str(e)}", show_alert=True)


def cancel_virtual_number(query, activation_id: str) -> None:
    """Cancel virtual number activation"""
    try:
        params = {
            'api_key': VIRTUAL_NUMBER_API_KEY,
            'action': 'setStatus',
            'id': activation_id,
            'status': '8'  # Cancel status
        }
        
        response = vn_request(params)
        resp_text = response.text.strip()
        logger.info(f"VN setStatus(cancel) response (HTTP {response.status_code}): {resp_text[:200]}")

        # Handle unexpected HTML/404 or non-OK
        if response.status_code != 200 or resp_text.lower().startswith('<!doctype') or '<html' in resp_text[:200].lower():
            query.answer("❌ Provider error (unexpected response). Please try again later.", show_alert=True)
            return
        
        if resp_text == 'ACCESS_CANCEL':
            query.answer("✅ Number cancelled successfully!", show_alert=True)
            # Mark order as cancelled
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("UPDATE orders SET status = 'Cancelled' WHERE supplier_order_id = ? AND link LIKE 'VN:%'", (activation_id,))
                conn.commit()
                conn.close()
            except Exception as db_e:
                logger.error(f"Failed to update VN order status to Cancelled: {db_e}")
            show_virtual_number_menu(query)
        else:
            query.answer(f"❌ {resp_text}", show_alert=True)
            
    except Exception as e:
        logger.error(f"Error cancelling number: {e}")
        query.answer(f"❌ Error: {str(e)}", show_alert=True)


def show_my_virtual_numbers(query, user_id: int) -> None:
    """Show user's active virtual numbers"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Query orders where the link contains activation ID (format: VN:activation_id)
        cursor.execute('''
            SELECT supplier_order_id, link, created, status 
            FROM orders 
            WHERE user_id = ? AND link LIKE 'VN:%' 
            ORDER BY created DESC LIMIT 10
        ''', (user_id,))
        numbers = cursor.fetchall()
        conn.close()
        
        if not numbers:
            text = """
📱 **My Active Numbers**
━━━━━━━━━━━━━━━━━━━━

You don't have any virtual numbers yet.

Purchase a virtual number to receive SMS for:
• WhatsApp
• Telegram  
• Instagram
• Facebook
• And more...
"""
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='menu_virtual_number')]]
        else:
            text = """
📱 **My Virtual Numbers**
━━━━━━━━━━━━━━━━━━━━

Your recent virtual number orders:

"""
            keyboard = []
            
            for supplier_id, link, created, status in numbers:
                # Extract activation_id from link (format: VN:activation_id)
                activation_id = link.split(':', 1)[1] if ':' in link else link
                status_emoji = "✅" if status == "Completed" else "⏳" if status == "Pending" else "❌"
                text += f"{status_emoji} ID: `{activation_id}`\n🕒 {created}\n💬 Status: {status}\n\n"
                
                if status in ["Pending", "Active"]:
                    keyboard.append([
                        InlineKeyboardButton(f"📱 Check SMS {activation_id[:8]}", callback_data=f'vn_status_{activation_id}')
                    ])
            
            keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='menu_virtual_number')])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error showing virtual numbers: {e}")
        query.answer(f"❌ Error loading numbers", show_alert=True)


def show_referral_menu(query, user_id: int, context) -> None:
    """Show referral program details"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get referral stats
    cursor.execute('SELECT COUNT(*) FROM users WHERE referred_by = ?', (user_id,))
    total_referrals = cursor.fetchone()[0]
    
    cursor.execute('SELECT referral_earnings FROM users WHERE id = ?', (user_id,))
    earnings_row = cursor.fetchone()
    referral_earnings = earnings_row[0] if earnings_row and earnings_row[0] else 0.0
    
    conn.close()
    
    # Get referral settings
    referral_commission = float(get_setting('referral_commission', '10'))
    
    # Generate referral link
    bot_username = context.bot.get_me().username
    referral_link = f"https://t.me/{bot_username}?start={user_id}"
    
    text = f"""
╔══════════════════════╗
   🎁 REFER & EARN
╚══════════════════════╝

Invite friends and earn rewards!

━━━━━ Your Stats ━━━━━

👥 Total Referrals: {total_referrals}
💰 Total Earnings: ₹{referral_earnings:.2f}

━━━━━ Reward ━━━━━

💎 Commission: {referral_commission:.0f}%
   (On every order they place)

━━━━━ How It Works ━━━━━

1️⃣ Share your referral link
2️⃣ Friends join using your link
3️⃣ You earn {referral_commission:.0f}% on ALL their orders
4️⃣ Lifetime passive income!

━━━━━ Your Referral Link ━━━━━

{referral_link}

━━━━━━━━━━━━━━━━━━━━

📱 Share your link and start earning!
"""
    
    keyboard = [
        [InlineKeyboardButton("📋 Copy Link", url=referral_link)],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


# ======================== ADMIN HANDLERS ========================

def show_admin_panel(query) -> None:
    """Show admin panel"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Dashboard", callback_data='admin_dashboard'),
            InlineKeyboardButton("💰 Pending Funds", callback_data='admin_pending_funds')
        ],
        [
            InlineKeyboardButton("📦 Recent Orders", callback_data='admin_orders'),
            InlineKeyboardButton("👥 User List", callback_data='admin_user_list')
        ],
        [
            InlineKeyboardButton("💳 Wallet Control", callback_data='admin_wallet_control'),
            InlineKeyboardButton("🚫 Block/Unblock", callback_data='admin_block_user')
        ],
        [
            InlineKeyboardButton("💸 Refunds", callback_data='admin_refunds'),
            InlineKeyboardButton("📣 Broadcast", callback_data='admin_broadcast')
        ],
        [
            InlineKeyboardButton("🔄 Sync Services", callback_data='admin_sync'),
            InlineKeyboardButton("⚙️ Settings", callback_data='admin_settings')
        ],
        [
            InlineKeyboardButton("📈 Analytics", callback_data='analytics_dashboard'),
            InlineKeyboardButton("⚙️ Service Status", callback_data='service_status_manager')
        ],
        [
            InlineKeyboardButton("🔍 Check Orders", callback_data='admin_check_orders')
        ],
        [InlineKeyboardButton("🔙 Back", callback_data='back_to_main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(
        "⚙️ Admin Panel\n\nSelect an option:",
        reply_markup=reply_markup)


def handle_admin_action(query, data: str) -> Optional[int]:
    """Handle admin panel actions"""
    # Ensure we have the caller's user id for permission checks inside this function
    user_id = getattr(query.from_user, 'id', None)

    if data == 'admin_dashboard':
        show_admin_dashboard(query)
    elif data == 'admin_pending_funds':
        show_pending_funds(query)
    elif data == 'admin_orders':
        show_recent_orders(query)
    elif data == 'admin_user_list':
        show_user_list(query)
    elif data == 'admin_wallet_control':
        return show_wallet_control_menu(query)
    elif data == 'admin_refunds':
        show_refund_menu(query)
    elif data == 'admin_block_user':
        show_block_user_menu(query)
    elif data == 'admin_broadcast':
        show_broadcast_menu(query)
    elif data == 'admin_broadcast_start':
        keyboard = [[InlineKeyboardButton("🔙 Cancel", callback_data='admin_panel')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        query.edit_message_text(
            "📣 **Broadcast Message**\n\n"
            "You can send:\n"
            "• 💬 Text message\n"
            "• 📷 Photo with caption\n\n"
            "Send your broadcast message now:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        return AWAITING_BROADCAST_MESSAGE
    
    # Analytics Dashboard Handlers (Admin only)
    elif data == 'analytics_dashboard' and user_id == ADMIN_ID:
        show_analytics_dashboard(query)
    elif data == 'analytics_top_services' and user_id == ADMIN_ID:
        show_top_services(query)
    elif data == 'analytics_top_users' and user_id == ADMIN_ID:
        show_top_users(query)
    elif data == 'export_users' and user_id == ADMIN_ID:
        export_users_csv(query)
    elif data == 'export_orders' and user_id == ADMIN_ID:
        export_orders_csv(query)
    
    # Service Status Manager Handlers (Admin only)
    elif data == 'service_status_manager' and user_id == ADMIN_ID:
        show_service_status_manager(query, page=1)
    elif data.startswith('service_status_manager_p') and user_id == ADMIN_ID:
        # Pagination: service_status_manager_p<page>
        page_str = data.replace('service_status_manager_p', '')
        try:
            page = int(page_str)
            show_service_status_manager(query, page=page)
        except ValueError:
            query.answer('Invalid page')
    elif data.startswith('service_status_') and not data == 'service_status_manager' and user_id == ADMIN_ID:
        # Format: service_status_<id>
        sid_str = data.replace('service_status_', '')
        try:
            service_id = int(sid_str)
            show_service_status_options(query, service_id)
        except ValueError:
            query.answer("Invalid service id")
    elif data.startswith('set_status_') and user_id == ADMIN_ID:
        # Format: set_status_<id>_<status>
        payload = data.replace('set_status_', '')
        if '_' in payload:
            sid_str, new_status = payload.split('_', 1)
            try:
                service_id = int(sid_str)
                set_service_status(query, service_id, new_status)
            except ValueError:
                query.answer("Invalid service id")
    
    elif data == 'admin_sync':
        sync_services_manual(query)
    elif data == 'admin_settings':
        show_admin_settings(query)
    elif data == 'admin_upi_settings':
        show_upi_settings(query)
    elif data == 'admin_razorpay_settings':
        show_razorpay_settings(query)
    elif data == 'razorpay_toggle':
        toggle_razorpay(query)
    elif data == 'set_razorpay_key_id':
        logger.info(f"Admin {query.from_user.id} clicked Set Razorpay Key ID button")
        text = """
🔑 Set Razorpay Key ID

━━━━━━━━━━━━━━━━━━━━

Please send your Razorpay Key ID in the next message.

Example: rzp_test_abcd1234

📝 You can find this in:
Razorpay Dashboard → Settings → API Keys
"""
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_razorpay_settings')]]
        query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        logger.info(f"Returning state AWAITING_RAZORPAY_KEY_ID = {AWAITING_RAZORPAY_KEY_ID}")
        return AWAITING_RAZORPAY_KEY_ID
    elif data == 'set_razorpay_key_secret':
        logger.info(f"Admin {query.from_user.id} clicked Set Razorpay Key Secret button")
        text = """
🔐 Set Razorpay Key Secret

━━━━━━━━━━━━━━━━━━━━

Please send your Razorpay Key Secret in the next message.

⚠️ IMPORTANT:
   • Keep this secret safe
   • Message will be deleted for security
   • Never share with anyone

Example: your_secret_key_here

📝 You can find this in:
Razorpay Dashboard → Settings → API Keys
"""
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_razorpay_settings')]]
        query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        logger.info(f"Returning state AWAITING_RAZORPAY_KEY_SECRET = {AWAITING_RAZORPAY_KEY_SECRET}")
        return AWAITING_RAZORPAY_KEY_SECRET
    elif data == 'upload_qr_code':
        text = "📷 Upload QR Code\n\nPlease send the UPI QR Code image."
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_upi_settings')]]
        query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        return AWAITING_QR_CODE
    elif data == 'admin_profit_increase':
        adjust_profit_margin(query, 0.1)
    elif data == 'admin_profit_decrease':
        adjust_profit_margin(query, -0.1)
    elif data == 'admin_commission_increase':
        adjust_referral_commission(query, 1)
    elif data == 'admin_commission_decrease':
        adjust_referral_commission(query, -1)
    elif data.startswith('check_razorpay_'):
        check_razorpay_payment_status(query, data)
    elif data.startswith('approve_fund_'):
        approve_fund_request(query, data)
    elif data.startswith('reject_fund_'):
        reject_fund_request(query, data)
    
    return None


def show_block_user_menu(query) -> None:
    """Show block/unblock user management with button"""
    text = """
🚫 Block/Unblock User

━━━━━━━━━━━━━━━━━━━━

📋 To block a user:
   Send user ID in next message

After sending ID, you'll be asked for:
   • Block reason (optional)

✅ To unblock, use: /unblock <user_id>
📊 Check status: /userstatus <user_id>
📜 View blocked: /blockedusers
"""
    keyboard = [
        [InlineKeyboardButton("🚫 Block User (Send ID)", callback_data='admin_block_start')],
        [InlineKeyboardButton("📜 View Blocked Users", callback_data='admin_view_blocked')],
        [InlineKeyboardButton("🔙 Back to Admin Panel", callback_data='admin_panel')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        query.edit_message_text(text, reply_markup=reply_markup)
    except:
        pass


def show_admin_dashboard(query) -> None:
    """Show admin dashboard with statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM orders')
    total_orders = cursor.fetchone()[0]
    
    cursor.execute('SELECT SUM(sell_price) FROM orders')
    total_revenue = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT COUNT(*) FROM orders WHERE status = "Pending"')
    pending_orders = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM pending_funds WHERE status = "pending"')
    pending_funds = cursor.fetchone()[0]
    
    conn.close()
    
    # Get supplier balance
    supplier_balance = get_supplier_balance()
    
    text = f"""
📊 Admin Dashboard

👥 Total Users: {total_users}
📦 Total Orders: {total_orders}
💰 Total Revenue: ₹{total_revenue:.2f}
⏳ Pending Orders: {pending_orders}
💳 Pending Fund Requests: {pending_funds}

🏦 Supplier Balance: ₹{supplier_balance:.2f}
"""
    
    if supplier_balance < 500:
        text += "\n⚠️ Warning: Supplier balance is low!"
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_pending_funds(query) -> None:
    """Show pending fund requests"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT pf.id, pf.user_id, u.username, pf.amount, pf.requested_date 
        FROM pending_funds pf
        JOIN users u ON pf.user_id = u.id
        WHERE pf.status = "pending"
        ORDER BY pf.requested_date DESC
        LIMIT 10
    ''')
    requests = cursor.fetchall()
    conn.close()
    
    if not requests:
        text = "💰 Pending Fund Requests\n\nNo pending requests."
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]]
    else:
        text = "💰 Pending Fund Requests\n\n"
        keyboard = []
        
        for req in requests:
            req_id, user_id, username, amount, date = req
            text += f"🆔 {req_id} | @{username} | ₹{amount:.2f}\n"
            text += f"📅 {date[:10]}\n\n"
            keyboard.append([
                InlineKeyboardButton(f"✅ Approve #{req_id}", callback_data=f'approve_fund_{req_id}'),
                InlineKeyboardButton(f"❌ Reject #{req_id}", callback_data=f'reject_fund_{req_id}')
            ])
        
        text += "\nUse /credit <user_id> <amount> to credit manually."
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='admin_panel')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    query.edit_message_text(text, reply_markup=reply_markup)


def show_recent_orders(query) -> None:
    """Show recent orders"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT o.id, o.user_id, u.username, o.service_id, o.qty, o.sell_price, o.status
        FROM orders o
        JOIN users u ON o.user_id = u.id
        ORDER BY o.created DESC
        LIMIT 15
    ''')
    orders = cursor.fetchall()
    conn.close()
    
    if not orders:
        text = "📦 Recent Orders\n\nNo orders yet."
    else:
        text = "📦 Recent Orders (Latest 15)\n\n"
        for order in orders:
            order_id, user_id, username, service_id, qty, price, status = order
            text += f"🆔 {order_id} | @{username}\n"
            text += f"Service: {service_id} | Qty: {qty} | ₹{price:.2f}\n"
            text += f"Status: *{status}*\n\n"
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_user_list(query, page: int = 1) -> None:
    """Show user list with full details"""
    users_per_page = 5
    offset = (page - 1) * users_per_page
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total user count
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    # Get users for current page
    cursor.execute('''
        SELECT id, username, wallet, join_date, is_blocked
        FROM users
        ORDER BY join_date DESC
        LIMIT ? OFFSET ?
    ''', (users_per_page, offset))
    users = cursor.fetchall()
    conn.close()
    
    if not users:
        text = "👥 User List\n\nNo users found."
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]]
    else:
        total_pages = (total_users + users_per_page - 1) // users_per_page
        text = f"👥 User List (Page {page}/{total_pages})\n\n"
        
        for user in users:
            user_id, username, wallet, join_date, is_blocked = user
            status_emoji = "🚫" if is_blocked else "✅"
            
            # Get order count for user
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM orders WHERE user_id = ?', (user_id,))
            order_count = cursor.fetchone()[0]
            conn.close()
            
            text += f"{status_emoji} *@{escape_markdown(username)}*\n"
            text += f"   ID: {user_id}\n"
            text += f"   💰 Wallet: ₹{wallet:.2f}\n"
            text += f"   📦 Orders: {order_count}\n"
            text += f"   📅 Joined: {join_date[:10]}\n"
            text += f"   Status: {'BLOCKED' if is_blocked else 'ACTIVE'}\n\n"
        
        # Navigation buttons
        keyboard = []
        nav_buttons = []
        if page > 1:
            nav_buttons.append(InlineKeyboardButton("◀️ Previous", callback_data=f'userlist_page_{page-1}'))
        if page < total_pages:
            nav_buttons.append(InlineKeyboardButton("Next ▶️", callback_data=f'userlist_page_{page+1}'))
        if nav_buttons:
            keyboard.append(nav_buttons)
        
        text += f"Total Users: {total_users}"
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='admin_panel')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        query.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error showing user list: {e}")
        query.answer("Error loading user list")


def show_refund_menu(query) -> None:
    """Show refund menu with button"""
    text = """
💸 Refund Orders

━━━━━━━━━━━━━━━━━━━━

📋 To refund an order:
   Send order ID in next message

💰 The order amount will be credited
   back to the user's wallet

📊 View recent orders first to
   find the order ID
"""
    
    keyboard = [
        [InlineKeyboardButton("💸 Refund Order (Send ID)", callback_data='admin_refund_start')],
        [InlineKeyboardButton("📦 View Recent Orders", callback_data='admin_orders')],
        [InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_broadcast_menu(query) -> None:
    """Show broadcast menu with button"""
    text = """
📣 Broadcast Message

━━━━━━━━━━━━━━━━━━━━

📢 Send a message to ALL users

💡 After clicking the button below:
   • Type your broadcast message
   • It will be sent to all users

⚠️ Use carefully - all users will
   receive your message!
"""
    
    keyboard = [
        [InlineKeyboardButton("📣 Start Broadcast", callback_data='admin_broadcast_start')],
        [InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def sync_services_manual(query) -> None:
    """Manually trigger service sync"""
    query.answer("Syncing services...")
    result = sync_services()
    
    if isinstance(result, tuple):
        count, balance = result
        text = f"✅ Service sync completed!\n\n{count} services synced.\n\n⚠️ Supplier balance: ₹{balance:.2f}"
    else:
        text = f"✅ Service sync completed!\n\n{result} services synced."
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_admin_settings(query) -> None:
    """Show admin settings menu"""
    profit_margin = round(float(get_setting('profit_margin', '1.5')), 2)
    profit_percent = round((profit_margin - 1) * 100, 1)
    referral_commission = round(float(get_setting('referral_commission', '10')), 1)
    
    text = f"""
╔═══════════════════════╗
      ⚙️ Admin Settings
╚═══════════════════════╝

💰 Current Profit Margin: {profit_percent:.0f}%
   (Sell Rate = Supplier Rate × {profit_margin:.1f})

🎁 Referral Commission: {referral_commission:.0f}%
   (Users earn on referred orders)

━━━━━━━━━━━━━━━━━━━━━━━

Select an option:
"""
    
    keyboard = [
        [InlineKeyboardButton("💵 UPI Settings", callback_data='admin_upi_settings')],
        [InlineKeyboardButton("💳 Razorpay Settings", callback_data='admin_razorpay_settings')],
        [
            InlineKeyboardButton("➕ Increase Profit (+10%)", callback_data='admin_profit_increase'),
            InlineKeyboardButton("➖ Decrease Profit (-10%)", callback_data='admin_profit_decrease')
        ],
        [
            InlineKeyboardButton("➕ Increase Commission (+1%)", callback_data='admin_commission_increase'),
            InlineKeyboardButton("➖ Decrease Commission (-1%)", callback_data='admin_commission_decrease')
        ],
        [InlineKeyboardButton("🔙 Back", callback_data='admin_panel')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        query.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error updating admin settings: {e}")


def show_upi_settings(query) -> None:
    """Show UPI settings"""
    upi_id = get_setting('upi_id', 'your-upi@bank')
    upi_name = get_setting('upi_name', 'Your Name')
    
    text = f"""
💵 UPI Settings

Current Settings:
UPI ID: {upi_id}
Name: {upi_name}

To update UPI settings, use these commands:
/set_upi_id <upi@bank>
/set_upi_name <name>

Example:
/set_upi_id svxhub@paytm
/set_upi_name SVXHUB
"""
    
    keyboard = [
        [InlineKeyboardButton("📷 Upload QR Code", callback_data='upload_qr_code')],
        [InlineKeyboardButton("🔙 Back", callback_data='admin_settings')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text, reply_markup=reply_markup)


def show_razorpay_settings(query) -> None:
    """Show Razorpay settings"""
    razorpay_enabled = is_razorpay_enabled()
    razorpay_key_id = get_setting('razorpay_key_id', '')
    razorpay_key_secret = get_setting('razorpay_key_secret', '')
    
    status_emoji = "✅" if razorpay_enabled else "❌"
    status_text = "Enabled" if razorpay_enabled else "Disabled"
    
    # Show masked key ID
    key_id_display = f"{razorpay_key_id[:8]}...{razorpay_key_id[-4:]}" if len(razorpay_key_id) > 12 else (razorpay_key_id if razorpay_key_id else "Not Set")
    key_secret_display = "✅ Set (Hidden)" if razorpay_key_secret else "❌ Not Set"
    
    text = f"""
╔══════════════════════╗
  💳 Razorpay Settings
╚══════════════════════╝

Status: {status_emoji} {status_text}

━━━━━━━━━━━━━━━━━━━━

🔧 API Configuration:

Key ID: {key_id_display}
Secret: {key_secret_display}

━━━━━━━━━━━━━━━━━━━━

📚 How to get Razorpay keys:

1. Sign up at https://razorpay.com
2. Dashboard → Settings → API Keys
3. Generate Test/Live keys
4. Click buttons below to set them

━━━━━━━━━━━━━━━━━━━━

⚡ Benefits:
   • Instant auto-credit
   • Multiple payment methods
   • Secure & trusted
   • 24/7 availability
"""
    
    toggle_text = "🔴 Disable Razorpay" if razorpay_enabled else "🟢 Enable Razorpay"
    
    keyboard = [
        [InlineKeyboardButton("🔑 Set Key ID", callback_data='set_razorpay_key_id')],
        [InlineKeyboardButton("🔐 Set Key Secret", callback_data='set_razorpay_key_secret')],
        [InlineKeyboardButton(toggle_text, callback_data='razorpay_toggle')],
        [InlineKeyboardButton("🔙 Back", callback_data='admin_settings')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        query.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error showing Razorpay settings: {e}")


def toggle_razorpay(query) -> None:
    """Toggle Razorpay enable/disable"""
    current = is_razorpay_enabled()
    new_value = 'false' if current else 'true'
    set_setting('razorpay_enabled', new_value)
    
    if new_value == 'true':
        query.answer("✅ Razorpay enabled!")
    else:
        query.answer("❌ Razorpay disabled!")
    
    show_razorpay_settings(query)


def adjust_profit_margin(query, change: float) -> None:
    """Adjust profit margin and update all service prices"""
    current = float(get_setting('profit_margin', '1.5'))
    new_margin = max(1.0, round(current + change, 2))  # Minimum 0% profit (1.0x), round to 2 decimals
    set_setting('profit_margin', str(new_margin))
    
    profit_percent = (new_margin - 1) * 100
    
    # Update all service prices in database without API sync
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all services and recalculate sell_rate
        cursor.execute('SELECT id, supplier_rate FROM services')
        services = cursor.fetchall()
        
        updated_count = 0
        for service_id, supplier_rate in services:
            new_sell_rate = round(supplier_rate * new_margin, 2)
            cursor.execute('UPDATE services SET sell_rate = ? WHERE id = ?', (new_sell_rate, service_id))
            updated_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated {updated_count} service prices with new profit margin {new_margin}")
        
        query.answer(f"✅ Profit updated to {profit_percent:.0f}%! {updated_count} prices updated.")
        show_admin_settings(query)
    except Exception as e:
        logger.error(f"Error in adjust_profit_margin: {e}")
        try:
            query.edit_message_text("✅ Profit updated successfully!")
        except:
            pass


def adjust_referral_commission(query, change: float) -> None:
    """Adjust referral commission percentage"""
    current = float(get_setting('referral_commission', '10'))
    new_commission = max(0, min(100, round(current + change, 1)))  # Keep between 0% and 100%, round to 1 decimal
    set_setting('referral_commission', str(new_commission))
    
    try:
        query.answer(f"✅ Referral commission updated to {new_commission:.0f}%!")
        show_admin_settings(query)
    except Exception as e:
        logger.error(f"Error in adjust_referral_commission: {e}")
        try:
            query.edit_message_text("✅ Commission updated successfully!")
        except:
            pass


def check_razorpay_payment_status(query, data: str) -> None:
    """Check Razorpay payment status"""
    payment_link_id = data.replace('check_razorpay_', '')
    user_id = query.from_user.id
    
    logger.info(f"Checking Razorpay payment status: user_id={user_id}, payment_link_id={payment_link_id}")
    
    # Get payment info from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, payment_id, amount, status 
        FROM razorpay_payments 
        WHERE payment_link_id = ? AND user_id = ?
    ''', (payment_link_id, user_id))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        logger.warning(f"Payment record not found for payment_link_id={payment_link_id}, user_id={user_id}")
        query.answer("❌ Payment record not found!")
        return
    
    db_id, payment_id, amount, status = result
    logger.info(f"Payment record found: db_id={db_id}, status={status}, amount={amount}")
    
    if status == 'completed':
        query.answer("✅ This payment was already credited!")
        return
    
    # Try to fetch payment status from Razorpay
    query.answer("🔍 Checking payment status... Please wait.")
    
    try:
        client = get_razorpay_client()
        if not client:
            logger.error("Razorpay client not configured")
            query.answer("❌ Razorpay not configured. Contact admin.")
            return
        
        # Fetch payment link details
        logger.info(f"Fetching payment link from Razorpay: {payment_link_id}")
        payment_link_data = client.payment_link.fetch(payment_link_id)
        logger.info(f"Payment link status: {payment_link_data.get('status')}")
        
        if payment_link_data.get('status') == 'paid':
            # Payment was successful, credit the wallet with 5% fee deduction
            payment_id = payment_link_data.get('payments', [{}])[0].get('payment_id') if payment_link_data.get('payments') else None
            
            # Calculate amount after 5% fee
            paid_amount = amount
            fee_amount = paid_amount * 0.05
            credit_amount = paid_amount - fee_amount
            
            # Credit user wallet with amount after fee
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET wallet = wallet + ? WHERE id = ?', (credit_amount, user_id))
            cursor.execute('''
                UPDATE razorpay_payments 
                SET status = ?, payment_id = ?, completed_date = ? 
                WHERE id = ?
            ''', ('completed', payment_id, datetime.now().isoformat(), db_id))
            conn.commit()
            conn.close()
            
            # Notify user
            text = f"""
✅ Payment Confirmed!

💰 Paid Amount: ₹{paid_amount:.2f}
💳 Convenience Fee (5%): ₹{fee_amount:.2f}
💵 Credited Amount: ₹{credit_amount:.2f}
🆔 Payment ID: {payment_id or 'N/A'}

Your wallet has been credited successfully!

Current Balance: Check your profile
"""
            query.edit_message_text(text)
            
            # Notify admin
            try:
                user_info = get_user_info(user_id)
                query.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=f"💳 Razorpay Payment Received\n\nUser: @{user_info['username']} ({user_id})\nPaid: ₹{paid_amount:.2f}\nFee: ₹{fee_amount:.2f}\nCredited: ₹{credit_amount:.2f}\nPayment ID: {payment_id or 'N/A'}\n\n✅ Wallet credited automatically"
                )
            except Exception as e:
                logger.error(f"Failed to notify admin: {e}")
                
        elif payment_link_data.get('status') == 'created':
            query.answer("⏳ Payment pending. Please complete the payment first.")
        elif payment_link_data.get('status') == 'expired':
            query.answer("⏰ Payment link expired. Please create a new one.")
        else:
            query.answer(f"❓ Payment status: {payment_link_data.get('status')}")
            
    except Exception as e:
        logger.error(f"Error checking Razorpay payment: {e}")
        query.answer("❌ Failed to check payment status. Try again later.")


def approve_fund_request(query, data: str) -> None:
    """Approve a fund request"""
    req_id = int(data.replace('approve_fund_', ''))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, amount FROM pending_funds WHERE id = ? AND status = "pending"', (req_id,))
    result = cursor.fetchone()
    
    if not result:
        query.answer("Request not found or already processed")
        return
    
    user_id, amount = result
    
    # Update request status
    cursor.execute('UPDATE pending_funds SET status = "approved" WHERE id = ?', (req_id,))
    conn.commit()
    conn.close()
    
    # Credit user wallet
    update_user_wallet(user_id, amount)
    
    query.answer(f"✅ Approved! ₹{amount:.2f} credited to user {user_id}")
    
    # Notify user
    try:
        query.message.bot.send_message(
            chat_id=user_id,
            text=f"✅ Fund Request Approved!\n\n💰 ₹{amount:.2f} has been credited to your wallet."
        )
    except Exception as e:
        logger.error(f"Failed to notify user {user_id}: {e}")
    
    # Refresh pending funds list
    show_pending_funds(query)


def reject_fund_request(query, data: str) -> None:
    """Reject a fund request"""
    req_id = int(data.replace('reject_fund_', ''))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_id FROM pending_funds WHERE id = ? AND status = "pending"', (req_id,))
    result = cursor.fetchone()
    
    if not result:
        query.answer("Request not found or already processed")
        return
    
    user_id = result[0]
    
    # Update request status
    cursor.execute('UPDATE pending_funds SET status = "rejected" WHERE id = ?', (req_id,))
    conn.commit()
    conn.close()
    
    query.answer(f"❌ Rejected fund request #{req_id}")
    
    # Notify user
    try:
        query.message.bot.send_message(
            chat_id=user_id,
            text="❌ Fund Request Rejected\n\nYour fund request has been rejected.\n\nPlease contact support for more information."
        )
    except Exception as e:
        logger.error(f"Failed to notify user {user_id}: {e}")
    
    # Refresh pending funds list
    show_pending_funds(query)


# ======================== COMMAND HANDLERS ========================

def order_command(update: Update, context: CallbackContext) -> None:
    """Handle /order command"""
    user_id = update.effective_user.id
    
    if len(context.args) < 3:
        update.message.reply_text(
            "❌ Invalid format!\n\nUsage: /order <service_id> <quantity> <link>\n\nExample: /order 123 1000 https://instagram.com/username")
        return
    
    try:
        service_id = int(context.args[0])
        qty = int(context.args[1])
        link = context.args[2]
    except ValueError:
        update.message.reply_text("❌ Invalid service ID or quantity. Please use numbers.")
        return
    
    # Create order
    result = create_order(user_id, service_id, qty, link)
    
    if result['success']:
        text = f"""
╔══════════════════════╗
   ✅ ORDER CONFIRMED
╚══════════════════════╝

Your order has been placed!

━━━━━ Order Info ━━━━━

🆔 Order ID: #{result['order_id']}
🆔 Supplier ID: {result['supplier_order_id']}
💰 Amount: ₹{result['total_price']:.2f}

━━━━━━━━━━━━━━━━━━━━

⚡ Processing started!
📊 Track: /status {result['order_id']}
"""
    else:
        text = f"❌ Order Failed\n\n{result['error']}"
    
    update.message.reply_text(text)


def status_command(update: Update, context: CallbackContext) -> None:
    """Handle /status command"""
    if len(context.args) < 1:
        update.message.reply_text(
            "❌ Usage: /status <order_id>")
        return
    
    try:
        order_id = int(context.args[0])
    except ValueError:
        update.message.reply_text("❌ Invalid order ID.")
        return
    
    result = check_order_status(order_id)
    
    if not result['success']:
        update.message.reply_text(f"❌ {result['error']}")
        return
    
    text = f"""
📦 Order Status

🆔 Order ID: {result['id']}
🔢 Service ID: {result['service_id']}
📊 Quantity: {result['qty']}
💰 Price: ₹{result['sell_price']:.2f}
📍 Status: *{result['status']}*
📅 Created: {result['created'][:16]}
"""
    
    if 'charge' in result:
        text += f"\n💳 Charge: ₹{result['charge']}"
    if 'start_count' in result:
        text += f"\n▶️ Start Count: {result['start_count']}"
    if 'remains' in result:
        text += f"\n⏳ Remains: {result['remains']}"
    
    update.message.reply_text(text)


def handle_order_quantity(update: Update, context: CallbackContext) -> int:
    """Handle order quantity input"""
    user_id = update.effective_user.id
    
    # Apply rate limiting (except for admins)
    if not is_admin(user_id):
        allowed, error_msg = check_rate_limit(user_id)
        if not allowed:
            update.message.reply_text(error_msg)
            return ConversationHandler.END
    
    try:
        quantity = int(update.message.text)
    except ValueError:
        update.message.reply_text("❌ Invalid quantity. Please send a number.")
        return AWAITING_ORDER_QUANTITY
    
    service_id = context.user_data.get('order_service_id')
    service = get_service_by_id(service_id)
    
    if not service:
        update.message.reply_text("❌ Service not found. Please start over.")
        return ConversationHandler.END
    
    if quantity < service['min_qty'] or quantity > service['max_qty']:
        update.message.reply_text(
            f"❌ Quantity must be between {service['min_qty']} and {service['max_qty']}.")
        return AWAITING_ORDER_QUANTITY
    
    context.user_data['order_quantity'] = quantity
    total_price = service['sell_rate'] * quantity
    
    text = f"🔗 Order Link\n\nPlease send the link where you want the service delivered.\n\nQuantity: {quantity}\nTotal Price: ₹{total_price:.2f}"
    keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='back_to_main')]]
    update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    return AWAITING_ORDER_LINK


def handle_order_link(update: Update, context: CallbackContext) -> int:
    """Handle order link input and create order"""
    link = update.message.text.strip()
    user_id = update.effective_user.id
    
    # Apply rate limiting (except for admins)
    if not is_admin(user_id):
        allowed, error_msg = check_rate_limit(user_id)
        if not allowed:
            update.message.reply_text(error_msg)
            return ConversationHandler.END
    
    service_id = context.user_data.get('order_service_id')
    quantity = context.user_data.get('order_quantity')
    
    if not service_id or not quantity:
        update.message.reply_text("❌ Order data missing. Please start over.")
        return ConversationHandler.END
    
    # Show processing message
    processing_msg = update.message.reply_text("⏳ Processing your order...\nPlease wait...")
    
    # Create order
    result = create_order(user_id, service_id, quantity, link)
    
    if result['success']:
        text = f"""
✅ <b>ORDER PLACED SUCCESSFULLY!</b>

━━━━━━━━━━━━━━━━━━━━
🆔 <b>Order ID:</b> #{result['order_id']}
🔢 <b>Supplier Order ID:</b> {result['supplier_order_id']}
💰 <b>Total Price:</b> ₹{result['total_price']:.2f}
📊 <b>Quantity:</b> {quantity:,}
🔗 <b>Link:</b> {link[:50]}...
━━━━━━━━━━━━━━━━━━━━

✨ <b>Your order is being processed!</b>

📬 You'll receive instant updates when:
   • Order starts processing
   • Order is completed
   • Any status changes occur

⏱️ Status checks run every 5 minutes automatically.

👇 Track your order below:
"""
        keyboard = [
            [InlineKeyboardButton("📦 My Orders", callback_data='menu_myorders')],
            [InlineKeyboardButton("🔄 New Order", callback_data='menu_order')],
            [InlineKeyboardButton("🏠 Main Menu", callback_data='back_to_main')]
        ]
        
        # Delete processing message and send success message
        try:
            processing_msg.delete()
        except:
            pass
        update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')
    else:
        text = f"❌ <b>ORDER FAILED</b>\n\n{result['error']}"
        keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data='back_to_main')]]
        
        # Delete processing message and send error message
        try:
            processing_msg.delete()
        except:
            pass
        update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')
    
    # Clear order data
    context.user_data.pop('order_service_id', None)
    context.user_data.pop('order_quantity', None)
    
    return ConversationHandler.END


def addfund_command(update: Update, context: CallbackContext) -> int:
    """Handle /addfund command"""
    update.message.reply_text(
        "💰 Add Funds\n\nPlease upload your payment screenshot:")
    return AWAITING_PAYMENT_PROOF


def handle_payment_proof(update: Update, context: CallbackContext) -> int:
    """Handle payment proof upload - Step 2 of 3"""
    user = update.effective_user
    
    # Apply rate limiting (except for admins)
    if not is_admin(user.id):
        allowed, error_msg = check_rate_limit(user.id)
        if not allowed:
            update.message.reply_text(error_msg)
            return ConversationHandler.END
    
    if not update.message.photo:
        update.message.reply_text("❌ Please send a photo of your payment proof.")
        return AWAITING_PAYMENT_PROOF
    
    photo = update.message.photo[-1]
    context.user_data['payment_photo'] = photo.file_id
    
    # Ask for UTR/Reference Number
    update.message.reply_text(
        "✅ Screenshot received!\n\n"
        "📝 Step 3/3: Enter UTR/Reference Number\n\n"
        "Please send the UTR or UPI Reference Number from your transaction.\n\n"
        "⚠️ IMPORTANT:\n"
        "• Must be 10-20 characters\n"
        "• Only letters and numbers\n"
        "• No spaces or special characters\n"
        "• Must contain numbers\n\n"
        "Examples:\n"
        "✅ 123456789012 (UPI UTR)\n"
        "✅ AB1234567890CD (Bank Ref)\n"
        "❌ 12345 (too short)\n"
        "❌ ABC-123 (contains special char)\n\n"
        "Find it in your payment app under transaction details."
    )
    
    return AWAITING_UTR_NUMBER


def handle_utr_number(update: Update, context: CallbackContext) -> int:
    """Handle UTR/Reference number input - Step 3 of 3"""
    user = update.effective_user
    utr_number = update.message.text.strip().upper()
    
    # Validate UTR format
    # UPI UTR is typically 12 digits, Bank reference can be alphanumeric 10-16 chars
    if not utr_number.isalnum():
        update.message.reply_text(
            "❌ Invalid UTR/Reference number!\n\n"
            "UTR should contain only letters and numbers.\n"
            "No spaces or special characters allowed.\n\n"
            "Please send a valid UTR/Reference number."
        )
        return AWAITING_UTR_NUMBER
    
    if len(utr_number) < 10:
        update.message.reply_text(
            "❌ Invalid UTR/Reference number!\n\n"
            "UTR/Reference number must be at least 10 characters.\n\n"
            "Examples of valid formats:\n"
            "• 123456789012 (12-digit UPI UTR)\n"
            "• AB1234567890 (Bank reference)\n\n"
            "Please send a valid UTR/Reference number."
        )
        return AWAITING_UTR_NUMBER
    
    if len(utr_number) > 20:
        update.message.reply_text(
            "❌ Invalid UTR/Reference number!\n\n"
            "UTR/Reference number is too long (max 20 characters).\n\n"
            "Please send a valid UTR/Reference number."
        )
        return AWAITING_UTR_NUMBER
    
    # Additional check: Must contain at least some digits
    if not any(char.isdigit() for char in utr_number):
        update.message.reply_text(
            "❌ Invalid UTR/Reference number!\n\n"
            "UTR must contain at least some numbers.\n\n"
            "Please send a valid UTR/Reference number from your payment."
        )
        return AWAITING_UTR_NUMBER
    
    # Check if UTR already exists
    existing_utr = check_utr_exists(utr_number)
    if existing_utr:
        # Block current user for using duplicate UTR
        block_user(user.id)
        
        # Notify admin about duplicate UTR attempt
        try:
            context.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"🚨 DUPLICATE UTR DETECTED!\n\n"
                     f"User @{user.username or user.first_name} ({user.id}) tried to use UTR that was already used by:\n\n"
                     f"Previous User: @{existing_utr['username']} ({existing_utr['user_id']})\n"
                     f"UTR: {utr_number}\n"
                     f"First Used: {existing_utr['used_date'][:10]}\n\n"
                     f"⚠️ Both users have been automatically blocked!"
            )
        except Exception as e:
            logger.error(f"Failed to notify admin about duplicate UTR: {e}")
        
        # Also block the previous user
        block_user(existing_utr['user_id'])
        
        update.message.reply_text(
            "🚫 FRAUD DETECTED!\n\n"
            "This UTR/Reference number has already been used. Your account has been blocked.\n\n"
            "If this is a mistake, please contact support.")
        return ConversationHandler.END
    
    # Get amount and photo from context
    amount = context.user_data.get('payment_amount')
    photo_file_id = context.user_data.get('payment_photo')
    
    if not amount or not photo_file_id:
        update.message.reply_text("❌ Missing information. Please start again with /addfund")
        return ConversationHandler.END
    
    # Add UTR to tracking
    if not add_utr_tracking(utr_number, user.id):
        update.message.reply_text("❌ Failed to process UTR. Please contact support.")
        return ConversationHandler.END
    
    # Get payment code from context
    payment_code = context.user_data.get('payment_code', '')
    
    # Save to pending funds with UTR and payment code
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO pending_funds (user_id, amount, photo_file_id, utr_number, payment_code, status, requested_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user.id, amount, photo_file_id, utr_number, payment_code, 'pending', datetime.now().isoformat()))
    pending_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Store pending_id for OCR verification
    context.user_data['pending_payment_id'] = pending_id
    
    # DO NOT notify admin yet - wait for OCR verification
    user_info = get_user_info(user.id)  # Get user info for later use
    
    update.message.reply_text(
        "✅ Your fund request has been submitted!\n\n"
        f"Amount: ₹{amount:.2f}\n"
        f"UTR: {utr_number}\n"
        f"Code: {payment_code}\n\n"
        "⏳ Verifying payment automatically..."
    )
    
    # Perform OCR verification with timeout
    import time
    start_time = time.time()
    
    try:
        # Show animated scanning progress
        scanning_msg = update.message.reply_text(
            "🔄 **Scanning your payment receipt...**\n\n"
            "▰▰▰▱▱▱▱▱▱▱ 30%\n\n"
            "Please wait...",
            parse_mode='Markdown'
        )
        
        # Download the payment screenshot (fast operation)
        photo_file = context.bot.get_file(photo_file_id)
        photo_bytes = photo_file.download_as_bytearray()
        
        # Update progress
        try:
            scanning_msg.edit_text(
                "🔄 **Scanning your payment receipt...**\n\n"
                "▰▰▰▰▰▰▱▱▱▱ 60%\n\n"
                "Analyzing receipt details...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        # Verify payment receipt using OCR
        logger.info(f"Starting OCR verification for user {user.id}, code: {payment_code}")
        verification_result = verify_payment_receipt(
            image_bytes=bytes(photo_bytes),
            expected_code=payment_code,
            min_amount=amount,
            utr_number=utr_number
        )
        
        # Update progress to complete
        try:
            scanning_msg.edit_text(
                "✅ **Scanning Complete!**\n\n"
                "▰▰▰▰▰▰▰▰▰▰ 100%\n\n"
                "Processing results...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        elapsed_time = time.time() - start_time
        logger.info(f"OCR verification completed in {elapsed_time:.2f} seconds for user {user.id}: {verification_result}")
        
        if verification_result['verified']:
            # OCR VERIFIED - Do NOT auto-credit, send to admin for approval
            
            # Delete scanning animation message
            try:
                scanning_msg.delete()
            except:
                pass
            
            # Notify user - payment is verified and sent to admin
            update.message.reply_text(
                f"✅ **PAYMENT VERIFIED!**\n\n"
                f"Your payment code has been verified successfully!\n\n"
                f"💰 Amount: ₹{amount:.2f}\n"
                f"✓ Verification Code Matched: {payment_code}\n"
                f"✓ Sent to admin for approval\n\n"
                f"You'll be notified once approved! 🙏",
                parse_mode='Markdown'
            )
            
            # Send to admin with verification success
            try:
                context.bot.send_photo(
                    chat_id=ADMIN_ID,
                    photo=photo_file_id,
                    caption=f"✅ **OCR VERIFIED** Payment Request\n\n"
                            f"User: @{user_info['username']} ({user.id})\n"
                            f"Amount: ₹{amount:.2f}\n"
                            f"Code: {payment_code} ✓ MATCHED\n"
                            f"UTR: {utr_number}\n\n"
                            f"✓ OCR verification successful\n"
                            f"✓ Code found in receipt\n"
                            f"✓ Verified in {elapsed_time:.1f}s\n\n"
                            f"Use /credit {user.id} {amount} to approve.",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to notify admin: {e}")
                
        else:
            # Delete scanning animation message
            try:
                scanning_msg.delete()
            except:
                pass
            
            # Auto-reject or flag for manual review
            reason = verification_result.get('reason', 'Unknown error')
            
            # Update status to rejected
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE pending_funds SET status = 'rejected' WHERE id = ?",
                (pending_id,)
            )
            conn.commit()
            conn.close()
            
            # Notify user
            update.message.reply_text(
                f"❌ **PAYMENT VERIFICATION FAILED**\n\n"
                f"Reason: {reason}\n\n"
                f"⚠️ Please ensure:\n"
                f"• You included code **{payment_code}** in payment notes/remarks\n"
                f"• The code is clearly visible in the screenshot\n"
                f"• Amount matches: ₹{amount:.2f}\n"
                f"• UTR number is correct: {utr_number}\n\n"
                f"Please try again with /addfund or contact support.",
                parse_mode='Markdown'
            )
            
            # DO NOT notify admin for rejected payments
            logger.info(f"Payment rejected for user {user.id}: {reason}")
                
    except Exception as ocr_error:
        logger.error(f"OCR verification failed with exception: {ocr_error}")
        
        # Delete scanning animation message
        try:
            scanning_msg.delete()
        except:
            pass
        
        # Do not auto-reject on OCR errors, leave for manual review
        update.message.reply_text(
            "⚠️ **Automatic verification unavailable**\n\n"
            "Your payment is pending manual review by admin.\n"
            "You will be notified once approved.\n\n"
            "Thank you for your patience! 🙏",
            parse_mode='Markdown'
        )
        
        # DO NOT notify admin for OCR errors - payment already in database as pending
        logger.info(f"OCR error for user {user.id}, payment left as pending")
    
    return ConversationHandler.END


def handle_payment_amount(update: Update, context: CallbackContext) -> int:
    """Handle payment amount - Step 1 of 3"""
    user = update.effective_user
    
    try:
        amount = float(update.message.text)
    except ValueError:
        update.message.reply_text("❌ Invalid amount. Please send a number.")
        return AWAITING_PAYMENT_AMOUNT
    
    if amount < 1:
        update.message.reply_text("❌ Minimum amount is ₹1.\n\nPlease enter at least ₹1:")
        return AWAITING_PAYMENT_AMOUNT
    
    if amount > 1000:
        update.message.reply_text("❌ Maximum add funds limit is ₹1000 per transaction.\n\nPlease enter an amount up to ₹1000:")
        return AWAITING_PAYMENT_AMOUNT
    
    # Store amount
    context.user_data['payment_amount'] = amount
    
    # Get payment code from context (generated when user clicked upload_payment)
    payment_code = context.user_data.get('payment_code', 'CODE_ERROR')

    # Get UPI details from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM settings WHERE key = ?', ('upi_id',))
    upi_id_row = cursor.fetchone()
    cursor.execute('SELECT value FROM settings WHERE key = ?', ('upi_name',))
    upi_name_row = cursor.fetchone()
    cursor.execute('SELECT value FROM settings WHERE key = ?', ('qr_code_file_id',))
    qr_code_row = cursor.fetchone()
    conn.close()
    
    upi_id = upi_id_row[0] if upi_id_row else "Not Set"
    upi_name = upi_name_row[0] if upi_name_row else "Not Set"
    qr_code_file_id = qr_code_row[0] if qr_code_row else None
    
    # Create message with payment instructions
    text = f"""
💰 Amount: ₹{amount:.2f}

━━━━━━━━━━━━━━━━━━━━

🔐 **VERIFICATION CODE:**
`{payment_code}` (tap to copy)

⚠️ **CRITICAL:** Include this code in payment notes/remarks/description field!

━━━━━━━━━━━━━━━━━━━━

📱 Step 2/3: Complete Payment

Please pay ₹{amount:.2f} to:

💳 UPI ID: `{upi_id}` (tap to copy)
👤 Name: {upi_name}



After payment, upload your payment screenshot below.

 Make sure screenshot shows:
    Transaction amount
    Verification code in notes/remarks: **{payment_code}**
    UPI Reference/UTR number
    Payment success status
"""
    
    # If QR code exists, send it as a photo with caption
    if qr_code_file_id:
        try:
            update.message.reply_photo(
                photo=qr_code_file_id,
                caption=text,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending QR code: {e}")
            update.message.reply_text(text, parse_mode='Markdown')
    else:
        update.message.reply_text(text, parse_mode='Markdown')
    
    return AWAITING_PAYMENT_PROOF


def handle_razorpay_amount(update: Update, context: CallbackContext) -> int:
    """Handle Razorpay payment amount"""
    user = update.effective_user
    
    try:
        amount = float(update.message.text)
    except ValueError:
        update.message.reply_text("❌ Invalid amount. Please send a number.")
        return AWAITING_RAZORPAY_AMOUNT
    
    if amount < 1:
        update.message.reply_text("❌ Minimum amount is ₹1.\n\nPlease enter at least ₹1:")
        return AWAITING_RAZORPAY_AMOUNT
    
    if amount > 1000:
        update.message.reply_text("❌ Maximum add funds limit is ₹1000 per transaction.\n\nPlease enter an amount up to ₹1000:")
        return AWAITING_RAZORPAY_AMOUNT
    
    # Generate Razorpay payment link
    update.message.reply_text("⏳ Generating secure payment link... Please wait.")
    
    user_name = user.username or user.first_name or "User"
    payment_link = create_razorpay_payment_link(user.id, amount, user_name)
    
    if not payment_link:
        update.message.reply_text(
            "❌ Failed to generate payment link. Please try manual payment or contact support."
        )
        return ConversationHandler.END
    
    # Store payment link info in database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO razorpay_payments (user_id, payment_link_id, amount, status, created_date)
        VALUES (?, ?, ?, ?, ?)
    ''', (user.id, payment_link['id'], amount, 'pending', datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    # Calculate amount after fee
    fee_amount = amount * 0.05
    credit_amount = amount - fee_amount
    
    # Send payment link to user
    text = f"""
✅ Payment Link Generated!

💰 Payment Amount: ₹{amount:.2f}
💳 Convenience Fee (5%): ₹{fee_amount:.2f}
💵 You'll Get: ₹{credit_amount:.2f}

🔗 Click below to pay securely:

{payment_link['short_url']}

━━━━━━━━━━━━━━━━━━━━

📱 Payment Methods Available:
   • UPI (Google Pay, PhonePe, Paytm, etc.)
   • Debit/Credit Card
   • Net Banking
   • Wallets

⚡ Your wallet will be credited INSTANTLY after payment!

⏰ Link valid for 24 hours
🔒 100% Secure Payment via Razorpay

━━━━━━━━━━━━━━━━━━━━

After payment, click "Check Status" below
Or wait for auto-credit (usually instant)
"""
    
    keyboard = [
        [InlineKeyboardButton("💳 Open Payment Link", url=payment_link['short_url'])],
        [InlineKeyboardButton("✅ I've Paid - Check Status", callback_data=f'check_razorpay_{payment_link["id"]}')],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_main')]
    ]
    
    update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    return ConversationHandler.END


def handle_qr_code_upload(update: Update, context: CallbackContext) -> int:
    """Handle QR code upload from admin"""
    user = update.effective_user
    
    if user.id != ADMIN_ID:
        update.message.reply_text("❌ This feature is for admins only.")
        return ConversationHandler.END
    
    if not update.message.photo:
        update.message.reply_text("❌ Please send a photo of the QR code.")
        return AWAITING_QR_CODE
    
    photo = update.message.photo[-1]
    photo_file_id = photo.file_id
    
    # Save QR code to database settings
    set_setting('qr_code_file_id', photo_file_id)
    
    update.message.reply_text(
        "✅ UPI QR Code updated successfully!\n\nUsers will now see this QR code when adding funds.")
    
    return ConversationHandler.END


def handle_razorpay_key_id_input(update: Update, context: CallbackContext) -> int:
    """Handle Razorpay Key ID input from admin"""
    user = update.effective_user
    logger.info(f"handle_razorpay_key_id_input called by user {user.id}")
    
    if user.id != ADMIN_ID:
        update.message.reply_text("❌ This feature is for admins only.")
        return ConversationHandler.END
    
    key_id = update.message.text.strip()
    logger.info(f"Received key_id: {key_id[:10]}...")
    
    # Validate key ID format (starts with rzp_test_ or rzp_live_)
    if not (key_id.startswith('rzp_test_') or key_id.startswith('rzp_live_')):
        update.message.reply_text(
            "❌ Invalid Key ID format!\n\n"
            "Key ID should start with:\n"
            "• rzp_test_ (for test mode)\n"
            "• rzp_live_ (for live mode)\n\n"
            "Please try again or click Cancel."
        )
        return AWAITING_RAZORPAY_KEY_ID
    
    set_setting('razorpay_key_id', key_id)
    logger.info(f"Razorpay key_id saved successfully")
    
    update.message.reply_text(
        f"✅ Razorpay Key ID updated successfully!\n\n"
        f"Key ID: {key_id[:8]}...{key_id[-4:]}\n\n"
        "⚠️ Don't forget to set the Key Secret as well!"
    )
    
    return ConversationHandler.END


def handle_razorpay_key_secret_input(update: Update, context: CallbackContext) -> int:
    """Handle Razorpay Key Secret input from admin"""
    user = update.effective_user
    logger.info(f"handle_razorpay_key_secret_input called by user {user.id}")
    
    if user.id != ADMIN_ID:
        update.message.reply_text("❌ This feature is for admins only.")
        return ConversationHandler.END
    
    key_secret = update.message.text.strip()
    logger.info(f"Received key_secret (length: {len(key_secret)})")
    
    # Validate key secret is not empty and has reasonable length
    if len(key_secret) < 10:
        update.message.reply_text(
            "❌ Invalid Key Secret!\n\n"
            "Key Secret seems too short. Please check and try again."
        )
        return AWAITING_RAZORPAY_KEY_SECRET
    
    set_setting('razorpay_key_secret', key_secret)
    logger.info(f"Razorpay key_secret saved successfully")
    
    # Delete the message containing the secret for security
    try:
        update.message.delete()
        logger.info("Secret message deleted for security")
    except:
        pass
    
    # Send confirmation
    update.message.reply_text(
        "✅ Razorpay Key Secret updated successfully!\n\n"
        "🔒 Your message was deleted for security.\n\n"
        "✨ Razorpay is now configured!\n"
        "Go to Settings → Razorpay → Enable to activate."
    )
    
    return ConversationHandler.END


def handle_wallet_add_amount(update: Update, context: CallbackContext) -> int:
    """Handle amount input for adding money to user wallet"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return ConversationHandler.END
    
    try:
        amount = float(update.message.text)
    except ValueError:
        update.message.reply_text("❌ Invalid amount. Please send a number.")
        return AWAITING_WALLET_ADD_AMOUNT
    
    if amount <= 0:
        update.message.reply_text("❌ Amount must be greater than 0.")
        return AWAITING_WALLET_ADD_AMOUNT
    
    target_user_id = context.user_data.get('wallet_target_user')
    if not target_user_id:
        update.message.reply_text("❌ Error: User ID not found. Please try again.")
        return ConversationHandler.END
    
    # Get current balance
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT wallet FROM users WHERE id = ?', (target_user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        update.message.reply_text(f"❌ User {target_user_id} not found in database.")
        return ConversationHandler.END
    
    old_balance = user[0]
    
    # Credit user wallet
    update_user_wallet(target_user_id, amount)
    new_balance = old_balance + amount
    
    update.message.reply_text(
        f"✅ **Balance Increased**\n\n"
        f"User ID: {target_user_id}\n"
        f"Amount Added: ₹{amount:.2f}\n\n"
        f"Old Balance: ₹{old_balance:.2f}\n"
        f"New Balance: ₹{new_balance:.2f}",
        parse_mode='Markdown'
    )
    
    # Notify user
    try:
        context.bot.send_message(
            chat_id=target_user_id,
            text=f"✅ **Wallet Credited**\n\n"
                 f"₹{amount:.2f} has been added to your wallet!\n\n"
                 f"New Balance: ₹{new_balance:.2f}\n\n"
                 f"You can now place orders.",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Failed to notify user {target_user_id}: {e}")
    
    return ConversationHandler.END


def handle_wallet_deduct_amount(update: Update, context: CallbackContext) -> int:
    """Handle amount input for deducting money from user wallet"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return ConversationHandler.END
    
    try:
        amount = float(update.message.text)
    except ValueError:
        update.message.reply_text("❌ Invalid amount. Please send a number.")
        return AWAITING_WALLET_DEDUCT_AMOUNT
    
    if amount <= 0:
        update.message.reply_text("❌ Amount must be greater than 0.")
        return AWAITING_WALLET_DEDUCT_AMOUNT
    
    target_user_id = context.user_data.get('wallet_target_user')
    if not target_user_id:
        update.message.reply_text("❌ Error: User ID not found. Please try again.")
        return ConversationHandler.END
    
    # Get current balance
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT wallet FROM users WHERE id = ?', (target_user_id,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        update.message.reply_text(f"❌ User {target_user_id} not found in database.")
        return ConversationHandler.END
    
    old_balance = user[0]
    
    if old_balance < amount:
        conn.close()
        update.message.reply_text(
            f"❌ **Insufficient Balance**\n\n"
            f"Cannot deduct ₹{amount:.2f}\n"
            f"User's current balance: ₹{old_balance:.2f}",
            parse_mode='Markdown'
        )
        return ConversationHandler.END
    
    # Deduct from user wallet
    new_balance = old_balance - amount
    cursor.execute('UPDATE users SET wallet = ? WHERE id = ?', (new_balance, target_user_id))
    conn.commit()
    conn.close()
    
    update.message.reply_text(
        f"✅ **Balance Deducted**\n\n"
        f"User ID: {target_user_id}\n"
        f"Amount Deducted: ₹{amount:.2f}\n\n"
        f"Old Balance: ₹{old_balance:.2f}\n"
        f"New Balance: ₹{new_balance:.2f}",
        parse_mode='Markdown'
    )
    
    # Notify user
    try:
        context.bot.send_message(
            chat_id=target_user_id,
            text=f"⚠️ **Wallet Deducted**\n\n"
                 f"₹{amount:.2f} has been deducted from your wallet.\n\n"
                 f"New Balance: ₹{new_balance:.2f}\n\n"
                 f"Contact support if you have questions.",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Failed to notify user {target_user_id}: {e}")
    
    return ConversationHandler.END


def refund_command(update: Update, context: CallbackContext) -> None:
    """Handle /refund command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    if len(context.args) < 1:
        update.message.reply_text(
            "❌ Usage: /refund <order_id>")
        return
    
    try:
        order_id = int(context.args[0])
    except ValueError:
        update.message.reply_text("❌ Invalid order ID.")
        return
    
    result = refund_order(order_id)
    
    if result['success']:
        update.message.reply_text(
            f"✅ Order {order_id} refunded successfully!\n\n₹{result['amount']:.2f} credited to user's wallet.")
    else:
        update.message.reply_text(f"❌ {result['error']}")


def support_command(update: Update, context: CallbackContext) -> None:
    """Handle /support command"""
    user = update.effective_user
    
    if len(context.args) < 1:
        update.message.reply_text(
            "❌ Usage: /support <your message>")
        return
    
    message = ' '.join(context.args)
    
    # Forward to admin
    try:
        user_info = get_user_info(user.id)
        context.bot.send_message(
            chat_id=ADMIN_ID,
            text=f"🧾 Support Request\n\nFrom: @{user_info['username']} ({user.id})\n\nMessage:\n{message}\n\nReply with: /reply {user.id} <message>"
        )
        
        update.message.reply_text(
            "✅ Your message has been sent to support!\n\nWe'll get back to you soon.")
    except Exception as e:
        logger.error(f"Failed to forward support message: {e}")
        update.message.reply_text("❌ Failed to send message. Please try again later.")


def handle_support_message(update: Update, context: CallbackContext) -> int:
    """Handle support message via button flow (text or photo with caption)"""
    user = update.effective_user
    
    # Get message text or caption
    if update.message.text:
        message = update.message.text.strip()
        photo_file_id = None
    elif update.message.photo and update.message.caption:
        message = update.message.caption.strip()
        photo_file_id = update.message.photo[-1].file_id  # Get highest resolution
    elif update.message.photo:
        message = "[Photo without caption]"
        photo_file_id = update.message.photo[-1].file_id
    else:
        update.message.reply_text("❌ Please send a text message or photo.")
        return AWAITING_SUPPORT_MESSAGE
    
    # Check if blocked user has exceeded daily ticket limit
    if is_user_blocked(user.id):
        can_send, tickets_sent = check_support_ticket_limit(user.id)
        if not can_send:
            keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data='back_to_main')]]
            update.message.reply_text(
                f"⚠️ Daily Limit Reached\n\n"
                f"You have sent {tickets_sent}/3 support tickets today.\n\n"
                f"Please try again tomorrow.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return ConversationHandler.END
    
    # Apply rate limiting (except for admins and blocked users sending support)
    if not is_admin(user.id) and not is_user_blocked(user.id):
        allowed, error_msg = check_rate_limit(user.id)
        if not allowed:
            update.message.reply_text(error_msg)
            return ConversationHandler.END
    
    if not message:
        update.message.reply_text("❌ Please send a valid message.")
        return AWAITING_SUPPORT_MESSAGE
    
    # Record the support ticket for blocked users
    if is_user_blocked(user.id):
        add_support_ticket(user.id, message)
    
    # Forward to admin
    try:
        user_info = get_user_info(user.id)
        blocked_status = " \\[BLOCKED\\]" if is_user_blocked(user.id) else ""
        can_send, tickets_sent = check_support_ticket_limit(user.id)
        
        support_text = (
            f"🧾 Support Request{blocked_status}\n\n"
            f"From: @{user_info['username']} ({user.id})\n"
            f"Tickets Today: {tickets_sent}/3\n\n"
            f"Message:\n{escape_markdown(message)}\n\n"
            f"Reply with: /reply {user.id} <message>"
        )
        
        # Send with photo if available
        if photo_file_id:
            context.bot.send_photo(
                chat_id=ADMIN_ID,
                photo=photo_file_id,
                caption=support_text
            )
        else:
            context.bot.send_message(
                chat_id=ADMIN_ID,
                text=support_text
            )
        
        text = "✅ Your message has been sent to support!\n\nWe'll get back to you soon."
        keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data='back_to_main')]]
        update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Failed to forward support message: {e}")
        update.message.reply_text("❌ Failed to send message. Please try again later.")
    
    return ConversationHandler.END


def reply_command(update: Update, context: CallbackContext) -> None:
    """Handle /reply command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    if len(context.args) < 2:
        update.message.reply_text(
            "❌ Usage: /reply <user_id> <message>")
        return
    
    try:
        user_id = int(context.args[0])
        message = ' '.join(context.args[1:])
    except ValueError:
        update.message.reply_text("❌ Invalid user ID.")
        return
    
    # Send to user
    try:
        context.bot.send_message(
            chat_id=user_id,
            text=f"💬 Support Reply\n\n{message}"
        )
        update.message.reply_text(f"✅ Reply sent to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to send reply to user {user_id}: {e}")
        update.message.reply_text("❌ Failed to send reply.")


def block_command(update: Update, context: CallbackContext) -> None:
    """Handle /block command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    if len(context.args) < 1:
        update.message.reply_text(
            "❌ Usage: /block <user_id> <reason>\n\nExample:\n/block 123456789 Duplicate UTR fraud")
        return
    
    try:
        user_id = int(context.args[0])
    except ValueError:
        update.message.reply_text("❌ Invalid user ID.")
        return
    
    if user_id == ADMIN_ID:
        update.message.reply_text("❌ Cannot block admin!")
        return
    
    # Get block reason (everything after user_id)
    reason = ' '.join(context.args[1:]) if len(context.args) > 1 else "Violation of terms"
    
    # Get user info
    user_info = get_user_info(user_id)
    if not user_info:
        update.message.reply_text(f"❌ User {user_id} not found.")
        return
    
    if block_user(user_id, reason):
        update.message.reply_text(
            f"✅ User Blocked\n\nUser: @{user_info['username']} ({user_id})\nReason: {escape_markdown(reason)}\n\nThe user can only access Support (3 tickets/day)."
        )
        
        # Notify the user with reason
        try:
            context.bot.send_message(
                chat_id=user_id,
                text=f"🚫 Account Blocked\n\nYour account has been blocked.\n\nReason: {escape_markdown(reason)}\n\nYou can send up to 3 support messages per day. Use /start to access support."
            )
        except:
            pass
    else:
        update.message.reply_text("❌ Failed to block user.")


def unblock_command(update: Update, context: CallbackContext) -> None:
    """Handle /unblock command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    if len(context.args) < 1:
        update.message.reply_text("❌ Usage: /unblock <user_id>")
        return
    
    try:
        user_id = int(context.args[0])
    except ValueError:
        update.message.reply_text("❌ Invalid user ID.")
        return
    
    # Get user info
    user_info = get_user_info(user_id)
    if not user_info:
        update.message.reply_text(f"❌ User {user_id} not found.")
        return
    
    if unblock_user(user_id):
        update.message.reply_text(
            f"✅ User Unblocked\n\nUser: @{user_info['username']} ({user_id})\n\nThe user can now use the bot again."
        )
        
        # Notify the user
        try:
            context.bot.send_message(
                chat_id=user_id,
                text="✅ Account Unblocked\n\nYour account has been unblocked. You can now use the bot again."
            )
        except:
            pass
    else:
        update.message.reply_text("❌ Failed to unblock user.")


def show_wallet_control_menu(query) -> int:
    """Show wallet control menu - asks for user ID"""
    keyboard = [[InlineKeyboardButton("🔙 Back to Admin", callback_data='admin_panel')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(
        "💳 **Wallet Control**\n\n"
        "Enter the User ID to manage their wallet:\n\n"
        "💡 Tip: You can find user IDs in:\n"
        "• User List\n"
        "• Pending Funds\n"
        "• Support messages\n\n"
        "Send user ID as a number:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    return AWAITING_USER_ID_FOR_WALLET


def handle_user_id_for_wallet(update: Update, context: CallbackContext) -> int:
    """Handle user ID input for wallet control"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return ConversationHandler.END
    
    try:
        user_id = int(update.message.text.strip())
    except ValueError:
        update.message.reply_text(
            "❌ Invalid user ID. Please send a valid number.\n\n"
            "Try again or use /start to cancel."
        )
        return AWAITING_USER_ID_FOR_WALLET
    
    # Get user info
    user_info = get_user_info(user_id)
    if not user_info:
        update.message.reply_text(
            f"❌ User {user_id} not found in database.\n\n"
            "Please check the ID and try again."
        )
        return AWAITING_USER_ID_FOR_WALLET
    
    is_blocked = is_user_blocked(user_id)
    status_emoji = "🚫" if is_blocked else "✅"
    status_text = "BLOCKED" if is_blocked else "ACTIVE"
    
    # Get block reason if user is blocked
    block_reason = get_block_reason(user_id) if is_blocked else None
    reason_line = f"Block Reason: {block_reason}\n" if block_reason else ""
    
    # Create wallet control buttons
    keyboard = [
        [
            InlineKeyboardButton("➕ Add Money", callback_data=f'wallet_add_{user_id}'),
            InlineKeyboardButton("➖ Deduct Money", callback_data=f'wallet_deduct_{user_id}')
        ],
        [InlineKeyboardButton("🔙 Back to Wallet Control", callback_data='admin_wallet_control')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        f"{status_emoji} **User Wallet Control**\n\n"
        f"👤 User: @{user_info['username']} ({user_id})\n"
        f"📊 Status: {status_text}\n"
        f"{reason_line}"
        f"💰 Wallet: ₹{user_info['wallet']:.2f}\n"
        f"📅 Joined: {user_info['join_date'][:10]}\n\n"
        f"Select an action:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    
    return ConversationHandler.END


def userstatus_command(update: Update, context: CallbackContext) -> None:
    """Handle /userstatus command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    if len(context.args) < 1:
        update.message.reply_text("❌ Usage: /userstatus <user_id>")
        return
    
    try:
        user_id = int(context.args[0])
    except ValueError:
        update.message.reply_text("❌ Invalid user ID.")
        return
    
    user_info = get_user_info(user_id)
    if not user_info:
        update.message.reply_text(f"❌ User {user_id} not found.")
        return
    
    is_blocked = is_user_blocked(user_id)
    status_emoji = "🚫" if is_blocked else "✅"
    status_text = "BLOCKED" if is_blocked else "ACTIVE"
    
    # Get block reason if user is blocked
    block_reason = get_block_reason(user_id) if is_blocked else None
    reason_line = f"Block Reason: {escape_markdown(block_reason)}\n" if block_reason else ""
    
    # Create wallet control buttons
    keyboard = [
        [
            InlineKeyboardButton("➕ Add Money", callback_data=f'wallet_add_{user_id}'),
            InlineKeyboardButton("➖ Deduct Money", callback_data=f'wallet_deduct_{user_id}')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        f"{status_emoji} User Status\n\n"
        f"User: @{user_info['username']} ({user_id})\n"
        f"Status: *{status_text}*\n"
        f"{reason_line}"
        f"Wallet: ₹{user_info['wallet']:.2f}\n"
        f"Joined: {user_info['join_date'][:10]}",
        reply_markup=reply_markup
    )


def blockedusers_command(update: Update, context: CallbackContext) -> None:
    """Handle /blockedusers command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, join_date, block_reason FROM users WHERE is_blocked = 1 ORDER BY join_date DESC LIMIT 20')
    blocked_users = cursor.fetchall()
    conn.close()
    
    if not blocked_users:
        update.message.reply_text("✅ No blocked users.")
        return
    
    text = "🚫 Blocked Users\n\n"
    for user in blocked_users:
        reason = user[3] if user[3] else "No reason specified"
        text += f"• @{user[1]} ({user[0]})\n"
        text += f"  Reason: {escape_markdown(reason)}\n"
        text += f"  Joined: {user[2][:10]}\n\n"
    
    text += f"\nTotal: {len(blocked_users)} user(s)"
    
    update.message.reply_text(text)


def set_upi_id_command(update: Update, context: CallbackContext) -> None:
    """Handle /set_upi_id command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    if len(context.args) < 1:
        update.message.reply_text("❌ Usage: /set_upi_id <upi@bank>")
        return
    
    upi_id = context.args[0]
    set_setting('upi_id', upi_id)
    update.message.reply_text(f"✅ UPI ID updated to: {upi_id}")


def set_upi_name_command(update: Update, context: CallbackContext) -> None:
    """Handle /set_upi_name command (admin only)"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    if len(context.args) < 1:
        update.message.reply_text("❌ Usage: /set_upi_name <name>")
        return
    
    upi_name = ' '.join(context.args)
    set_setting('upi_name', upi_name)
    update.message.reply_text(f"✅ UPI Name updated to: {upi_name}")


def handle_broadcast_message(update: Update, context: CallbackContext) -> int:
    """Handle broadcast message (text or photo) from admin"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return ConversationHandler.END
    
    # Get message text or caption
    if update.message.text:
        message = update.message.text.strip()
        photo_file_id = None
    elif update.message.photo and update.message.caption:
        message = update.message.caption.strip()
        photo_file_id = update.message.photo[-1].file_id  # Get highest resolution
    elif update.message.photo:
        message = ""  # Photo without caption
        photo_file_id = update.message.photo[-1].file_id
    else:
        update.message.reply_text("❌ Please send a text message or photo.")
        return AWAITING_BROADCAST_MESSAGE
    
    # Get all users
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users')
    users = cursor.fetchall()
    conn.close()
    
    sent_count = 0
    failed_count = 0
    
    # Send broadcast to all users
    for user in users:
        user_id = user[0]
        try:
            if photo_file_id:
                # Send as photo with caption
                if message:
                    context.bot.send_photo(
                        chat_id=user_id,
                        photo=photo_file_id,
                        caption=f"📣 **Broadcast**\n\n{message}",
                        parse_mode='Markdown'
                    )
                else:
                    context.bot.send_photo(
                        chat_id=user_id,
                        photo=photo_file_id
                    )
            else:
                # Send as text message
                context.bot.send_message(
                    chat_id=user_id,
                    text=f"📣 **Broadcast**\n\n{message}",
                    parse_mode='Markdown'
                )
            sent_count += 1
        except Exception as e:
            logger.error(f"Failed to send broadcast to {user_id}: {e}")
            failed_count += 1
    
    broadcast_type = "📷 Photo broadcast" if photo_file_id else "📝 Text broadcast"
    update.message.reply_text(
        f"✅ **{broadcast_type} completed!**\n\n"
        f"✓ Sent: {sent_count}\n"
        f"✗ Failed: {failed_count}",
        parse_mode='Markdown'
    )
    
    return ConversationHandler.END


# ==================== ANALYTICS DASHBOARD ====================

def show_analytics_dashboard(query) -> None:
    """Show comprehensive analytics dashboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get today's date
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # Total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # New users today
        cursor.execute("SELECT COUNT(*) FROM users WHERE DATE(join_date) = ?", (today,))
        users_today = cursor.fetchone()[0]
        
        # New users this week
        cursor.execute("SELECT COUNT(*) FROM users WHERE DATE(join_date) >= ?", (week_ago,))
        users_week = cursor.fetchone()[0]
        
        # Total orders
        cursor.execute("SELECT COUNT(*) FROM orders")
        total_orders = cursor.fetchone()[0]
        
        # Orders today
        cursor.execute("SELECT COUNT(*) FROM orders WHERE DATE(created) = ?", (today,))
        orders_today = cursor.fetchone()[0]
        
        # Total revenue
        cursor.execute("SELECT COALESCE(SUM(sell_price), 0) FROM orders WHERE status = 'Completed'")
        total_revenue = cursor.fetchone()[0]
        
        # Revenue today
        cursor.execute("SELECT COALESCE(SUM(sell_price), 0) FROM orders WHERE status = 'Completed' AND DATE(created) = ?", (today,))
        revenue_today = cursor.fetchone()[0]
        
        # Revenue this week
        cursor.execute("SELECT COALESCE(SUM(sell_price), 0) FROM orders WHERE status = 'Completed' AND DATE(created) >= ?", (week_ago,))
        revenue_week = cursor.fetchone()[0]
        
        # Revenue this month
        cursor.execute("SELECT COALESCE(SUM(sell_price), 0) FROM orders WHERE status = 'Completed' AND DATE(created) >= ?", (month_ago,))
        revenue_month = cursor.fetchone()[0]
        
        # Pending payments
        cursor.execute("SELECT COUNT(*) FROM pending_funds WHERE status = 'pending'")
        pending_payments = cursor.fetchone()[0]
        
        # Total pending amount
        cursor.execute("SELECT COALESCE(SUM(amount), 0) FROM pending_funds WHERE status = 'pending'")
        pending_amount = cursor.fetchone()[0]
        
        conn.close()
        
        text = f"""
📊 **Analytics Dashboard**
━━━━━━━━━━━━━━━━━━━━

👥 **User Statistics:**
   • Total Users: `{total_users}`
   • New Today: `{users_today}`
   • New This Week: `{users_week}`

📦 **Order Statistics:**
   • Total Orders: `{total_orders}`
   • Orders Today: `{orders_today}`

💰 **Revenue Statistics:**
   • Total Revenue: `₹{total_revenue:.2f}`
   • Today: `₹{revenue_today:.2f}`
   • This Week: `₹{revenue_week:.2f}`
   • This Month: `₹{revenue_month:.2f}`

⏳ **Pending Payments:**
   • Count: `{pending_payments}`
   • Amount: `₹{pending_amount:.2f}`

━━━━━━━━━━━━━━━━━━━━
📈 Select an option below:
"""
        
        keyboard = [
            [InlineKeyboardButton("📊 Top Services", callback_data='analytics_top_services')],
            [InlineKeyboardButton("👑 Top Users", callback_data='analytics_top_users')],
            [InlineKeyboardButton("📁 Export Users CSV", callback_data='export_users')],
            [InlineKeyboardButton("📁 Export Orders CSV", callback_data='export_orders')],
            [InlineKeyboardButton("🔙 Back to Admin", callback_data='admin_panel')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        query.edit_message_text(f"❌ Error loading analytics: {e}")


def show_top_services(query) -> None:
    """Show top 5 services by order count"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.name, COUNT(o.id) AS order_count,
                   COALESCE(SUM(o.sell_price), 0) AS total_revenue
            FROM services s
            LEFT JOIN orders o ON s.id = o.service_id
            GROUP BY s.id
            ORDER BY order_count DESC
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        text = """
📊 **Top 5 Services by Orders**
━━━━━━━━━━━━━━━━━━━━

"""
        
        if results:
            for idx, (service_name, order_count, revenue) in enumerate(results, 1):
                medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
                text += f"{medal} **{idx}. {service_name}**\n"
                text += f"   Orders: `{order_count}` | Revenue: `₹{revenue:.2f}`\n\n"
        else:
            text += "No data available yet.\n"
        
        text += "━━━━━━━━━━━━━━━━━━━━"
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to Analytics", callback_data='analytics_dashboard')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        safe_edit_message_text(query, text, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        query.edit_message_text(f"❌ Error loading top services: {e}")


def show_top_users(query) -> None:
    """Show top 5 users by total spending"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.username, COUNT(o.id) as order_count, 
                   COALESCE(SUM(o.sell_price), 0) as total_spent
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            GROUP BY u.id
            HAVING total_spent > 0
            ORDER BY total_spent DESC
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        text = """
👑 **Top 5 Users by Spending**
━━━━━━━━━━━━━━━━━━━━

"""
        
        if results:
            for idx, (user_id, username, order_count, total_spent) in enumerate(results, 1):
                medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
                username_display = f"@{username}" if username else f"ID: {user_id}"
                text += f"{medal} **{idx}. {username_display}**\n"
                text += f"   Orders: `{order_count}` | Spent: `₹{total_spent:.2f}`\n\n"
        else:
            text += "No data available yet.\n"
        
        text += "━━━━━━━━━━━━━━━━━━━━"
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to Analytics", callback_data='analytics_dashboard')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        query.edit_message_text(f"❌ Error loading top users: {e}")


def export_users_csv(query) -> None:
    """Export all users to CSV"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, username, wallet, join_date FROM users ORDER BY join_date DESC")
        users = cursor.fetchall()
        conn.close()
        
        # Create CSV in memory
        from io import StringIO
        import csv
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['User ID', 'Username', 'Wallet Balance', 'Joined Date'])
        
        for user in users:
            user_id, username, wallet, join_date = user
            username_display = username if username else 'N/A'
            writer.writerow([user_id, username_display, f"₹{wallet:.2f}", join_date])
        
        # Send CSV file
        csv_content = output.getvalue()
        from io import BytesIO
        csv_bytes = BytesIO(csv_content.encode('utf-8'))
        csv_bytes.name = f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        query.message.reply_document(
            document=csv_bytes,
            filename=csv_bytes.name,
            caption=f"📊 **Users Export**\n\nTotal Users: `{len(users)}`\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            parse_mode='Markdown'
        )
        
        # Also send back to analytics
        query.answer("✅ CSV exported successfully!")
        
    except Exception as e:
        query.answer(f"❌ Export failed: {e}", show_alert=True)


def export_orders_csv(query) -> None:
    """Export all orders to CSV"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT o.id, o.user_id, u.username, s.name, o.qty, 
                   o.sell_price, o.status, o.created
            FROM orders o
            LEFT JOIN users u ON o.user_id = u.id
            LEFT JOIN services s ON o.service_id = s.id
            ORDER BY o.created DESC
        """)
        orders = cursor.fetchall()
        conn.close()
        
        # Create CSV in memory
        from io import StringIO
        import csv
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Order ID', 'User ID', 'Username', 'Service', 'Quantity', 'Price', 'Status', 'Created At'])
        
        for order in orders:
            order_id, user_id, username, service_name, quantity, price, status, created_at = order
            username_display = username if username else 'N/A'
            writer.writerow([order_id, user_id, username_display, service_name, quantity, f"₹{price:.2f}", status, created_at])
        
        # Send CSV file
        csv_content = output.getvalue()
        from io import BytesIO
        csv_bytes = BytesIO(csv_content.encode('utf-8'))
        csv_bytes.name = f"orders_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        query.message.reply_document(
            document=csv_bytes,
            filename=csv_bytes.name,
            caption=f"📊 **Orders Export**\n\nTotal Orders: `{len(orders)}`\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            parse_mode='Markdown'
        )
        
        # Also send back to analytics
        query.answer("✅ CSV exported successfully!")
        
    except Exception as e:
        query.answer(f"❌ Export failed: {e}", show_alert=True)


def show_service_status_manager(query, page: int = 1) -> None:
    """Show service status management interface with pagination to avoid long texts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, service_status FROM services ORDER BY name")
        services = cursor.fetchall()
        conn.close()

        total = len(services)
        page_size = 10  # keep buttons within safe limits
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        services_page = services[start:end]

        text = f"""
⚙️ **Service Status Manager**
━━━━━━━━━━━━━━━━━━━━

Mark services with special status:
✅ Normal - Default status
⚠️ Maintenance - Slow/issues
🔥 Fast - Quick delivery

**Current Services (Page {page}/{total_pages}, showing {len(services_page)} of {total}):**

"""

        if services_page:
            for _, service_name, status in services_page:
                status_emoji = "⚠️" if status == "maintenance" else "🔥" if status == "fast" else "✅"
                text += f"{status_emoji} `{service_name}`\n"
        else:
            text += "No services available.\n"

        text += "\n━━━━━━━━━━━━━━━━━━━━"

        # Create buttons for each service in this page
        keyboard = []
        for service_id, service_name, status in services_page:
            # Use compact ID-based callback to avoid Telegram 64-byte limit
            keyboard.append([InlineKeyboardButton(f"⚙️ {service_name[:40]}", callback_data=f'service_status_{service_id}')])

        # Pagination controls
        nav_row = []
        if page > 1:
            nav_row.append(InlineKeyboardButton("⬅️ Prev", callback_data=f'service_status_manager_p{page-1}'))
        if page < total_pages:
            nav_row.append(InlineKeyboardButton("Next ➡️", callback_data=f'service_status_manager_p{page+1}'))
        if nav_row:
            keyboard.append(nav_row)

        keyboard.append([InlineKeyboardButton("🔙 Back to Admin", callback_data='admin_panel')])
        reply_markup = InlineKeyboardMarkup(keyboard)

        query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    except Exception as e:
        query.edit_message_text(f"❌ Error loading services: {e}")


def show_service_status_options(query, service_id: int) -> None:
    """Show status options for a specific service (by ID to keep callbacks short)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, service_status FROM services WHERE id = ?", (service_id,))
        result = cursor.fetchone()
        service_name = result[0] if result else "Unknown"
        current_status = result[1] if result else "normal"
        conn.close()
        
        status_display = "⚠️ Maintenance" if current_status == "maintenance" else "🔥 Fast" if current_status == "fast" else "✅ Normal"
        
        text = f"""
⚙️ **Service Status Manager**
━━━━━━━━━━━━━━━━━━━━

**Service:** `{service_name}`
**Current Status:** {status_display}

Select new status:
"""
        
        keyboard = [
            [InlineKeyboardButton("✅ Normal", callback_data=f'set_status_{service_id}_normal')],
            [InlineKeyboardButton("⚠️ Maintenance", callback_data=f'set_status_{service_id}_maintenance')],
            [InlineKeyboardButton("🔥 Fast Delivery", callback_data=f'set_status_{service_id}_fast')],
            [InlineKeyboardButton("🔙 Back", callback_data='service_status_manager')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        query.edit_message_text(f"❌ Error: {e}")


def set_service_status(query, service_id: int, new_status: str) -> None:
    """Update service status in database (by ID)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE services SET service_status = ? WHERE id = ?", (new_status, service_id))
        conn.commit()
        conn.close()
        
        status_display = "⚠️ Maintenance" if new_status == "maintenance" else "🔥 Fast" if new_status == "fast" else "✅ Normal"
        
        query.answer(f"✅ Status updated to {status_display}", show_alert=True)
        show_service_status_manager(query)
        
    except Exception as e:
        query.answer(f"❌ Error: {e}", show_alert=True)




def check_orders_command(update: Update, context: CallbackContext) -> None:
    """Handle /checkorders command (admin only) - manually trigger order status checker"""
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    update.message.reply_text("🔄 Checking all pending orders...\n\nPlease wait...")
    
    try:
        # Run the automated checker
        automated_order_status_checker(context.bot)
        update.message.reply_text("✅ Order status check completed!\n\nCheck logs for details.")
    except Exception as e:
        logger.error(f"Error in manual order check: {e}")
        update.message.reply_text(f"❌ Error checking orders: {str(e)}")


def broadcast_command(update: Update, context: CallbackContext) -> None:
    """Handle /broadcast command (admin only) - supports text and photos
    Usage: 
    - Text only: /broadcast <message>
    - Reply to a photo: /broadcast <caption> (reply to photo message)
    """
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ This command is for admins only.")
        return
    
    # Check if replying to a photo
    replied_message = update.message.reply_to_message
    photo_file_id = None
    
    if replied_message and replied_message.photo:
        photo_file_id = replied_message.photo[-1].file_id  # Get highest resolution
    
    # Get message/caption
    if len(context.args) < 1:
        update.message.reply_text(
            "❌ Usage:\n"
            "• Text: /broadcast <message>\n"
            "• Photo: Reply to a photo with /broadcast <caption>"
        )
        return
    
    message = ' '.join(context.args)
    
    # Get all users
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users')
    users = cursor.fetchall()
    conn.close()
    
    sent_count = 0
    failed_count = 0
    
    for user in users:
        user_id = user[0]
        try:
            if photo_file_id:
                # Send as photo with caption
                context.bot.send_photo(
                    chat_id=user_id,
                    photo=photo_file_id,
                    caption=f"📣 **Broadcast**\n\n{message}",
                    parse_mode='Markdown'
                )
            else:
                # Send as text message
                context.bot.send_message(
                    chat_id=user_id,
                    text=f"📣 **Broadcast**\n\n{message}",
                    parse_mode='Markdown'
                )
            sent_count += 1
        except Exception as e:
            logger.error(f"Failed to send broadcast to {user_id}: {e}")
            failed_count += 1
    
    broadcast_type = "📷 Photo broadcast" if photo_file_id else "📝 Text broadcast"
    update.message.reply_text(
        f"✅ {broadcast_type} completed!\n\n✓ Sent: {sent_count}\n✗ Failed: {failed_count}")


# ======================== SCHEDULED TASKS ========================

def scheduled_service_sync(context):
    """Scheduled service sync task"""
    logger.info("Running scheduled service sync...")
    sync_services()


# ======================== ERROR HANDLER ========================

def error_handler(update: Update, context: CallbackContext) -> None:
    """Handle errors with improved network error handling"""
    error = context.error
    
    # Check if it's a network-related error
    if isinstance(error, Exception):
        error_str = str(error).lower()
        
        # Network connection errors - don't spam logs, just log once
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'reset', 'aborted', 'unreachable']):
            logger.warning(f"Network error (will auto-retry): {error.__class__.__name__}")
            # Don't notify user for network errors - they're handled automatically
            return
        
        # Telegram API errors
        if 'telegram' in error_str or 'badrequest' in error_str:
            logger.error(f"Telegram API error: {error}")
            if update and update.effective_message:
                try:
                    update.effective_message.reply_text(
                        "⚠️ Telegram API error. Please try again in a moment.")
                except:
                    pass
            return
    
    # Log all other errors normally
    logger.error(f"Update {update} caused error {error}")
    
    # Try to notify user if possible
    if update and update.effective_message:
        try:
            update.effective_message.reply_text(
                "❌ An error occurred. Please try again later.")
        except:
            pass  # If we can't send message, just log it


# ======================== MAIN ========================

def main():
    """Start the bot"""
    # Initialize database
    init_database()
    
    # Create updater with improved request timeout settings
    from telegram.ext import Defaults
    from telegram.utils.request import Request
    
    # Configure request with optimized timeouts and retry logic
    request = Request(
        connect_timeout=10.0,  # Time to establish connection
        read_timeout=15.0,     # Time to wait for response
        con_pool_size=8        # Connection pool size for better handling
    )
    updater = Updater(
        BOT_TOKEN, 
        use_context=True,
        request_kwargs={'connect_timeout': 30.0, 'read_timeout': 30.0}
    )
    dp = updater.dispatcher
    
    # Register handlers
    dp.add_handler(CommandHandler('start', start))
    
    # Admin commands only (wallet control now button-based via admin panel)
    dp.add_handler(CommandHandler('refund', refund_command))
    dp.add_handler(CommandHandler('reply', reply_command))
    dp.add_handler(CommandHandler('block', block_command))
    dp.add_handler(CommandHandler('unblock', unblock_command))
    dp.add_handler(CommandHandler('blockedusers', blockedusers_command))
    dp.add_handler(CommandHandler('broadcast', broadcast_command))
    dp.add_handler(CommandHandler('checkorders', check_orders_command))
    dp.add_handler(CommandHandler('set_upi_id', set_upi_id_command))
    dp.add_handler(CommandHandler('set_upi_name', set_upi_name_command))
    
    # Main conversation handler with callback queries
    main_conversation = ConversationHandler(
        entry_points=[CallbackQueryHandler(button_handler)],
        states={
            AWAITING_PAYMENT_PROOF: [
                MessageHandler(Filters.photo, handle_payment_proof),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_UTR_NUMBER: [
                MessageHandler(Filters.text & ~Filters.command, handle_utr_number),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_PAYMENT_AMOUNT: [
                MessageHandler(Filters.text & ~Filters.command, handle_payment_amount),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_RAZORPAY_AMOUNT: [
                MessageHandler(Filters.text & ~Filters.command, handle_razorpay_amount),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_SUPPORT_MESSAGE: [
                MessageHandler(Filters.text & ~Filters.command, handle_support_message),
                MessageHandler(Filters.photo, handle_support_message),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_ORDER_QUANTITY: [
                MessageHandler(Filters.text & ~Filters.command, handle_order_quantity),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_ORDER_LINK: [
                MessageHandler(Filters.text & ~Filters.command, handle_order_link),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_QR_CODE: [
                MessageHandler(Filters.photo, handle_qr_code_upload),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_RAZORPAY_KEY_ID: [
                MessageHandler(Filters.text & ~Filters.command, handle_razorpay_key_id_input),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_RAZORPAY_KEY_SECRET: [
                MessageHandler(Filters.text & ~Filters.command, handle_razorpay_key_secret_input),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_WALLET_ADD_AMOUNT: [
                MessageHandler(Filters.text & ~Filters.command, handle_wallet_add_amount),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_WALLET_DEDUCT_AMOUNT: [
                MessageHandler(Filters.text & ~Filters.command, handle_wallet_deduct_amount),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_USER_ID_FOR_WALLET: [
                MessageHandler(Filters.text & ~Filters.command, handle_user_id_for_wallet),
                CallbackQueryHandler(button_handler)
            ],
            AWAITING_BROADCAST_MESSAGE: [
                MessageHandler(Filters.text & ~Filters.command, handle_broadcast_message),
                MessageHandler(Filters.photo, handle_broadcast_message),
                CallbackQueryHandler(button_handler)
            ]
        },
        fallbacks=[
            CallbackQueryHandler(button_handler),
            CommandHandler('start', start)
        ],
        allow_reentry=True,
        per_message=False
    )
    dp.add_handler(main_conversation)
    
    # Fallback callback query handler
    dp.add_handler(CallbackQueryHandler(button_handler))
    
    # Error handler
    dp.add_error_handler(error_handler)
    
    # Setup scheduler for service sync
    from pytz import utc
    scheduler = BackgroundScheduler(timezone=utc)
    
    # Schedule regular service sync every 2 hours
    scheduler.add_job(
        scheduled_service_sync,
        'interval',
        hours=2,
        args=[updater]
    )
    
    # Schedule new service detection and notification every 30 minutes
    scheduler.add_job(
        check_and_notify_new_services,
        'interval',
        minutes=30,
        args=[updater.bot]
    )
    
    # Schedule automated order status checker every 5 minutes (faster checking)
    scheduler.add_job(
        automated_order_status_checker,
        'interval',
        minutes=5,
        args=[updater.bot]
    )
    
    # Run initial service sync immediately (after 2 seconds)
    scheduler.add_job(
        sync_services,
        'date',
        run_date=datetime.now() + __import__('datetime').timedelta(seconds=2)
    )
    
    # Run initial order status check immediately (after 5 seconds)
    scheduler.add_job(
        automated_order_status_checker,
        'date',
        run_date=datetime.now() + __import__('datetime').timedelta(seconds=5),
        args=[updater.bot]
    )
    
    scheduler.start()
    
    # ======================== HYBRID WEBHOOK/POLLING SYSTEM ========================
    
    import threading
    import socket
    
    # Webhook configuration
    WEBHOOK_URL = os.getenv('WEBHOOK_URL', 'https://your-domain.com')
    WEBHOOK_PORT = int(os.getenv('PORT', os.getenv('WEBHOOK_PORT', '8443')))
    ENABLE_HYBRID_MODE = os.getenv('ENABLE_HYBRID_MODE', 'true').lower() == 'true'
    CHECK_INTERVAL_MINUTES = 10  # Check webhook availability every 10 minutes
    
    current_mode = {'mode': None, 'lock': threading.Lock()}  # Track current mode
    
    def is_webhook_reachable(url: str, timeout: int = 5) -> bool:
        """Check if webhook URL is reachable via HTTP HEAD request"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            
            # Quick TCP socket check (faster than HTTP request)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                # Host is reachable, do a quick HTTP check
                import requests
                response = requests.head(url, timeout=timeout, allow_redirects=True)
                return response.status_code < 500  # Accept any non-server-error
            return False
        except Exception as e:
            logger.debug(f"Webhook check failed: {e}")
            return False
    
    def start_webhook_mode():
        """Start bot in webhook mode"""
        with current_mode['lock']:
            if current_mode['mode'] == 'webhook':
                return  # Already in webhook mode
            
            # Stop polling if running
            if current_mode['mode'] == 'polling':
                try:
                    updater.stop()
                    logger.info("🔄 Stopped polling mode")
                except:
                    pass
            
            # Start webhook
            try:
                logger.info(f"🌐 Starting WEBHOOK mode on port {WEBHOOK_PORT}...")
                updater.start_webhook(
                    listen="0.0.0.0",
                    port=WEBHOOK_PORT,
                    url_path="webhook",
                    webhook_url=f"{WEBHOOK_URL}/webhook",
                    allowed_updates=['message', 'callback_query'],
                    drop_pending_updates=True
                )
                current_mode['mode'] = 'webhook'
                logger.info(f"✅ WEBHOOK MODE ACTIVE: {WEBHOOK_URL}/webhook")
            except Exception as e:
                logger.error(f"Failed to start webhook: {e}")
                start_polling_mode()  # Fallback to polling
    
    def start_polling_mode():
        """Start bot in polling mode"""
        with current_mode['lock']:
            if current_mode['mode'] == 'polling':
                return  # Already in polling mode
            
            # Stop webhook if running
            if current_mode['mode'] == 'webhook':
                try:
                    updater.stop()
                    logger.info("🔄 Stopped webhook mode")
                except:
                    pass
            
            # Start polling
            try:
                logger.info("📡 Starting POLLING mode...")
                updater.start_polling(
                    timeout=30,
                    drop_pending_updates=True,
                    allowed_updates=['message', 'callback_query']
                )
                current_mode['mode'] = 'polling'
                logger.info("✅ POLLING MODE ACTIVE")
            except Exception as e:
                logger.error(f"Failed to start polling: {e}")
                # Retry with exponential backoff
                import time
                for attempt in range(3):
                    time.sleep(2 ** attempt)
                    try:
                        updater.start_polling(
                            timeout=30,
                            drop_pending_updates=True,
                            allowed_updates=['message', 'callback_query']
                        )
                        current_mode['mode'] = 'polling'
                        logger.info("✅ POLLING MODE ACTIVE (after retry)")
                        break
                    except:
                        continue
    
    def check_and_switch_mode():
        """Periodically check webhook availability and switch modes"""
        while True:
            try:
                import time
                time.sleep(CHECK_INTERVAL_MINUTES * 60)  # Wait 10 minutes
                
                webhook_available = is_webhook_reachable(WEBHOOK_URL)
                
                with current_mode['lock']:
                    if webhook_available and current_mode['mode'] != 'webhook':
                        logger.info("🔄 Webhook is now reachable, switching to WEBHOOK mode...")
                        start_webhook_mode()
                    elif not webhook_available and current_mode['mode'] != 'polling':
                        logger.info("🔄 Webhook unreachable, switching to POLLING mode...")
                        start_polling_mode()
                    else:
                        logger.debug(f"✓ Mode check: {current_mode['mode'].upper()} mode still optimal")
            except Exception as e:
                logger.error(f"Error in mode checker: {e}")
    
    # Initial mode selection
    if ENABLE_HYBRID_MODE:
        logger.info("🔀 HYBRID MODE ENABLED - Auto-switching between webhook and polling")
        
        # Check if webhook is reachable
        if is_webhook_reachable(WEBHOOK_URL):
            logger.info("✓ Webhook URL is reachable")
            start_webhook_mode()
        else:
            logger.info("✗ Webhook URL not reachable, using polling")
            start_polling_mode()
        
        # Start background thread to monitor and switch modes
        monitor_thread = threading.Thread(target=check_and_switch_mode, daemon=True)
        monitor_thread.start()
        logger.info(f"🔍 Mode monitor active (checking every {CHECK_INTERVAL_MINUTES} minutes)")
    else:
        # Legacy mode - use environment variable
        USE_WEBHOOK = os.getenv('USE_WEBHOOK', 'true').lower() == 'true'
        if USE_WEBHOOK:
            start_webhook_mode()
        else:
            start_polling_mode()
    
    # Keep bot running and handle connection errors gracefully
    try:
        updater.idle()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Bot idle error: {e}")
        logger.info("Bot continues running with scheduler...")
    
    # Shutdown scheduler
    scheduler.shutdown()


if __name__ == '__main__':
    print("🚀 Starting SVXHUB SMM Bot...")
    print("=" * 50)
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
