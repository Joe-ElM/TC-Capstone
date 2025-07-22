import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from schemas import UserProfile

#=============================================================================
# DATABASE MANAGER CLASS
#=============================================================================

class DatabaseManager:
    """Manages SQLite database operations for user profiles and conversation history"""
    
    def __init__(self, db_path: str = "health_agent.db"):
        self.db_path = db_path
        self.init_database()
    
    #=========================================================================
    # DATABASE INITIALIZATION
    #=========================================================================
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    personal_info TEXT,
                    medical_history TEXT,
                    chronic_conditions TEXT,
                    medications TEXT,
                    dietary_restrictions TEXT,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP
                )
            """)
            
            # Conversation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    conversation_data TEXT,
                    summary TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            conn.commit()
    
    #=========================================================================
    # CONNECTION MANAGEMENT
    #=========================================================================
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    #=========================================================================
    # USER PROFILE OPERATIONS
    #=========================================================================
    
    def create_user_profile(self, user_profile: UserProfile) -> bool:
        """Create new user profile"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_profiles 
                    (user_id, personal_info, medical_history, chronic_conditions, 
                     medications, dietary_restrictions, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_profile.user_id,
                    json.dumps(user_profile.personal_info),
                    json.dumps(user_profile.medical_history),
                    json.dumps(user_profile.chronic_conditions),
                    json.dumps(user_profile.medications),
                    json.dumps(user_profile.dietary_restrictions),
                    user_profile.created_at,
                    user_profile.last_updated
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_profiles WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return UserProfile(
                        user_id=row[0],
                        personal_info=json.loads(row[1]),
                        medical_history=json.loads(row[2]),
                        chronic_conditions=json.loads(row[3]),
                        medications=json.loads(row[4]),
                        dietary_restrictions=json.loads(row[5]),
                        created_at=row[6],
                        last_updated=row[7],
                        conversation_summaries=[]  # Load separately if needed
                    )
        except Exception as e:
            print(f"Error retrieving user profile: {e}")
        return None
    
    def update_user_profile(self, user_profile: UserProfile) -> bool:
        """Update existing user profile"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_profiles SET
                    personal_info = ?, medical_history = ?, chronic_conditions = ?,
                    medications = ?, dietary_restrictions = ?, last_updated = ?
                    WHERE user_id = ?
                """, (
                    json.dumps(user_profile.personal_info),
                    json.dumps(user_profile.medical_history),
                    json.dumps(user_profile.chronic_conditions),
                    json.dumps(user_profile.medications),
                    json.dumps(user_profile.dietary_restrictions),
                    datetime.now(),
                    user_profile.user_id
                ))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating user profile: {e}")
            return False
    
    #=========================================================================
    # CONVERSATION OPERATIONS
    #=========================================================================
    
    def save_conversation(self, conversation_id: str, user_id: str, 
                         conversation_data: Dict[str, Any], summary: str = "") -> bool:
        """Save conversation to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO conversations 
                    (conversation_id, user_id, conversation_data, summary, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    conversation_id,
                    user_id,
                    json.dumps(conversation_data),
                    summary,
                    datetime.now()
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations for user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT conversation_id, conversation_data, summary, created_at 
                    FROM conversations 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (user_id, limit))
                
                conversations = []
                for row in cursor.fetchall():
                    conversations.append({
                        'conversation_id': row[0],
                        'conversation_data': json.loads(row[1]),
                        'summary': row[2],
                        'created_at': row[3]
                    })
                return conversations
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    #=========================================================================
    # UTILITY METHODS
    #=========================================================================
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user profile exists"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM user_profiles WHERE user_id = ?", (user_id,))
                return cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return False