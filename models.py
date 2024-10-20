import mysql.connector
from mysql.connector import Error, pooling, IntegrityError
import json
from datetime import datetime, timedelta
import logger_config
from flask_login import UserMixin
import sys
import os
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

# Add the directory above the current one to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

logger = logger_config.get_logger(__name__)

class DatabasePool:
    _pool = None

    @classmethod
    def get_pool(cls):
        if cls._pool is None:
            try:
                cls._pool = pooling.MySQLConnectionPool(
                    pool_name="tiktok_pool",
                    pool_size=5,
                    host=Config.DB_HOST,
                    database=Config.DB_NAME,
                    user=Config.DB_USER,
                    password=Config.DB_PASSWORD,
                    autocommit=True,  # Enable autocommit
                    consume_results=True  # Automatically consume results
                )
                logger.info("Database connection pool created")
            except Error as e:
                logger.error(f"Error creating database pool: {e}")
                raise
        return cls._pool

class Database:
    def __init__(self):
        self.connection = None
        self.cursor = None

    def __enter__(self):
        try:
            self.connection = DatabasePool.get_pool().get_connection()
            self.cursor = self.connection.cursor(buffered=True)  # Use buffered cursor
            logger.info("Acquired database connection from pool")
            return self
        except Error as e:
            logger.error(f"Error acquiring database connection: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("Released database connection back to pool")

    def execute_query(self, query, params=None):
        try:
            logger.debug(f"Executing query: {query}")
            if params:
                logger.debug(f"Query parameters: {params}")
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            logger.info("Query executed successfully")
            return self.cursor
        except Error as e:
            logger.error(f"Error executing query: {e}")
            raise

class ProcessedVideo:
    def __init__(self, id, url, task_id, video_id=None, video_duration=None, processing_time=None, results=None):
        self.id = id
        self.url = url
        self.task_id = task_id
        self.video_id = video_id
        self.video_duration = video_duration
        self.processing_time = processing_time
        self.results = results
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    @staticmethod
    def create_table():
        logger.info("Attempting to create processed_videos table")
        with Database() as db:
            query = """
            CREATE TABLE IF NOT EXISTS processed_videos (
                id INT AUTO_INCREMENT PRIMARY KEY,
                url VARCHAR(255) NOT NULL UNIQUE,
                task_id VARCHAR(36) NOT NULL UNIQUE,
                video_id VARCHAR(255),
                video_duration FLOAT,
                processing_time FLOAT,
                results JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
            try:
                db.execute_query(query)
                logger.info("processed_videos table created successfully")
            except Error as e:
                logger.error(f"Error creating processed_videos table: {e}")
                raise

    @staticmethod
    def add(url, task_id):
        logger.info(f"Adding new processed video: URL={url}, Task ID={task_id}")
        with Database() as db:
            query = """
            INSERT INTO processed_videos (url, task_id)
            VALUES (%s, %s)
            """
            params = (url, task_id)
            try:
                cursor = db.execute_query(query, params)
                new_id = cursor.lastrowid  # Get the ID of the newly inserted row
                logger.info(f"Processed video added successfully with ID: {new_id}")
                return ProcessedVideo(id=new_id, url=url, task_id=task_id)
            except Error as e:
                logger.error(f"Error adding processed video: {e}")
                raise

    @staticmethod
    def get_by_url(url):
        logger.info(f"Retrieving processed video by URL: {url}")
        with Database() as db:
            query = "SELECT * FROM processed_videos WHERE url = %s"
            try:
                cursor = db.execute_query(query, (url,))
                result = cursor.fetchone()
                if result:
                    id, url, task_id, video_id, video_duration, processing_time, results, created_at, updated_at = result
                    logger.info(f"Found processed video: Task ID={task_id}")
                    return ProcessedVideo(id, url, task_id, video_id, video_duration, processing_time, json.loads(results) if results else None)
                logger.info("No processed video found for the given URL")
                return None
            except Error as e:
                logger.error(f"Error retrieving processed video: {e}")
                raise

    @staticmethod
    def update_results(url, results, video_id=None, video_duration=None, processing_time=None):
        logger.info(f"Updating results for processed video: URL={url}")
        with Database() as db:
            query = """
            UPDATE processed_videos 
            SET results = %s, video_id = %s, video_duration = %s, processing_time = %s, updated_at = %s 
            WHERE url = %s
            """
            params = (json.dumps(results), video_id, video_duration, processing_time, datetime.now(), url)
            try:
                db.execute_query(query, params)
                logger.info("Results updated successfully")
            except Error as e:
                logger.error(f"Error updating results: {e}")
                raise

    @staticmethod
    def get_by_task_id(task_id):
        logger.info(f"Retrieving processed video by task ID: {task_id}")
        with Database() as db:
            query = "SELECT * FROM processed_videos WHERE task_id = %s"
            try:
                cursor = db.execute_query(query, (task_id,))
                result = cursor.fetchone()
                if result:
                    id, url, task_id, video_id, video_duration, processing_time, results, created_at, updated_at = result
                    logger.info(f"Found processed video: URL={url}")
                    return ProcessedVideo(id, url, task_id, video_id, video_duration, processing_time, json.loads(results) if results else None)
                logger.info("No processed video found for the given task ID")
                return None
            except Error as e:
                logger.error(f"Error retrieving processed video: {e}")
                raise

class User(UserMixin):
    def __init__(self, id, email, password_hash):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.created_at = datetime.now()
        self.last_login = None
        self.reset_token = None
        self.reset_token_expiry = None

    def get_id(self):
        return str(self.id)

    @staticmethod
    def create_table():
        logger.info("Attempting to create users table")
        with Database() as db:
            query = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL,
                reset_token VARCHAR(100) NULL,
                reset_token_expiry TIMESTAMP NULL
            )
            """
            try:
                db.execute_query(query)
                logger.info("users table created successfully")
            except Error as e:
                logger.error(f"Error creating users table: {e}")
                raise

    @staticmethod
    def add(email, password):
        logger.info(f"Adding new user: {email}")
        with Database() as db:
            query = """
            INSERT INTO users (email, password_hash, created_at)
            VALUES (%s, %s, %s)
            """
            password_hash = generate_password_hash(password)
            params = (email, password_hash, datetime.now())
            try:
                db.execute_query(query, params)
                logger.info("User added successfully")
                # Fetch the newly created user
                return User.get_by_email(email)
            except Error as e:
                logger.error(f"Error adding user: {e}")
                raise

    @staticmethod
    def get_by_username(username):
        logger.info(f"Retrieving user by username: {username}")
        with Database() as db:
            query = "SELECT * FROM users WHERE username = %s"
            try:
                cursor = db.execute_query(query, (username,))
                result = cursor.fetchone()
                if result:
                    id, username, email, password_hash, created_at, last_login = result
                    user = User(username, email, '')
                    user.password_hash = password_hash
                    user.created_at = created_at
                    user.last_login = last_login
                    return user
                return None
            except Error as e:
                logger.error(f"Error retrieving user: {e}")
                raise

    @staticmethod
    def get_by_email(email):
        logger.info(f"Retrieving user by email: {email}")
        with Database() as db:
            query = "SELECT id, email, password_hash, created_at, last_login FROM users WHERE email = %s"
            try:
                cursor = db.execute_query(query, (email,))
                result = cursor.fetchone()
                if result:
                    id, email, password_hash, created_at, last_login = result
                    user = User(id, email, password_hash)
                    user.created_at = created_at
                    user.last_login = last_login
                    return user
                return None
            except Error as e:
                logger.error(f"Error retrieving user: {e}")
                raise

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def update_last_login(user_id):
        logger.info(f"Updating last login for user ID: {user_id}")
        with Database() as db:
            query = "UPDATE users SET last_login = %s WHERE id = %s"
            params = (datetime.now(), user_id)
            try:
                db.execute_query(query, params)
                logger.info("Last login updated successfully")
            except Error as e:
                logger.error(f"Error updating last login: {e}")
                raise

    @staticmethod
    def get_by_id(user_id):
        logger.info(f"Retrieving user by id: {user_id}")
        with Database() as db:
            query = "SELECT id, email, password_hash, created_at, last_login FROM users WHERE id = %s"
            try:
                cursor = db.execute_query(query, (user_id,))
                result = cursor.fetchone()
                if result:
                    id, email, password_hash, created_at, last_login = result
                    user = User(id, email, password_hash)
                    user.created_at = created_at
                    user.last_login = last_login
                    return user
                return None
            except Error as e:
                logger.error(f"Error retrieving user: {e}")
                raise

    def set_password_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiry = datetime.now() + timedelta(hours=1)
        with Database() as db:
            query = "UPDATE users SET reset_token = %s, reset_token_expiry = %s WHERE id = %s"
            params = (self.reset_token, self.reset_token_expiry, self.id)
            try:
                db.execute_query(query, params)
                logger.info(f"Password reset token set for user ID: {self.id}")
            except Error as e:
                logger.error(f"Error setting password reset token: {e}")
                raise

    @staticmethod
    def get_by_reset_token(token):
        logger.info(f"Retrieving user by reset token")
        with Database() as db:
            query = """
            SELECT id, email, password_hash, created_at, last_login, reset_token_expiry 
            FROM users WHERE reset_token = %s
            """
            try:
                cursor = db.execute_query(query, (token,))
                result = cursor.fetchone()
                if result:
                    id, email, password_hash, created_at, last_login, reset_token_expiry = result
                    user = User(id, email, password_hash)
                    user.created_at = created_at
                    user.last_login = last_login
                    user.reset_token = token
                    user.reset_token_expiry = reset_token_expiry
                    return user
                return None
            except Error as e:
                logger.error(f"Error retrieving user by reset token: {e}")
                raise

    def reset_password(self, new_password):
        self.password_hash = generate_password_hash(new_password)
        with Database() as db:
            query = """
            UPDATE users 
            SET password_hash = %s, reset_token = NULL, reset_token_expiry = NULL 
            WHERE id = %s
            """
            params = (self.password_hash, self.id)
            try:
                db.execute_query(query, params)
                logger.info(f"Password reset for user ID: {self.id}")
            except Error as e:
                logger.error(f"Error resetting password: {e}")
                raise

class UserProcessedVideo:
    def __init__(self, user_id, processed_video_id, created_at=None):
        self.user_id = user_id
        self.processed_video_id = processed_video_id
        self.created_at = created_at or datetime.now()

    @staticmethod
    def create_table():
        logger.info("Attempting to create user_processed_videos table")
        with Database() as db:
            query = """
            CREATE TABLE IF NOT EXISTS user_processed_videos (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                processed_video_id INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (processed_video_id) REFERENCES processed_videos(id),
                UNIQUE KEY user_video_unique (user_id, processed_video_id)
            )
            """
            try:
                db.execute_query(query)
                logger.info("user_processed_videos table created successfully")
            except Error as e:
                logger.error(f"Error creating user_processed_videos table: {e}")
                raise

    @staticmethod
    def add(user_id, processed_video_id):
        logger.info(f"Adding new user processed video: user_id={user_id}, processed_video_id={processed_video_id}")
        logger.info(f"Type of user_id: {type(user_id)}, Type of processed_video_id: {type(processed_video_id)}")
        with Database() as db:
            query = """
            INSERT INTO user_processed_videos (user_id, processed_video_id)
            VALUES (%s, %s)
            """
            params = (user_id, processed_video_id)
            try:
                db.execute_query(query, params)
                logger.info("User processed video added successfully")
            except IntegrityError as e:
                logger.warning(f"User has already processed this video: {e}")
            except Error as e:
                logger.error(f"Error adding user processed video: {e}")
                raise

    @staticmethod
    def get_by_user(user_id):
        logger.info(f"Retrieving processed videos for user_id: {user_id}")
        with Database() as db:
            query = """
            SELECT pv.* FROM user_processed_videos upv
            JOIN processed_videos pv ON upv.processed_video_id = pv.id
            WHERE upv.user_id = %s
            ORDER BY upv.created_at DESC
            """
            try:
                cursor = db.execute_query(query, (user_id,))
                results = cursor.fetchall()
                return [ProcessedVideo(*row) for row in results]
            except Error as e:
                logger.error(f"Error retrieving user processed videos: {e}")
                raise

    @staticmethod
    def get_by_user_and_url(user_id, url):
        logger.info(f"Retrieving processed video for user_id: {user_id} and url: {url}")
        with Database() as db:
            query = """
            SELECT pv.* FROM user_processed_videos upv
            JOIN processed_videos pv ON upv.processed_video_id = pv.id
            WHERE upv.user_id = %s AND pv.url = %s
            """
            try:
                cursor = db.execute_query(query, (user_id, url))
                result = cursor.fetchone()
                if result:
                    return ProcessedVideo(*result)
                return None
            except Error as e:
                logger.error(f"Error retrieving user processed video: {e}")
                raise

# Create the table when this module is imported
UserProcessedVideo.create_table()

# Create the table when this module is imported
User.create_table()

# Create the table when this module is imported
ProcessedVideo.create_table()
