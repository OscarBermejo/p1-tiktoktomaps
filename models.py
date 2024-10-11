import mysql.connector
from mysql.connector import Error, pooling, IntegrityError
import json
from datetime import datetime
import logger_config
from flask_login import UserMixin
import sys
import os

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
    def __init__(self, url, task_id, video_id=None, video_duration=None, processing_time=None, results=None):
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
    def add(url, task_id, video_id=None, video_duration=None, processing_time=None, results=None):
        with Database() as db:
            query = """
            INSERT INTO processed_videos (url, task_id, video_id, video_duration, processing_time, results)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            task_id = VALUES(task_id),
            video_id = VALUES(video_id),
            video_duration = VALUES(video_duration),
            processing_time = VALUES(processing_time),
            results = VALUES(results)
            """
            params = (url, task_id, video_id, video_duration, processing_time, json.dumps(results) if results else None)
            db.execute_query(query, params)

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
                    return ProcessedVideo(url, task_id, video_id, video_duration, processing_time, json.loads(results) if results else None)
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
                    return ProcessedVideo(url, task_id, video_id, video_duration, processing_time, json.loads(results) if results else None)
                logger.info("No processed video found for the given task ID")
                return None
            except Error as e:
                logger.error(f"Error retrieving processed video: {e}")
                raise

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# ... (keep your existing imports and Database class)

class User(UserMixin):
    def __init__(self, email, password=None):
        self.email = email
        self.password_hash = generate_password_hash(password) if password else None
        self.created_at = datetime.now()
        self.last_login = None

    def get_id(self):
        return self.email

    @staticmethod
    def create_table():
        logger.info("Attempting to create users table")
        with Database() as db:
            query = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL
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
                return User(email, password)
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
            query = "SELECT * FROM users WHERE email = %s"
            try:
                cursor = db.execute_query(query, (email,))
                result = cursor.fetchone()
                if result:
                    id, email, password_hash, created_at, last_login = result
                    user = User(email)
                    user.password_hash = password_hash
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
    def update_last_login(username):
        logger.info(f"Updating last login for user: {username}")
        with Database() as db:
            query = "UPDATE users SET last_login = %s WHERE username = %s"
            params = (datetime.now(), username)
            try:
                db.execute_query(query, params)
                logger.info("Last login updated successfully")
            except Error as e:
                logger.error(f"Error updating last login: {e}")
                raise

# Create the table when this module is imported
User.create_table()

# Create the table when this module is imported
ProcessedVideo.create_table()