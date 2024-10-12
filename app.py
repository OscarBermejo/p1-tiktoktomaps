from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
from celery.result import AsyncResult
from models import ProcessedVideo, User, UserProcessedVideo
from utils import validate_tiktok_url
import logger_config
from celery_app import celery_app  
from tasks import process_video 
import sys
import os

# Add the directory above the current one to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config


logger = logger_config.get_logger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
app.config.from_object(Config)
celery_app.conf.update(app.config)

logger.info("Flask app configuration:")
logger.info(f"CELERY_RESULT_BACKEND: {app.config['CELERY_RESULT_BACKEND']}")

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)  

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.get_by_email(email)
        logger.info(f'user: {user}')
        if user:
            if check_password_hash(user.password_hash, password):
                login_user(user)
                logger.info(f"User {email} logged in successfully")
                flash('Logged in successfully.')
                return redirect(url_for('index'))
            else:
                logger.info(f"Invalid password for user: {email}")
        else:
            logger.info(f"No user found with email: {email}")
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.get_by_email(email):
            flash('Email already registered')
        else:       
            new_user = User.add(email, password)
            login_user(new_user)
            flash('Registered successfully. You are now logged in.')
            return redirect(url_for('index'))
      
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    logger.info("Accessing index route")
    logger.info(f"User authenticated: {current_user.is_authenticated}")
    if current_user.is_authenticated:
        logger.info(f"Current user: {current_user.email}")
    if request.method == 'POST':
        tiktok_url = request.form['tiktok_url']
        logger.info(f'Extracted URL from web: {tiktok_url}')
        
        if not validate_tiktok_url(tiktok_url):
            logger.warning(f'Invalid TikTok URL: {tiktok_url}')
            return render_template('index.html', error='Invalid TikTok URL')

        logger.info('Checking if video has been processed before')
        try:
            processed_video = ProcessedVideo.get_by_url(url=tiktok_url)
            if processed_video and processed_video.results:
                logger.info(f'Video already processed. Task ID: {processed_video.task_id}')
                return redirect(url_for('result', task_id=processed_video.task_id))
        except Exception as e:
            logger.error(f'Error checking processed video: {str(e)}')
            return render_template('index.html', error='Database error')

        # If we reach here, the video hasn't been processed yet
        logger.info('Video not processed yet. Creating a new task in Celery')
        task = process_video.delay(tiktok_url, current_user.email)
        logger.info(f'Celery task created. Task ID: {task.id}')

        logger.info(f'Saving celery task info to database. Task ID: {task.id}')
        try:
            processed_video = ProcessedVideo.add(url=tiktok_url, task_id=task.id)
            UserProcessedVideo.add(current_user.id, processed_video.id)
            logger.info('Task info saved to database successfully')
        except Exception as e:
            logger.error(f'Error saving task info to database: {str(e)}')

        return redirect(url_for('waiting', task_id=task.id))
    
    logger.info(f'Render index template')
    return render_template('index.html')

@app.route('/waiting/<task_id>')
def waiting(task_id):
    return render_template('waiting.html', task_id=task_id)

@app.route('/result/<task_id>')
def result(task_id):
    logger.info(f"Accessing result route for task_id: {task_id}")
    try:
        processed_video = ProcessedVideo.get_by_task_id(task_id)
        if processed_video and processed_video.results:
            logger.info(f"Results found for task {task_id}")
            return render_template('result.html', 
                                   links=processed_video.results, 
                                   video_id=processed_video.video_id,
                                   video_duration=processed_video.video_duration,
                                   processing_time=processed_video.processing_time,
                                   error=None)
        else:
            logger.warning(f"No results found for task {task_id}")
            return render_template('result.html', links=None, error='No results found')
    except Exception as e:
        logger.error(f"Error in result route: {str(e)}")
        return render_template('result.html', links=None, error=f'Error: {str(e)}')

@app.route('/process', methods=['POST'])
def process():
    if not current_user.is_authenticated:
        logger.warning("Unauthenticated user tried to access /process")
        return render_template('unauthenticated.html'), 401

    data = request.get_json()
    logger.info(f"Received data: {data}")
    
    if not data:
        logger.warning("No JSON data received")
        return jsonify({'error': 'No data provided'}), 400

    tiktok_url = data.get('tiktok_url')
    logger.info(f'Extracted URL from web: {tiktok_url}')
    
    if not tiktok_url or not validate_tiktok_url(tiktok_url):
        logger.warning(f'Invalid TikTok URL: {tiktok_url}')
        return jsonify({'error': 'Invalid TikTok URL'}), 400

    logger.info('Checking if video has been processed before')
    processed_video = ProcessedVideo.get_by_url(url=tiktok_url)
    
    if processed_video:
        logger.info(f'Video already processed. Task ID: {processed_video.task_id}')
        if processed_video.results:
            # Add record to UserProcessedVideo
            UserProcessedVideo.add(current_user.id, processed_video.id)
            return jsonify({
                'status': 'completed',
                'result': processed_video.results,
                'task_id': processed_video.task_id
            })
        else:
            # If we have a task_id but no results, it might still be processing
            # Still add a record to UserProcessedVideo
            logger.info(f"Adding record to UserProcessedVideo for processed_video: {processed_video.id}")
            UserProcessedVideo.add(current_user.id, processed_video.id)
            return jsonify({'status': 'processing', 'task_id': processed_video.task_id})

    logger.info('Creating a new task in Celery')
    try:
        # Pass the user's email to the Celery task
        task = process_video.delay(tiktok_url, current_user.email)
        logger.info(f'Celery task created. Task ID: {task.id}')
    except Exception as e:
        logger.error(f'Error creating Celery task: {str(e)}')
        return jsonify({'error': 'Failed to create task'}), 500

    logger.info(f'Saving celery task info to database. Task ID: {task.id}')
    try:
        processed_video = ProcessedVideo.add(url=tiktok_url, task_id=task.id)
        logger.info(f"ProcessedVideo created with id: {processed_video.id}")
        # Add a record to UserProcessedVideo
        UserProcessedVideo.add(current_user.id, processed_video.id)
        logger.info('Task info saved to database successfully')
    except Exception as e:
        logger.error(f'Error saving task info to database: {str(e)}')
        return jsonify({'error': 'Database error'}), 500

    return jsonify({'task_id': task.id})

@app.route('/check_task/<task_id>')
def check_task(task_id):
    logger.info(f"Checking task status for task_id: {task_id}")
    task = AsyncResult(task_id, app=celery_app)
    logger.info(f"Task state: {task.state}, Task result: {task.result}")
        
    if task.ready():
        logger.info(f"Task {task_id} is completed")
        if task.successful():
            # Update the database with the results
            processed_video = ProcessedVideo.get_by_task_id(task_id)
            if processed_video:
                if task.result.get('results'):
                    ProcessedVideo.update_results(
                        processed_video.url, 
                        task.result.get('results'),
                        video_id=task.result.get('video_id'),
                        video_duration=task.result.get('video_duration'),
                        processing_time=task.result.get('processing_time')
                    )
                return jsonify({'status': 'completed', 'result': task.result})
            else:
                logger.warning(f"No processed video found for task_id: {task_id}")
                return jsonify({'status': 'completed', 'result': None})
        else:
            # Task failed
            error_msg = str(task.result) if task.result else "Unknown error occurred"
            logger.error(f"Task {task_id} failed: {error_msg}")
            return jsonify({'status': 'failed', 'error': error_msg}), 500
    else:
        logger.info(f"Task {task_id} is still processing, state: {task.state}")
        return jsonify({'status': 'processing', 'state': task.state})


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=8080, debug=True)
