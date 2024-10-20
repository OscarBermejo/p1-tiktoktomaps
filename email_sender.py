import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys

# Add the directory above the current one to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Email configuration
SMTP_SERVER = "smtp.mail.eu-west-1.awsapps.com"  # Replace with your WorkMail region if different
SMTP_PORT = 465
SENDER_EMAIL = "info@tiktoktomaps.com"
SENDER_PASSWORD = os.environ.get("EMAIL_PASSWORD")  # Store this securely, preferably as an environment variable

def send_welcome_email(recipient_email):
    subject = "Welcome to TikTok To Maps!"
    body = """
    Welcome to TikTok To Maps!

    Thank you for signing up. We're excited to have you on board.

    With TikTok To Maps, you can easily extract location information from TikTok videos.

    If you have any questions, feel free to reply to this email.

    Best regards,
    The TikTok To Maps Team
    """

    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = recipient_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"Welcome email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Failed to send welcome email to {recipient_email}. Error: {str(e)}")


def send_password_reset_email(recipient_email, reset_token):
    subject = "Reset Your TikTok To Maps Password"
    body = f"""
    Hello,

    You have requested to reset your password for TikTok To Maps.

    Please click on the following link to reset your password:
    {Config.BASE_URL}/reset_password/{reset_token}

    If you did not request this, please ignore this email.

    Best regards,
    The TikTok To Maps Team
    """

    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = recipient_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"Password reset email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Failed to send password reset email to {recipient_email}. Error: {str(e)}")


if __name__ == "__main__":
    
    recipient_email = 'obermejocorrea@gmail.com'
    send_welcome_email(recipient_email)
    print(f"Attempted to send welcome email to {recipient_email}")
