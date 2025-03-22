import sys
import smtplib
import json
import google.generativeai as genai
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

genai.configure(api_key="AIzaSyAQvW-7i3jnNu5qwolDOPV9q2HhdkKtrAU") 

SENDER_EMAIL = "scorematrixx@gmail.com"
SENDER_PASSWORD = "rigu skzn dqxm yiqn"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

if len(sys.argv) < 3:
    print("Error: Missing required arguments (recipient_email, subject)")
    sys.exit(1)

recipient_email = sys.argv[1]
email_subject = sys.argv[2]

students_json_path = "../frontend/data/students.json"  
try:
    with open(students_json_path, "r", encoding="utf-8") as f:
        students_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{students_json_path}' was not found.")
    sys.exit(1)

student = next((s for s in students_data if s["email"] == recipient_email), None)
if not student:
    print(f"Error: No student found with email '{recipient_email}'.")
    sys.exit(1)

def generate_study_tips(weak_topics):
    prompt = f"""The student has the following weak topics:
{weak_topics}

Provide a list of topics the student should study and tips for improvement. Be concise and actionable, the response should be simple text with little to no formatting.,if you have no tips to give/no weak topics given or in any case that data is not present then send a No tips this time, but hey—take a deep breath, grab some water, and relax. You did great!(pls ensure that it is unformatted ie it contains only text)"""
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return response.text.strip() if response.text else "No tips this time, but hey—take a deep breath, grab some water, and relax. You did great!"

weak_topics = student["weakTopics"]
study_tips = generate_study_tips(weak_topics)

email_content = f"""Hello {student["name"]},

This is an automated email from ScoreMatrix. Based on your recent performance, here are the topics you should focus on and some tips for improvement:

{study_tips}

Please find the attached document for further reference.

Best regards,
Team ScoreMatrix
"""

msg = MIMEMultipart()
msg["From"] = SENDER_EMAIL
msg["To"] = recipient_email
msg["Subject"] = email_subject

msg.attach(MIMEText(email_content, "plain"))

pdf_filename = "../frontend/outputs/answers.docx"
try:
    with open(pdf_filename, "rb") as attachment:
        pdf_part = MIMEBase("application", "octet-stream")
        pdf_part.set_payload(attachment.read())
        encoders.encode_base64(pdf_part)
        pdf_part.add_header("Content-Disposition", f"attachment; filename=Answers.docx")
        msg.attach(pdf_part)
except FileNotFoundError:
    print(f"Error: The file '{pdf_filename}' was not found.")
    sys.exit(1)

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
    server.quit()
    print("Email with attachment sent successfully!")
    sys.exit(0)
except Exception as e:
    print(f"Error sending email: {e}")
    sys.exit(1)