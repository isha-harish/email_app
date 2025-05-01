import os
from dotenv import load_dotenv
import base64
import json
from collections import defaultdict
from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Gmail API scopes and Mistral API key
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
load_dotenv() 
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 

user_interactions = defaultdict(int)

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def fetch_emails(service):
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=30).execute()
    messages = results.get('messages', [])
    email_data = []

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])

        from_email = subject = body = "N/A"
        for header in headers:
            if header['name'] == 'From':
                from_email = header['value']
            elif header['name'] == 'Subject':
                subject = header['value']

        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    break
        elif 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')

        email_data.append({
            'from': from_email,
            'subject': subject,
            'body': body,
            'timestamp': datetime.now().isoformat()
        })

        user_interactions[from_email] += 1

    return email_data

# --------------------- LLM Setup ----------------

if MISTRAL_API_KEY:
    print("Mistral API key loaded successfully.")
else:
    print("Warning: Mistral API key is not set!")

# HuggingFace LLM for summary + category
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    device=-1,
    max_new_tokens=40,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=generator)

summary_template = PromptTemplate(
    input_variables=["body"],
    template="Summarize this email in 20 words or fewer: {body}"
)

categorization_template = PromptTemplate(
    input_variables=["body"],
    template="Classify this email as Lead, Complaint, Inquiry, Churn Risk, or General. Reply with only the category. Email: {body}"
)

# ----------------- Mistral Structured Sentiment Classification -------------

class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(..., enum=["spanish", "english", "french", "german", "italian"])

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

structured_llm = init_chat_model(
    "mistral-large-latest",
    model_provider="mistralai"
).with_structured_output(Classification)

# -----------------------------------------------------------------------------

def sanitize_text(text, max_length=1500):
    return text.replace("\n", " ").strip()[:max_length]

def fetch_and_process_emails(service):
    emails = fetch_emails(service)
    processed_emails = []

    for email in emails:
        frequency = user_interactions[email['from']]
        body = sanitize_text(email['body'])

        try:
            summary = llm.invoke(summary_template.invoke({"body": body})).strip()
            category = llm.invoke(categorization_template.invoke({"body": body})).strip()

            # Sentiment classification using Mistral
            prompt = tagging_prompt.invoke({"input": body})
            classification = structured_llm.invoke(prompt)

            email.update({
                'summary': summary or "N/A",
                'category': category or "N/A",
                'interaction_count': frequency,
                'sentiment': classification.sentiment,
                'aggressiveness': classification.aggressiveness,
                'language': classification.language
            })

            processed_emails.append(email)
        except Exception as e:
            print(f"Error processing email from {email['from']}: {e}")
            continue

    return processed_emails
