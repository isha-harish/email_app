from flask import Flask, render_template
from gmail_fetcher import authenticate_gmail, fetch_and_process_emails

app = Flask(__name__)

@app.route('/')
def index():
    try:
        service = authenticate_gmail()
        emails = fetch_and_process_emails(service)
        return render_template('index.html', emails=emails)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
