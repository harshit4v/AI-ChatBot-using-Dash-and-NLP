import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import random

# Load the dataset
data = pd.read_csv('chatbot_dataset.csv')

# Preprocess the data
nltk.download('punkt_tab')
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    answer = model.predict([question])[0]
    return answer

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout with dark gradient styling and transparency
app.layout = html.Div([
    html.H1("🤖 AI Chatbot", 
            style={
                'textAlign': 'center',
                'background': 'linear-gradient(135deg, rgba(30, 30, 60, 0.9) 0%, rgba(60, 30, 90, 0.9) 50%, rgba(90, 30, 60, 0.9) 100%)',
                'color': '#ffffff',
                'padding': '25px',
                'margin': '0 0 20px 0',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '2.2em',
                'fontWeight': '300',
                'textShadow': '0 0 15px rgba(255,255,255,0.3)',
                'borderRadius': '15px',
                'boxShadow': '0 6px 20px rgba(0,0,0,0.3)'
            }),
    
    html.Div([
        dcc.Textarea(
            id='user-input',
            placeholder='Type your question here...',
            value='',
            style={
                'width': '100%', 
                'height': '100px',
                'padding': '15px',
                'fontSize': '14px',
                'border': '2px solid rgba(100,100,150,0.3)',
                'borderRadius': '12px',
                'background': 'linear-gradient(135deg, rgba(50, 50, 100, 0.1) 0%, rgba(80, 50, 120, 0.1) 100%)',
                'color': '#333',
                'fontFamily': 'Arial, sans-serif',
                'resize': 'none',
                'outline': 'none',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'transition': 'all 0.3s ease'
            }
        ),
        
        html.Button('Submit ✨', 
                   id='submit-button', 
                   n_clicks=0,
                   style={
                       'width': '100%',
                       'padding': '12px',
                       'marginTop': '15px',
                       'fontSize': '16px',
                       'fontWeight': '600',
                       'border': 'none',
                       'borderRadius': '12px',
                       'background': 'linear-gradient(135deg, rgba(100, 50, 150, 0.8) 0%, rgba(150, 50, 200, 0.8) 100%)',
                       'color': '#ffffff',
                       'cursor': 'pointer',
                       'fontFamily': 'Arial, sans-serif',
                       'boxShadow': '0 4px 15px rgba(100, 50, 150, 0.3)',
                       'transition': 'all 0.3s ease'
                   }),
        
        html.Div(id='chatbot-output', 
                style={
                    'padding': '20px',
                    'marginTop': '20px',
                    'minHeight': '80px',
                    'borderRadius': '12px',
                    'background': 'linear-gradient(135deg, rgba(240, 240, 250, 0.8) 0%, rgba(230, 240, 255, 0.8) 100%)',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                    'fontFamily': 'Arial, sans-serif',
                    'border': '1px solid rgba(200,200,220,0.3)'
                })
    ], style={
        'maxWidth': '600px',
        'margin': '0 auto',
        'padding': '25px',
        'background': 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(250, 250, 255, 0.9) 100%)',
        'borderRadius': '15px',
        'boxShadow': '0 8px 25px rgba(0,0,0,0.15)',
        'border': '1px solid rgba(200,200,220,0.3)'
    })
], style={
    'minHeight': '100vh',
    'background': '#ffffff',
    'fontFamily': 'Arial, sans-serif',
    'padding': '20px'
})

# Remove floating particles to keep it simple

# Define callback to update chatbot response
@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    if n_clicks > 0 and user_input.strip():
        response = get_response(user_input)
        return html.Div([
            html.P(f"👤 You: {user_input}", 
                  style={
                      'margin': '15px 0',
                      'padding': '10px 15px',
                      'background': 'linear-gradient(135deg, rgba(80, 60, 140, 0.8) 0%, rgba(120, 60, 180, 0.8) 100%)',
                      'color': '#ffffff',
                      'borderRadius': '15px 15px 5px 15px',
                      'fontSize': '14px',
                      'boxShadow': '0 3px 10px rgba(0,0,0,0.2)',
                      'maxWidth': '80%',
                      'marginLeft': 'auto',
                      'textAlign': 'right'
                  }),
            html.P(f"🤖 Bot: {response}", 
                  style={
                      'margin': '15px 0',
                      'padding': '10px 15px',
                      'background': 'linear-gradient(135deg, rgba(60, 100, 140, 0.8) 0%, rgba(60, 140, 180, 0.8) 100%)',
                      'color': '#ffffff',
                      'borderRadius': '15px 15px 15px 5px',
                      'fontSize': '14px',
                      'boxShadow': '0 3px 10px rgba(0,0,0,0.2)',
                      'maxWidth': '80%',
                      'marginRight': 'auto'
                  })
        ])
    return html.Div("✨ Ask me something! ✨", 
                   style={
                       'textAlign': 'center',
                       'fontSize': '16px',
                       'color': '#666',
                       'fontStyle': 'italic',
                       'padding': '30px'
                   })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)