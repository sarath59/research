import os
from flask import Flask, request, jsonify, send_from_directory
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

# Initialize Flask app
app = Flask(__name__)

# Set up API keys
os.environ["SERPER_API_KEY"] = "d4e81a83028357dc5af70321c25760e1dd07e44f"
os.environ["OPENAI_API_KEY"] = "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"

# Instantiate tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
web_search_tool = WebsiteSearchTool()

# Create Immigration Information Specialist agent
immigration_agent = Agent(
    role='Immigration Information Specialist',
    goal='Provide detailed and up-to-date answers to immigration-related queries',
    backstory=(
        "An expert in immigration policies with extensive knowledge of the latest regulations and procedures."
        " Ensure to include a disclaimer advising users to consult an immigration attorney for final decisions."
    ),
    tools=[search_tool, scrape_tool, web_search_tool],
    verbose=True
)

# Define research task
research_task = Task(
    description='Research the latest immigration policies and procedures from specified websites.',
    expected_output='A summary of the latest immigration policies and procedures.',
    agent=immigration_agent
)

# Function to handle user questions
def handle_user_questions(question):
    # Define query task
    query_task = Task(
        description=f'Answer the following immigration-related question: "{question}". Ensure to include a disclaimer advising users to consult an immigration attorney.',
        expected_output=f'Detailed answer to the question: "{question}" with a disclaimer.',
        agent=immigration_agent
    )
    
    # Assemble a crew
    immigration_crew = Crew(
        agents=[immigration_agent],
        tasks=[research_task, query_task],
        verbose=2
    )
    
    # Execute tasks
    result = immigration_crew.kickoff()
    return result

# Serve the frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'immigration.html')

# Define Flask routes
@app.route('/api/v1/immigration-question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        response = handle_user_questions(question)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)