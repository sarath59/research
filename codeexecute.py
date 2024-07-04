import os
import warnings
import atexit
from concurrent.futures import ThreadPoolExecutor
import traceback

warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Set API keys directly in the code
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = "d4e81a83028357dc5af70321c25760e1dd07e44f"

from crewai_tools import CodeDocsSearchTool

code_docs_tool = CodeDocsSearchTool()

# Define the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Code Executor Agent
code_executor = Agent(
    role="Code Executor",
    goal="Execute the given code snippets and provide feedback and results.",
    verbose=True,
    memory=True,
    backstory=(
        "You are an experienced software engineer specializing in executing and reviewing code. "
        "Your goal is to ensure the correctness, efficiency, and maintainability of the code. "
        "You use various tools to analyze and execute code, check for potential issues, and suggest improvements."
    ),
    allow_delegation=True,
    allow_code_execution=True,
    cache=True,
    llm=llm
)

# Code Execution Task
code_execution_task = Task(
    description=(
        "Execute the given code snippet(s) and provide detailed feedback. "
        "Ensure the code runs correctly and highlight any issues found. "
        "Provide suggestions for improvements if necessary."
    ),
    expected_output=(
        "The execution results of the given code snippet(s), including any errors encountered and suggestions for improvements."
    ),
    agent=code_executor,
    async_execution=False
)

# Define Crew
code_execution_crew = Crew(
    agents=[code_executor],
    tasks=[code_execution_task],
    process=Process.sequential,
)

executor = ThreadPoolExecutor()

def shutdown():
    executor.shutdown(wait=True)

atexit.register(shutdown)

def main():
    try:
        print("Enter the code snippet to review (enter 'END' on a new line to finish):")
        code_lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            code_lines.append(line)
        code_snippet = "\n".join(code_lines)
        result = code_execution_crew.kickoff(inputs={'code_snippet': code_snippet})
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}\n{traceback.format_exc()}")
    finally:
        shutdown()

if __name__ == "__main__":
    main()
