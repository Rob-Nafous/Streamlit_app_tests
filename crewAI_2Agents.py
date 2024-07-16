import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

# Initialize the language model for analysis
tllm = Ollama(
    model="tinyllama",
    base_url="http://localhost:11434"
)

# Initialize the language model for speaking
llm3 = Ollama(
    model="llama3",
    base_url="http://localhost:11434"
)


# Create the tone and emotion analyzer agent using tinyllama
tone_emotion_analyzer_agent = Agent(
    role="Tone and Emotion Analyzer",
    goal="Analyze the tone and emotions of the given text and provide a quick detailed report limited to 2 sentences",
    backstory="You are an expert in analyzing the tone and emotions of text, and can provide insights into the underlying sentiments.",
    allow_delegation=False,
    verbose=True,
    llm=tllm
)   

# Create the response generator agent using llama3
response_generator_agent = Agent(
    role="Response Generator",
    goal="Generate a response to the user's query based on the detected tone and emotions.",
    backstory="You are skilled in crafting responses that are empathetic and appropriate based on the tone and emotions of the input text.",
    allow_delegation=False,
    verbose=True,
    llm=llm3
)

# Streamlit app
st.title("Tone and Emotion Analyzer and Responder")

# Get the text input from the user
user_text = st.text_area("Veuillez entrer le texte à analyser et à répondre :")

if st.button("Analyser et Répondre"):
    if user_text.strip() == "":
        st.write("Veuillez entrer du texte à analyser et à répondre.")
    else:
        # Create the task for tone and emotion analysis
        analysis_task = Task(
            description=user_text,
            agent=tone_emotion_analyzer_agent,
            expected_output="A quick detailed report on the polarity, subjectivity, and emotions of the text. Focus the mains emotions, limited to 2 sentences. Be fast"
        )

        # Create the crew with the tone and emotion analyzer agent
        analysis_crew = Crew(
            agents=[tone_emotion_analyzer_agent],
            tasks=[analysis_task],
            verbose=1
        )

        # Function to execute the analysis task using the LLM
        def analyze_tone_and_emotions(crew, task):
            # Prompt to analyze tone
            tone_prompt = f"Analyze the tone of the following text:\n\n{task.description}\n\nProvide the polarity and subjectivity."
            tone_response = crew.agents[0].llm.invoke(tone_prompt)

            # Prompt to analyze emotions
            emotion_prompt = f"Analyze the emotions in the following text:\n\n{task.description}\n\nList the emotions present and their intensity."
            emotion_response = crew.agents[0].llm.invoke(emotion_prompt)

            return f"Tone Analysis:\n{tone_response}\n\nEmotion Analysis:\n{emotion_response}"

        # Analyze the tone and emotions
        analysis_result = analyze_tone_and_emotions(analysis_crew, analysis_task)

        # Create the task for generating a response based on the analysis
        response_task = Task(
            description=f"{user_text}\n\n{analysis_result}",
            agent=response_generator_agent,
            expected_output="A response to the user's query considering the detected tone and emotions."
        )

        # Create the crew with the response generator agent
        response_crew = Crew(
            agents=[response_generator_agent],
            tasks=[response_task],
            verbose=1
        )

        response_result = response_crew.kickoff()

        # Display the response only
        st.write(response_result)