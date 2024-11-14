

1. You need to have Python 3.7+ and pip installed.
2. Clone the repo and navigate to the project directory.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Get your OpenAI API key and set it as an environment variable.
5. Run `streamlit run main.py` in the terminal to launch the application in your web browser.

```bash
streamlit run main.py

```
App Walkthrough
Setup Section:
API Key: Input your OpenAI API key here if not set as an environment variable.
Model Selection: Choose from available models (gpt-4, gpt-3.5-turbo, or gpt-3.5-turbo-16k).
Temperature: Set the randomness level for text generation, with values between 0.0 (deterministic) to 1.0 (highly random).
Data Summarization:
Dataset Selection: Choose a preloaded dataset, or upload your own CSV or JSON file.
Summarization Method: Select from options:
llm: Generates summaries using the LLM.
default: Provides summaries based on dataset column statistics.
columns: Summarizes by displaying column names only.
Goal Selection:
Generate Goals: Choose the number of goals to generate, or add a custom goal.
Select Goal: Choose a specific goal for visualization.
Visualization Library:
Choose Visualization Library: Select from available libraries (seaborn, matplotlib, plotly).
Number of Visualizations: Select the number of visualizations to generate for the chosen goal.
Visualizations:
Select and View Visualizations: Choose a generated visualization to view or modify.
Visualization Code: View the Python code for the visualization.
NLP Customization: Modify the visualization by inputting customization instructions in natural language.
Customization Options
Goal Generation: Manually add goals to suit specific insights or analysis.
Visualization Modifications: Use NLP instructions to update aspects of visualizations such as colors, chart type, and size.
Add New Datasets: Upload additional datasets to enhance dashboard customization capabilities.
