

1. You need to have Python 3.7+ and pip installed.
2. Clone the repo and navigate to the project directory.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Get your OpenAI API key and set it as an environment variable.
5. Run `streamlit run main.py` in the terminal to launch the application in your web browser.

```bash
streamlit run main.py

```
## App Walkthrough

### Setup Section
- **API Key**: Enter your OpenAI API key if it's not set as an environment variable.
- **Model Selection**: Choose a language model from `gpt-4`, `gpt-3.5-turbo`, or `gpt-3.5-turbo-16k`.
- **Temperature**: Set the creativity level for text generation, ranging from `0.0` (deterministic) to `1.0` (highly random).

### Data Summarization
- **Dataset Selection**: Choose from preloaded datasets or upload your own CSV or JSON file.
- **Summarization Method**:
  - `llm`: Summarizes using the language model with additional details.
  - `default`: Generates a summary based on dataset column statistics.
  - `columns`: Uses only the column names for summarization.

### Goal Selection
- **Generate Goals**: Select the number of goals or add a custom goal for visualization.
- **Select Goal**: Choose a specific goal for generating visualizations.

### Visualization Library
- **Choose Visualization Library**: Select a visualization library (`seaborn`, `matplotlib`, or `plotly`).
- **Number of Visualizations**: Set the number of visualizations to generate based on the chosen goal.

### Visualizations
- **Select and View Visualizations**: Choose a generated visualization to display or edit.
- **Visualization Code**: View the underlying Python code for the visualization.
- **NLP Customization**: Update visualization properties (e.g., color, size) with natural language instructions.

### Customization Options
- **Goal Generation**: Add custom goals to focus on specific insights.
- **Visualization Modifications**: Use NLP commands to adjust visual elements, such as chart type or color.
- **Add New Datasets**: Upload additional datasets to expand dashboard options and customization.

