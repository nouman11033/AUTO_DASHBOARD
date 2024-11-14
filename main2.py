import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="PBL: Adaptive Dashboards Powered by Generative LLMs ",
    page_icon="🥰"
)

st.write("# PBL: Adaptive Dashboards Powered by Generative LLMs")
st.markdown("***By: Nouman Jinabade (R&A), Shivansh Chutani (E&TC), Rika Mallika (E&TC)***")

# Sidebar setup
st.sidebar.write("## Setup")
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
    if openai_key:
        display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
        st.sidebar.write(f"Current key: {display_key}")
    else:
        st.sidebar.write("Please enter OpenAI API key.")
else:
    display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
    st.sidebar.write(f"OpenAI API key loaded from environment variable: {display_key}")

# Initialize summary and goals variables to None
summary = None
goals = None  # Initialize goals here

# Step 2 - Dataset and summarization method
if openai_key:
    # Select model and temperature
    st.sidebar.write("## Text Generation Model")
    models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    selected_model = st.sidebar.selectbox('Choose a model', options=models, index=0)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)
    use_cache = st.sidebar.checkbox("Use cache", value=True)

    # Handle dataset selection and upload
    st.sidebar.write("## Data Summarization")
    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    ]
    selected_dataset_label = st.sidebar.selectbox('Choose a dataset', options=[dataset["label"] for dataset in datasets], index=0)
    upload_own_data = st.sidebar.checkbox("Upload your own data")

    selected_dataset = None  # Initialize selected_dataset to None by default

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])
        if uploaded_file is not None:
            file_name, file_extension = os.path.splitext(uploaded_file.name)
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)
            selected_dataset = uploaded_file_path  # Set selected_dataset after file is uploaded
            datasets.append({"label": file_name, "url": uploaded_file_path})
    else:
        selected_dataset = datasets[[dataset["label"] for dataset in datasets].index(selected_dataset_label)]["url"]

    st.sidebar.write("### Choose a summarization method")
    summarization_methods = [
        {"label": "llm", "description": "LLM-based detailed summary"},
        {"label": "default", "description": "Column statistics summary"},
        {"label": "columns", "description": "Only column names summary"}
    ]
    selected_method_label = st.sidebar.selectbox('Choose a method', options=[method["label"] for method in summarization_methods], index=0)
    selected_method = summarization_methods[[method["label"] for method in summarization_methods].index(selected_method_label)]["label"]
    selected_summary_method_description = summarization_methods[[method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(f"<span> {selected_summary_method_description} </span>", unsafe_allow_html=True)

# Step 3 - Data summarization
if openai_key and selected_dataset and selected_method:  # Ensure selected_dataset is set
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(n=1, temperature=temperature, model=selected_model, use_cache=use_cache)

    st.write("## Summary")
    
    # Perform summarization
    try:
        summary = lida.summarize(selected_dataset, summary_method=selected_method, textgen_config=textgen_config)
        if "dataset_description" in summary:
            st.write(summary["dataset_description"])
        if "fields" in summary:
            fields = summary["fields"]
            nfields = []
            for field in fields:
                flatted_fields = {"column": field["column"]}
                for row in field["properties"].keys():
                    flatted_fields[row] = str(field["properties"][row]) if row == "samples" else field["properties"][row]
                nfields.append(flatted_fields)
            nfields_df = pd.DataFrame(nfields)
            st.write(nfields_df)
        else:
            st.write(str(summary))
    except Exception as e:
        st.error(f"Error during summarization: {e}")

# Step 4 - Goal generation with additional dataset
if summary:
    st.sidebar.write("### Goal Selection")
    num_goals = st.sidebar.slider("Number of goals to generate", min_value=1, max_value=10, value=4)
    own_goal = st.sidebar.checkbox("Add Your Own Goal")

    if own_goal:
        user_goal = st.sidebar.text_input("Describe Your Goal")
        add_another_dataset = st.sidebar.checkbox("Add another dataset for the goal")

        if add_another_dataset:
            second_dataset_upload = st.sidebar.file_uploader("Upload another dataset (CSV or JSON)", type=["csv", "json"])
            if second_dataset_upload:
                file_name, file_extension = os.path.splitext(second_dataset_upload.name)
                if file_extension.lower() == ".csv":
                    second_data = pd.read_csv(second_dataset_upload)
                elif file_extension.lower() == ".json":
                    second_data = pd.read_json(second_dataset_upload)
                second_dataset_path = os.path.join("data", second_dataset_upload.name)
                second_data.to_csv(second_dataset_path, index=False)
                
                # Summarize second dataset
                second_summary = lida.summarize(second_dataset_path, summary_method=selected_method, textgen_config=textgen_config)
                summary = {**summary, **second_summary}  # Combine summaries of both datasets

        if user_goal:
            new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
            goals = [new_goal]  # Set the goals here
        else:
            # Generate goals using the combined summary of both datasets
            goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)  # Set the goals here
        
        # Ensure that goals were generated
        if goals:
            st.write(f"## Goals ({len(goals)})")
            goal_questions = [goal.question for goal in goals if goal.question]  # Ensure non-None questions
            if goal_questions:  # Check if goal_questions is non-empty
                selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)
                st.write(goals[goal_questions.index(selected_goal)])
            else:
                st.error("No valid goal questions generated.")
        else:
            st.error("No goals were generated.")

# Step 5 - Visualization
if goals:
    st.sidebar.write("## Visualization Library")
    visualization_libraries = ["seaborn", "matplotlib", "plotly"]

    selected_library = st.sidebar.selectbox(
        'Choose a visualization library',
        options=visualization_libraries,
        index=0
    )

    # Select the goal to visualize
    st.write("## Visualizations")
    selected_visual_goal = st.sidebar.selectbox('Choose a goal for visualization', options=goal_questions, index=0)
    selected_goal_object = goals[goal_questions.index(selected_visual_goal)]

    # Slider for number of visualizations
    num_visualizations = st.sidebar.slider(
        "Number of visualizations to generate",
        min_value=1,
        max_value=10,
        value=2)

    textgen_config = TextGenerationConfig(
        n=num_visualizations, temperature=temperature,
        model=selected_model,
        use_cache=use_cache)

    # Use lida.visualize to generate visualizations
    try:
        visualizations = lida.visualize(
            summary=summary,
            goal=selected_goal_object,
            textgen_config=textgen_config,
            library=selected_library
        )

        # Check if visualizations were generated successfully
        if visualizations:
            # Ensure we only include non-None visualizations
            viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations)) if visualizations[i] is not None]
            visualizations = [viz for viz in visualizations if viz is not None]

            if viz_titles:  # Check if viz_titles is non-empty
                selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)
                selected_viz = visualizations[viz_titles.index(selected_viz_title)]

                if selected_viz.raster:
                    from PIL import Image
                    import io
                    import base64

                    imgdata = base64.b64decode(selected_viz.raster)
                    img = Image.open(io.BytesIO(imgdata))
                    st.image(img, caption=selected_viz_title, use_column_width=True)

                st.write("### Visualization Code")
                st.code(selected_viz.code)

                # Get explanations for the visualization
                explanations = lida.explain(code=selected_viz.code, library=selected_library, textgen_config=textgen_config)

                st.write("### Explanations")
                for row in explanations[0]:
                    st.write(f"{row['section']} ** {row['explanation']}")

                st.write("## Customize Visualization using NLP")
                instructions = st.text_area(
                    "Enter customization instructions (e.g., 'Change color to red')", 
                    "make the chart height and width equal"
                )

                if st.button("Apply Customization"):
                    edited_charts = lida.edit(
                        code=selected_viz.code, summary=summary, instructions=[instructions], 
                        library=selected_library, textgen_config=textgen_config
                    )
                    st.code(edited_charts[0].code)

                    if edited_charts[0].raster:
                        img = Image.open(io.BytesIO(base64.b64decode(edited_charts[0].raster)))
                        st.image(img, caption="Edited Visualization", use_column_width=True)
            else:
                st.error("No valid visualizations generated.")
        else:
            st.error("No visualizations were generated.")

    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
