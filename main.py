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

st.write("# PBL: Adaptive Dashboards Powered by Generative LLMs ")

import streamlit as st
from PIL import Image
import os



# Load your images (replace these paths with your own images)
img1 = Image.open("/Users/numaan/lida-streamlit/images/n.png")
img2 = Image.open("/Users/numaan/lida-streamlit/images/r.png")
img3 = Image.open("/Users/numaan/lida-streamlit/images/s.png")
img4 = Image.open("/Users/numaan/lida-streamlit/images/d.png")  # Add the path for Dr. Deepak Dharrao's image

# Display the guide's image on the second row
st.markdown("***Guide : Dr. Deepak Dharrao***")
st.image(img4, width=200, caption="Dr. Deepak Dharrao")  
st.markdown("***By: Nouman Jinabade (R&A), Shivansh Chutani (E&TC), Rika Mallika (E&TC)***")
# Create 3 columns for the first row of images
col1, col2, col3 = st.columns(3)

# Display the first row of images with captions
with col1:
    st.image(img1, use_column_width=True, caption="Nouman Jinabade")

with col2:
    st.image(img2, use_column_width=True, caption="Rika Mallika")

with col3:
    st.image(img3, use_column_width=True, caption="Shivansh Chutani")

# Display the guide's image on a new line


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

# Step 1 - Get OpenAI API key
if openai_key:
    # Initialize selected_dataset to None
    selected_dataset = None

    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    st.sidebar.write("## Text Generation Model")
    models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    selected_model = st.sidebar.selectbox(
        'Choose a model',
        options=models,
        index=0
    )

    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0)

    # set use_cache in sidebar
    use_cache = st.sidebar.checkbox("Use cache", value=True)

    # Handle dataset selection and upload
    st.sidebar.write("## Data Summarization")
    st.sidebar.write("### Choose a dataset")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    ]

    selected_dataset_label = st.sidebar.selectbox(
        'Choose a dataset',
        options=[dataset["label"] for dataset in datasets],
        index=0
    )

    upload_own_data = st.sidebar.checkbox("Upload your own data")

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

        if uploaded_file is not None:
            # Get the original file name and extension
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Load the data depending on the file type
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            # Save the data using the original file name in the data dir
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})

            # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    else:
        selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]

    if not selected_dataset:
        st.info("To continue, select a dataset from the sidebar on the left or upload your own.")

    st.sidebar.write("### Choose a summarization method")
    # summarization_methods = ["default", "llm", "columns"]
    summarization_methods = [
        {"label": "llm",
         "description":
         "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
        {"label": "default",
         "description": "Uses dataset column statistics and column names as the summary"},

        {"label": "columns", "description": "Uses the dataset column names as the summary"}]

    # selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)
    selected_method_label = st.sidebar.selectbox(
        'Choose a method',
        options=[method["label"] for method in summarization_methods],
        index=0
    )

    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)

# Step 3 - Generate data summary
if openai_key and selected_dataset and selected_method:
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)

    st.write("## Summary")
    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    # Step 4 - Generate goals
    if summary:
        st.sidebar.write("### Goal Selection")

        num_goals = st.sidebar.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.sidebar.checkbox("Add Your Own Goal")

        # **** lida.goals *****
        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
        st.write(f"## Goals ({len(goals)})")

        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input("Describe Your Goal")

            if user_goal:

                new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)

        # st.markdown("### Selected Goal")
        selected_goal_index = goal_questions.index(selected_goal)
        st.write(goals[selected_goal_index])

        selected_goal_object = goals[selected_goal_index]

        # Step 5 - Generate visualizations
        if selected_goal_object:
            st.sidebar.write("## Visualization Library")
            visualization_libraries = ["seaborn", "matplotlib", "plotly"]

            selected_library = st.sidebar.selectbox(
                'Choose a visualization library',
                options=visualization_libraries,
                index=0
            )

            # Update the visualization generation call to use the selected library.
            st.write("## Visualizations")

            # slider for number of visualizations
            num_visualizations = st.sidebar.slider(
                "Number of visualizations to generate",
                min_value=1,
                max_value=10,
                value=2)

            textgen_config = TextGenerationConfig(
                n=num_visualizations, temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            # **** lida.visualize *****
            visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library)

            viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]

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
                

