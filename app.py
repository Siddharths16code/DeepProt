
import streamlit as st
import pandas as pd

from inference import DeepProtPredictor


st.set_page_config(
    page_title="DeepProt | Protein Function Prediction",
    page_icon="🧬",
    layout="wide"
)


@st.cache_resource
def load_predictor():
    return DeepProtPredictor()


def parse_fasta(fasta_text):
    """Extract protein sequence from FASTA content."""
    lines = fasta_text.strip().splitlines()

    sequence_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(">")
    ]

    if not sequence_lines:
        raise ValueError("No protein sequence found in FASTA file.")

    return "".join(sequence_lines)


st.title("🧬 DeepProt")
st.subheader("Protein Function Prediction Using Deep Learning")

st.markdown(
    """
    Predict potential Gene Ontology (GO) annotations
    from protein amino acid sequences using a trained
    Multi-Kernel Convolutional Neural Network.
    """
)

st.info(
    "Supported alphabet: 20 standard amino acids "
    "(ACDEFGHIKLMNPQRSTVWY). Maximum processed length: 3000 residues."
)

st.subheader("Input Protein Sequence")

input_method = st.radio(
    "Choose input method:",
    ["Paste Sequence", "Upload FASTA File"],
    horizontal=True
)

sequence = ""

if input_method == "Paste Sequence":

    sequence = st.text_area(
        "Enter Protein Sequence",
        height=200,
        placeholder="Paste your amino acid sequence here..."
    )

else:

    uploaded_file = st.file_uploader(
        "Upload Protein FASTA File",
        type=["fasta", "fa", "faa", "txt"]
    )

    if uploaded_file is not None:
        fasta_content = uploaded_file.getvalue().decode("utf-8")

        try:
            sequence = parse_fasta(fasta_content)
            st.success("FASTA file loaded successfully!")
            st.caption(f"Extracted sequence length: {len(sequence)} residues")

        except Exception as error:
            st.error(f"FASTA parsing error: {error}")


predict_button = st.button(
    "🔬 Predict Protein Functions",
    type="primary",
    use_container_width=True
)


if predict_button:

    if not sequence.strip():
        st.error("Please provide a protein sequence or upload a FASTA file.")

    else:
        try:
            with st.spinner("Analyzing protein sequence..."):

                predictor = load_predictor()
                result = predictor.predict(
                    sequence,
                    top_k=226
                )

            st.success("Prediction completed successfully!")

            if result["truncated"]:
                st.warning(
                    "Sequence exceeds 3000 residues. "
                    "Only the first 3000 residues were processed."
                )

            col1, col2, col3 = st.columns(3)

            col1.metric("Sequence Length", result["sequence_length"])
            col2.metric("Processed Length", result["processed_length"])
            col3.metric("Predicted GO Terms", result["total_predicted_terms"])

            st.divider()

            st.subheader("Predicted GO Functions")

            all_scores = result["all_predictions"]

            predicted = all_scores[
                all_scores["Predicted"] == True
            ].copy()

            if not predicted.empty:

                display_df = predicted[
                    [
                        "GO_ID",
                        "GO_Name",
                        "GO_Namespace",
                        "Confidence_Percent"
                    ]
                ].rename(
                    columns={
                        "Confidence_Percent": "Model Score (%)"
                    }
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Predicted GO Terms (CSV)",
                    data=display_df.to_csv(index=False),
                    file_name="deepprot_predictions.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "No GO terms crossed the prediction threshold."
                )

            with st.expander("View all 226 GO-term scores"):

                st.dataframe(
                    all_scores[
                        [
                            "GO_ID",
                            "GO_Name",
                            "GO_Namespace",
                            "Confidence_Percent",
                            "Predicted"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download All GO Scores (CSV)",
                    data=all_scores.to_csv(index=False),
                    file_name="deepprot_all_go_scores.csv",
                    mime="text/csv"
                )

            st.caption(
                "Model scores are not calibrated probabilities. "
                "Predictions are research-oriented estimates, "
                "not experimentally confirmed annotations."
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")


st.divider()

st.caption(
    "DeepProt | Multi-Kernel CNN | 226 GO terms | "
    "Prediction threshold: 0.46"
)
