import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

import plotly.express as px
import re

# ========== 0. Global model/tokenizer initialization (loaded only once) ==========
st.set_page_config(page_title="TWOSA", layout="centered")
@st.cache_resource
def load_model_and_tokenizer(model_name="DICTA-il/BEREL_2.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# ========== 1. Data loading functions ==========

@st.cache_data
def load_data():
    Yerushalmi_path = "./Yerushalmi.csv"  # columns: [Tractate, Page, Text]
    Bavli_path = "./Bavli_talmud_texts.csv"  # columns: [Talmud, Tractate, Page, Text]
    yeru_df = pd.read_csv(Yerushalmi_path, encoding='utf-8')
    bavli_df = pd.read_csv(Bavli_path, encoding='utf-8')

    # Collect the list of distinct tractates
    yeru_chapters = sorted(yeru_df["Tractate"].dropna().unique())
    bavli_chapters = sorted(bavli_df["Tractate"].dropna().unique())
    return yeru_df, bavli_df, yeru_chapters, bavli_chapters

# ========== 2. Text cleaning & extracting windowed snippets ==========

def extract_halakha_sentences(text, word, window):
    """
    Find all snippets in the given text that contain the target word,
    within a specified window size on both sides.
    Returns a list of raw text snippets, without any citation info.
    """
    # Convert to string to avoid errors on None or non-str input
    text = str(text)
    tokens = re.findall(r"\S+", text)
    halakha_indices = [i for i, t in enumerate(tokens) if t == word]
    matches = []
    for idx in halakha_indices:
        start_idx = max(0, idx - window)
        end_idx = min(len(tokens), idx + window + 1)
        window_text = " ".join(tokens[start_idx:end_idx])
        if window_text not in matches:
            matches.append(window_text)
    return matches

@st.cache_data
def get_word_embedding(text):
    """
    Given a piece of text, return its mean-pooled embedding vector of shape [1,768].
    """
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens).last_hidden_state  # [1, seq_len, 768]
    return output.mean(dim=1)  # [1, 768]

# ========== 3. Based on user selection, retrieve "Set A" / "Set B" texts & labels ==========

def filter_chapters(df, tractate_col, selected):
    """
    If 'All' is selected, return the entire DataFrame;
    otherwise, keep only the rows whose tractate appears in 'selected'.
    """
    if "All" in selected:
        return df
    else:
        return df[df[tractate_col].isin(selected)]

def build_set(df, target_word, window_size, label_for_source):
    """
    From the given DataFrame (which includes columns: Tractate, Page, and Text),
    extract the relevant snippets and compute embeddings.

    We keep two types of text:
      - clean_texts: used for embedding computation (no citation info)
      - display_texts: used for user display in the interface (appends "(Tractate Page)")
    """
    clean_texts = []
    display_texts = []  # For display in the final output

    for i, row in df.iterrows():
        tractate = str(row.get("Tractate", "Unknown"))
        page = str(row.get("Page", "X"))
        text_col = row.get("Text", "")

        # Extract all windowed snippets in this row's text
        snippets = extract_halakha_sentences(text_col, target_word, window_size)

        for snippet in snippets:
            # 1) Add to clean_texts (used for embeddings)
            clean_texts.append(snippet.strip())

            # 2) Add to display_texts (with reference appended, for showing in UI)
            snippet_with_ref = snippet.strip() + f" ({tractate} {page})"
            display_texts.append(snippet_with_ref)

    # Compute embeddings for clean_texts only
    embeddings_list = []
    for txt in clean_texts:
        emb = get_word_embedding(txt).squeeze(0)  # shape=[768]
        embeddings_list.append(emb)
    if len(embeddings_list) == 0:
        return [], None, [], []

    embeddings_tensor = torch.stack(embeddings_list)  # shape=[N,768]
    # Create a source label list of the same length as clean_texts
    source_list = [label_for_source]*len(clean_texts)

    # Return the texts, embeddings, source labels, and text for display
    return clean_texts, embeddings_tensor, source_list, display_texts

# ========== 4. Combine "Set A" + "Set B" for similarity / PCA / clustering ==========

def compare_two_sets(
    textsA, embA, sourceA, dispA,
    textsB, embB, sourceB, dispB,
    k_clusters, hier_clusters
):
    """
    Given two sets (A and B), each having:
      - textsX: raw text snippets for embedding
      - embX: corresponding vectors
      - sourceX: labels denoting which set each snippet belongs to
      - dispX: display text with references

    Compute pairwise cosine similarity (A vs B), run PCA dimensionality reduction,
    run K-Means and hierarchical clustering, then assemble a DataFrame of results.

    In the final DataFrame, the column "text" contains the display text.
    """
    # Pairwise similarity (A vs B)
    similarities = []
    if embA is not None and embB is not None:
        for vA in embA:
            for vB in embB:
                sim = F.cosine_similarity(vA, vB, dim=0)
                similarities.append(sim.item())

    avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0

    # Combine embeddings
    A_np = embA.numpy() if embA is not None else np.array([])
    B_np = embB.numpy() if embB is not None else np.array([])

    if A_np.size == 0 and B_np.size == 0:
        return None, 0.0, []

    if A_np.size == 0:
        # Only B data is present
        all_embeddings = B_np
        all_texts = textsB
        all_disp = dispB
        all_source = sourceB
    elif B_np.size == 0:
        # Only A data is present
        all_embeddings = A_np
        all_texts = textsA
        all_disp = dispA
        all_source = sourceA
    else:
        all_embeddings = np.vstack((A_np, B_np))
        all_texts = textsA + textsB
        all_disp = dispA + dispB
        all_source = sourceA + sourceB

    # PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # K-Means
    kmeans = KMeans(n_clusters=k_clusters, random_state=42).fit(reduced_embeddings)
    kmeans_labels = kmeans.labels_

    # Hierarchical
    linked = linkage(reduced_embeddings, method='ward')
    hier_labels = fcluster(linked, hier_clusters, criterion='maxclust')

    # Assemble the result DataFrame
    df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "text": all_disp,         # Text with citation
        "source": all_source,
        "kmeans_label": kmeans_labels,
        "hier_label": hier_labels
    })

    return df, avg_similarity, similarities

# ========== 5. Streamlit main function: select Set A/B in sidebar => build each => compare ==========

def main():
    import streamlit.components.v1 as components
    html_content = """
    <div style='text-align: center; padding: 20px;'>
        <h1 style='margin-bottom: 0;'>Talmud Word Similarity Analysis</h1>

        <!-- Author -->
        <p style='font-size: 18px; margin-top: 5px;'>
            By 
            <a href="https://vivo.brown.edu/display/msatlow" target="_blank" style="text-decoration: none; color: #3366CC;">
                <strong>Michael Satlow</strong>
            </a>, 
            <a href="https://engineering.brown.edu/people/songkai-zhao" target="_blank" style="text-decoration: none; color: #3366CC;">
                <strong>Songkai Zhao</strong>
            </a>, 
            and 
            <a href="" target="_blank" style="text-decoration: none; color: #3366CC;">
                <strong>Gabriel Burstyn</strong>
            </a>
        </p>
    </div>
    <hr style='margin: 0 0 20px 0;'>
    """
    components.html(html_content, height=200)
    instructions_placeholder = st.empty()
    with instructions_placeholder:
        st.markdown("""
         This is an experimental tool that uses machine learning techniques to show word similarities in the Babylonian and Jerusalem Talmuds.  It will also allow you to compare how specific tractates use a word.  It works by first mapping phrases (the length of which is determined by the "Window") into a multidimensional matrix, then computing the distance between those occurrences, and finally sorting these occurrences into clusters based on the distances.  The parameters are explained further in the menu on the left.  You can hover over the points on the visualization to see more data.
    
    A similarity score of under 0.7 generally indicates that there is a likelihood of differing usage of the same word.  We suggest that you begin with a K-Means and Hierarchical Cluster of "2" for exploratory purposes and then adjust from there.
    
    Note that this presently works only with exact strings and you must use the Hebrew Unicode alphabet.  So, for example, אמר and שנאמר are treated as two separate words.  We are looking into how to incorporate stemmed and lemmatized words into this tool.
    
    We want to hear from you!  Let us know if you have suggestions for improving the tool or have found some interesting results.  What kinds of new research questions does this tool raise?  What would make it more useful?  If you have ideas, please send them to Michael_Satlow@Brown.edu.
    
    To start:
    
    1. Enter the word you want to analyze.
    2. Choose a window size.
    3. Select the source (Yerushalmi or Bavli) for the chosen tractate(s).
    4. Select the desired tractate(s) from the dropdown menu.
    5. Click "Compare".   
    
    The development of this tool has been supported by Brown University and the Center for Digital Scholarship at the Brown University Library. The texts have been downloaded from [Sefaria](https://github.com/sefaria) and further refined by Michael Sperling. The code for this application can be found here: [GitHub Repository](https://github.com/songkai-z/twosa).
       
                    """)

    # 1. General parameters
    st.sidebar.header("Parameters")
    target_word = st.sidebar.text_input("Target Word:", value="קרא")
    with st.sidebar:
        with st.expander("About Target Word"):
            st.markdown("""
            Provide any word from the Talmud for analysis.
                        """)
    window_size = st.sidebar.number_input("Window Size:", min_value=1, max_value=50, value=10)
    with st.sidebar:
        with st.expander("About Window Size"):
            st.markdown("""
            Window Size refers to how many words on each side of the target word are included when extracting context snippets. In this application, we compare the context around the target word across different Talmudic texts to see how usage may differ.

- Larger Window

  - Pros: Captures more surrounding context, giving richer information about the word’s usage.
  - Cons: Might include extra, unrelated text, diluting the focus on the actual word usage.
- Smaller Window
  - Pros: Ensures a sharper, more specific view of how the target word is used.
  - Cons: May lose broader contextual clues that could be important for understanding nuances or idiomatic expressions.
                        """)
    k_clusters = st.sidebar.number_input("K-Means Clusters:", min_value=1, max_value=10, value=2)
    hier_clusters = st.sidebar.number_input("Hierarchical Clusters:", min_value=1, max_value=10, value=2)
    with st.sidebar:
        with st.expander("About Clusters"):
            st.markdown("""
            The selected number represents the number of clusters the algorithm will divide the snippets into. For exploratory purposes or first time user, the default value of 2 is OK. For more details, see the "About K-Means and Hierarchical Clustering" in the results.
                            """)
    # 2. Load data and get lists of chapters
    yeru_df, bavli_df, yeru_chapters, bavli_chapters = load_data()

    # ============= Set A selection =============
    st.sidebar.write("### Set A Selection")
    corpus_A = st.sidebar.radio("Corpus for Set A:", ["Yerushalmi", "Bavli"], index=0)
    if corpus_A == "Yerushalmi":
        selected_A = st.sidebar.multiselect("Select Tractates (Set A)",
                                            options=["All"] + yeru_chapters,
                                            default=["All"])
    else:
        selected_A = st.sidebar.multiselect("Select Tractates (Set A)",
                                            options=["All"] + bavli_chapters,
                                            default=["All"])

    # ============= Set B selection =============
    st.sidebar.write("### Set B Selection")
    corpus_B = st.sidebar.radio("Corpus for Set B:", ["Yerushalmi", "Bavli"], index=1)
    if corpus_B == "Yerushalmi":
        selected_B = st.sidebar.multiselect("Select Tractates (Set B)",
                                            options=["All"] + yeru_chapters,
                                            default=["All"])
    else:
        selected_B = st.sidebar.multiselect("Select Tractates (Set B)",
                                            options=["All"] + bavli_chapters,
                                            default=["All"])
    with st.sidebar:
        with st.expander("About Set Selection"):
            st.markdown("""
            1. Select a Corpus:
- In the sidebar, under “Set A Selection,” choose either Yerushalmi or Bavli. This determines which corpus of Talmudic text will be used for Set A.

2. Pick Tractates:

- A list of available tractates (or “All”) will be shown beneath the corpus choice.
- You can select one or more tractates, or simply pick All if you want the entire corpus.
3. Repeat for Set B:
- Scroll down to “Set B Selection” and do the same: choose Yerushalmi or Bavli, then pick tractates. This can be the same corpus as Set A (for an internal comparison) or a different corpus (for a cross-corpus comparison).

4. Run the Comparison:
- Finally, click the Compare button below.
                    """)
    # 3. On "Compare" button: filter + build SetA/SetB => compare
    if st.sidebar.button("Compare"):
        instructions_placeholder.empty()
        with st.spinner("Processing..."):

            def filter_for_corpus(corpus, selected):
                if corpus == "Yerushalmi":
                    df_ = filter_chapters(yeru_df, "Tractate", selected)
                else:
                    df_ = filter_chapters(bavli_df, "Tractate", selected)
                return df_

            A_df = filter_for_corpus(corpus_A, selected_A)
            B_df = filter_for_corpus(corpus_B, selected_B)

            # Build Set A / Set B
            (
                textsA, embA, sourceA, dispA
            ) = build_set(A_df, target_word, window_size, label_for_source="Set A")

            (
                textsB, embB, sourceB, dispB
            ) = build_set(B_df, target_word, window_size, label_for_source="Set B")

            # If neither Set A nor Set B has valid texts, warn user
            if (embA is None or embA.size(0) == 0) and (embB is None or embB.size(0) == 0):
                st.warning("No valid texts found in both Set A and Set B.")
                return
            if embA is None or embA.size(0) == 0:
                st.warning("No valid texts found in Set A.")
            if embB is None or embB.size(0) == 0:
                st.warning("No valid texts found in Set B.")

            # Compare
            df, avg_sim, similarities = compare_two_sets(
                textsA, embA, sourceA, dispA,
                textsB, embB, sourceB, dispB,
                k_clusters, hier_clusters
            )

        # If the result is None => means no data from both sets
        if df is None or df.empty:
            st.warning("Comparison failed or no data to visualize.")
            return

        st.success(f"Comparison complete! Average Similarity = {avg_sim:.4f} (Set A vs. Set B)")
        with st.expander("About Average Similarity"):
            st.markdown("""
            Average similarity refers to the mean cosine similarity between all pairwise snippets (texts selected window size before and after the target word) from Set A and Set B. Specifically, for each text snippet in Set A and each text snippet in Set B, the application computes a cosine similarity score between their embeddings. It then sums all those scores and divides by the total number of A–B pairs, producing a single numeric value that reflects how similar the two sets are overall. A lower average similarity score indicates greater differences in usage. If the score falls below **0.7**, it suggests a likelihood of differing usage of the same word. In such cases, the snippets should be examined to verify and explain the nature of these differences.
            """)

        # ========== Bold the target_word within Plotly tooltips ==========
        df["hover_text"] = df["text"].str.replace(target_word, f"<b>{target_word}</b>")

        # 4. Visualizations
        # A) Similarity distribution
        if len(similarities) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(similarities, bins=10, kde=True, color="blue", ax=ax)
            ax.set_title("Distribution of Cosine Similarities (A vs B)")
            st.pyplot(fig)
        else:
            st.write("No similarity scores found.")
        with st.expander("About this Histogram"):
            st.markdown("""
            The histogram displays the distribution of cosine similarity scores between every pair of snippets from Set A and Set B. **Each bar shows how frequently a certain similarity value appears** (e.g., how many A–B pairs have a similarity in that range). This gives a quick visual overview of whether most pairs are highly similar, weakly similar, or somewhere in between.
            """)
        # B) Plotly scatter - colored by source
        st.subheader("Plotly PCA Scatter (Color by Source A/B)")
        fig_plotly_source = px.scatter(
            df,
            x="x", y="y",
            color="source",
            custom_data=["hover_text"],
            title=f"PCA Visualization (Hover for Text+Reference) - AvgSim: {avg_sim:.4f}"
        )
        fig_plotly_source.update_traces(hovertemplate="%{customdata}<extra></extra>")
        st.plotly_chart(fig_plotly_source, use_container_width=True)
        with st.expander("About PCA"):
            st.markdown("""
            PCA (Principal Component Analysis) is a dimensionality reduction technique that reduces all snippet embeddings (which is 768-dimensional) to two dimensions, then these 2D points are plotted in the graph above. **Proximity in the 2D plot usually implies higher similarity**.
            """)
        # C) Plotly scatter - colored by K-Mean
        st.subheader("Plotly PCA Scatter (Color by K-Means Label)")
        fig_kmeans = px.scatter(
            df,
            x="x", y="y",
            color=df["kmeans_label"].astype(str),
            custom_data=["hover_text"],
            title="K-Means Clustering"
        )
        fig_kmeans.update_traces(hovertemplate="%{customdata}<extra></extra>")
        st.plotly_chart(fig_kmeans, use_container_width=True)

        # E) Display K-Means cluster distribution
        st.write("### K-Means Cluster Source Distribution")
        kmeans_counts = df.groupby(["kmeans_label", "source"]).size().unstack(fill_value=0)
        st.dataframe(kmeans_counts)

        # D) Plotly scatter - colored by Hierarchical clustering
        st.subheader("Plotly PCA Scatter (Color by Hierarchical Label)")
        fig_hier = px.scatter(
            df,
            x="x", y="y",
            color=df["hier_label"].astype(str),
            custom_data=["hover_text"],
            title="Hierarchical Clustering"
        )
        fig_hier.update_traces(hovertemplate="%{customdata}<extra></extra>")
        st.plotly_chart(fig_hier, use_container_width=True)

        st.write("### Hierarchical Cluster Source Distribution")
        hier_counts = df.groupby(["hier_label", "source"]).size().unstack(fill_value=0)
        st.dataframe(hier_counts)

        with st.expander("About K-Means and Hierarchical Clustering"):
            st.markdown("""
            K-Means and Hierarchical clustering are unsupervised machine learning techniques which automatically divide the snippets into clusters. These methods are applied to the reduced embeddings (after PCA):
- **K-Means**: Partitions your snippets into *k* clusters by trying to minimize within-cluster distances.
- **Hierarchical**: Starts with each snippet as its own cluster and merges the closest pairs of clusters step by step until the desired number of clusters is reached.

By checking how snippets from Set A and Set B distribute across these clusters, you can see whether the two sets truly differ (i.e., they mostly fall into separate clusters) or if they are relatively similar (i.e., they appear in the same clusters).
            """)
        # ========== Show sample texts in each cluster at the bottom ==========
        with st.expander("Sample texts in each K-Means cluster"):
            for label in sorted(df["kmeans_label"].unique()):
                subset = df[df["kmeans_label"] == label]
                st.markdown(f"**Cluster {label}** (showing up to 5 random samples)")
                sample_subset = subset.sample(min(5, len(subset)))
                for _, row in sample_subset.iterrows():
                    highlighted_text = row["text"].replace(target_word, f"**{target_word}**")
                    st.markdown(f"- {row['source']}: {highlighted_text}")

        with st.expander("Sample texts in each Hierarchical cluster"):
            for label in sorted(df["hier_label"].unique()):
                subset = df[df["hier_label"] == label]
                st.markdown(f"**Cluster {label}** (showing up to 5 random samples)")
                sample_subset = subset.sample(min(5, len(subset)))
                for _, row in sample_subset.iterrows():
                    highlighted_text = row["text"].replace(target_word, f"**{target_word}**")
                    st.markdown(f"- {row['source']}: {highlighted_text}")
    st.markdown("""
        ---
        **Reference**:
        1. Avi Shmidman, Joshua Guedalia, Shaltiel Shmidman, Cheyn Shmuel Shmidman, Eli Handel, Moshe Koppel, "Introducing BEREL: BERT Embeddings for Rabbinic-Encoded Language", Aug 2022 [arXiv:2208.01875]  
        """)
if __name__ == "__main__":
    main()
