# TwoSA (Talmud Word Similarity Analysis)  

## Overview  
This repository contains the source code for [TwoSA (Talmud Word Similarity Analysis)](https://twosa.us.reclaim.cloud/) — a web-based tool that uses machine learning techniques to analyze word similarities in the Babylonian and Jerusalem Talmuds.  

TwoSA allows researchers to compare how specific tractates use a given word by computing word embeddings (using [BEREL](https://huggingface.co/dicta-il/BEREL_2.0)) and clustering them based on similarity scores. It maps phrases into a multidimensional matrix, computes distances between occurrences, and sorts them into clusters using K-Means and Hierarchical Clustering algorithms. The results are visualized, enabling researchers to explore patterns in Talmudic language usage.  

## Running Locally  
1. **Pull the Docker Image:**  
   ```bash
   docker pull ghcr.io/songkai-z/twosa:latest
   ```
2. **Run the Docker Container**
   ```bash
   docker run -p 80:80 ghcr.io/songkai-z/twosa:latest
   ```
3. Access the Tool
   Open your browser and visit:
   ```
   http://localhost:80
   ```
### Start the Streamlit App Directly (Optional)
If you want to run the Streamlit app directly without Docker:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
## Development
### Code Structure
   ```
   ├── Bavli_talmud_texts.csv   # Babylonian Talmud texts
   ├── Yerushalmi.csv           # Jerusalem Talmud texts
   ├── Dockerfile               # Docker configuration for building and running the app
   ├── LICENSE                  # Licensing information (MIT License)
   ├── app.py                   # Main application logic (Streamlit-based)
   ├── requirements.txt         # Python dependencies
   ```
## Acknowledgements
The development has been supported by Brown University and the Center for Digital Scholarship at the Brown University Library. The texts have been downloaded from [Sefaria](https://github.com/sefaria) and further refined by Michael Sperling.
## Reference
1. Avi Shmidman, Joshua Guedalia, Shaltiel Shmidman, Cheyn Shmuel Shmidman, Eli Handel, Moshe Koppel, "Introducing BEREL: BERT Embeddings for Rabbinic-Encoded Language", Aug 2022 [arXiv:2208.01875]
## License
This project is licensed under the MIT License – see the [LICENSE](https://github.com/songkai-z/twosa/blob/master/LICENSE) file for details.
