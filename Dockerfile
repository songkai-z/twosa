FROM continuumio/miniconda3

WORKDIR /srv

COPY requirements.txt /srv/
# install the cpu-only torch (or any other torch-related packages)
# you might modify it to install another version
RUN pip install --default-timeout=100 torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# any packages that depend on pytorch must be installed after the previous RUN command
RUN pip install -r requirements.txt --no-cache-dir

COPY . /srv

EXPOSE 80

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=80", \
            "--server.headless=true", \
            "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false", \
            "--server.enableStaticServing=true", \
            "--server.fileWatcherType=none", \
            # hide the Streamlit menu
            "--client.toolbarMode=viewer"]