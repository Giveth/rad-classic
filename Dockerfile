FROM python:3.9.1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -q --disable-pip-version-check;
CMD ["/bin/bash"]